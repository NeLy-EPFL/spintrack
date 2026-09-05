//! Compiled core of spintrack: the per-pixel kernels behind trackball rotation tracking.
//!
//! Everything that loops over window pixels or map cells lives here; configuration,
//! geometry setup, the per-frame state machine and I/O stay in Python.

// Index loops read best for the dense 3x3 algebra used throughout.
#![allow(clippy::needless_range_loop)]

mod geom;
mod map;
mod solve;

use geom::Mat3;
use map::{Map, Touched, box_blur};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use solve::{Level, SolveParams};

/// Version of the compiled core, taken from `Cargo.toml`.
#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Result of one rotation solve.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct SolveResult {
    /// Rotation vector of the increment `R R_prev^T` (window frame), radians.
    pub w: [f64; 3],
    /// New ball orientation (3x3, row-major nested lists).
    pub r: [[f64; 3]; 3],
    /// Weighted mean squared residual (normalized-intensity units).
    pub cost: f64,
    /// RMS residual over inliers.
    pub rms: f64,
    /// Fraction of overlapping pixels that are inliers.
    pub inlier_frac: f64,
    /// Fraction of window pixels landing on seen map cells.
    pub overlap: f64,
    /// Gauss-Newton iterations spent (all levels).
    pub iters: u32,
    pub converged: bool,
    /// Weighted normal matrix at the solution (for covariance estimates).
    pub hessian: [[f64; 3]; 3],
    /// Norm (rad) of each Gauss-Newton step, in order.
    pub steps: Vec<f64>,
}

fn read_mat3(arr: &PyReadonlyArray2<f64>) -> PyResult<Mat3> {
    let a = arr.as_array();
    if a.shape() != [3, 3] {
        return Err(PyValueError::new_err("expected a 3x3 matrix"));
    }
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[[i, j]];
        }
    }
    Ok(m)
}

/// Per-configuration tracking engine: window geometry plus the surface maps.
#[pyclass]
pub struct Engine {
    n: usize,
    surface: Vec<[f32; 3]>,
    index: Vec<u32>,
    valid_f32: Vec<f32>,
    subsets: Vec<Vec<u32>>,
    map: Map,
    /// Map holding only the last accepted frame, built on demand for the fallback solve.
    prev_map: Map,
    prev_obs: Vec<f32>,
    prev_r: Option<Mat3>,
    prev_map_valid: bool,
    touched: Touched,
}

impl Engine {
    fn read_obs(&self, obs: &PyReadonlyArray2<f32>) -> PyResult<Vec<f32>> {
        let a = obs.as_array();
        if a.shape() != [self.n, self.n] {
            return Err(PyValueError::new_err(format!(
                "observation must be {}x{}",
                self.n, self.n
            )));
        }
        Ok(a.iter().copied().collect())
    }

    /// Per-level observation values indexed by pixel `k` (blurred over valid pixels).
    fn observation_levels(&self, obs: &[f32], n_levels: usize) -> Vec<Vec<f32>> {
        let n = self.n;
        let level0: Vec<f32> = self.index.iter().map(|&i| obs[i as usize]).collect();
        if n_levels == 1 {
            return vec![level0];
        }
        let valid = &self.valid_f32;
        let masked: Vec<f32> = obs
            .iter()
            .zip(valid)
            .map(|(o, v)| if *v > 0.0 { *o } else { 0.0 })
            .collect();
        let mut out = vec![level0];
        for level in 1..n_levels {
            let radius = (1usize << level) - 1;
            let num = box_blur(&masked, n, n, radius, false);
            let den = box_blur(valid, n, n, radius, false);
            let img: Vec<f32> = num
                .iter()
                .zip(&den)
                .map(|(a, d)| if *d > 1e-6 { a / d } else { 0.0 })
                .collect();
            out.push(self.index.iter().map(|&i| img[i as usize]).collect());
        }
        out
    }

    /// Coarse copies of the map for levels 1.. (level 0 is the map itself).
    fn coarse_maps(map: &Map, n_levels: usize) -> Vec<Map> {
        (1..n_levels).map(|l| map.coarse(l)).collect()
    }

    fn run_solve(
        &self,
        obs: &[f32],
        r_prev: &Mat3,
        w0: [f64; 3],
        use_prev: bool,
        params: &SolveParams,
        n_levels: usize,
    ) -> solve::SolveOutput {
        let base = if use_prev { &self.prev_map } else { &self.map };
        let n_levels = n_levels.clamp(1, self.subsets.len());
        let coarse = Self::coarse_maps(base, n_levels);
        let obs_levels = self.observation_levels(obs, n_levels);
        // Coarse to fine: highest level index first.
        let levels: Vec<Level> = (0..n_levels)
            .rev()
            .map(|l| Level {
                map: if l == 0 { base } else { &coarse[l - 1] },
                obs: &obs_levels[l],
                subset: &self.subsets[l],
            })
            .collect();
        solve::solve(&self.surface, &levels, r_prev, w0, params)
    }
}

#[pymethods]
impl Engine {
    /// `surface`: (N, 3) unit vectors; `index`: (N,) flat row-major pixel indices in the
    /// `window_size` x `window_size` window; `levels`: pyramid depth (1 = no pyramid);
    /// `max_pixels`: if set, the finest level uses a spatial stride so that at most about
    /// this many pixels enter the solve (all pixels still update the map).
    #[new]
    #[pyo3(signature = (surface, index, window_size, map_w, map_h, levels=3, max_pixels=None))]
    fn new(
        surface: PyReadonlyArray2<f32>,
        index: PyReadonlyArray1<i64>,
        window_size: usize,
        map_w: usize,
        map_h: usize,
        levels: usize,
        max_pixels: Option<usize>,
    ) -> PyResult<Self> {
        let s = surface.as_array();
        let idx = index.as_array();
        if s.shape()[1] != 3 || s.shape()[0] != idx.len() {
            return Err(PyValueError::new_err(
                "surface must be (N, 3) matching index (N,)",
            ));
        }
        let n = window_size;
        let mut valid = vec![0u8; n * n];
        let mut index_v = Vec::with_capacity(idx.len());
        for &i in idx.iter() {
            if i < 0 || i as usize >= n * n {
                return Err(PyValueError::new_err("index out of range"));
            }
            valid[i as usize] = 1;
            index_v.push(i as u32);
        }
        let surface_v: Vec<[f32; 3]> = (0..s.shape()[0])
            .map(|k| [s[[k, 0]], s[[k, 1]], s[[k, 2]]])
            .collect();
        let levels = levels.max(1);
        let base_stride = match max_pixels {
            Some(m) if m > 0 && index_v.len() > m => {
                ((index_v.len() as f64 / m as f64).sqrt().ceil() as usize).max(1)
            }
            _ => 1,
        };
        let subsets = (0..levels)
            .map(|l| {
                let stride = (1usize << l).max(base_stride);
                index_v
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| {
                        let i = **i as usize;
                        (i / n) % stride == 0 && (i % n) % stride == 0
                    })
                    .map(|(k, _)| k as u32)
                    .collect()
            })
            .collect();
        let valid_f32 = valid.iter().map(|&v| v as f32).collect();
        Ok(Engine {
            n,
            surface: surface_v,
            index: index_v,
            valid_f32,
            subsets,
            map: Map::new(map_w, map_h),
            prev_map: Map::new(map_w, map_h),
            prev_obs: Vec::new(),
            prev_r: None,
            prev_map_valid: false,
            touched: Touched::new(map_w * map_h),
        })
    }

    #[getter]
    fn n_valid(&self) -> usize {
        self.index.len()
    }

    #[getter]
    fn levels(&self) -> usize {
        self.subsets.len()
    }

    /// Align `obs` (window_size x window_size float32, normalized) starting from
    /// `exp(w0) r_prev`, against the accumulated map or the previous-frame map.
    #[pyo3(signature = (obs, r_prev, w0, use_prev=false, levels=None, max_iter=10, tol=3e-4,
                        huber=1.345, tukey=4.685, w_min=0.1, w_sat=3.0, damping=1e-6,
                        level_tol_factor=4.0, coarse_max_iter=3, reweight_iters=1000))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        py: Python<'_>,
        obs: PyReadonlyArray2<f32>,
        r_prev: PyReadonlyArray2<f64>,
        w0: [f64; 3],
        use_prev: bool,
        levels: Option<usize>,
        max_iter: u32,
        tol: f64,
        huber: f32,
        tukey: f32,
        w_min: f32,
        w_sat: f32,
        damping: f64,
        level_tol_factor: f64,
        coarse_max_iter: u32,
        reweight_iters: u32,
    ) -> PyResult<SolveResult> {
        let obs_v = self.read_obs(&obs)?;
        let r_prev_m = read_mat3(&r_prev)?;
        let params = SolveParams {
            max_iter,
            tol,
            huber,
            tukey,
            w_min,
            w_sat,
            damping,
            level_tol_factor,
            coarse_max_iter,
            reweight_iters,
        };
        let n_levels = levels.unwrap_or(self.subsets.len());
        if use_prev {
            self.ensure_prev_map(w_sat.max(1.0));
        }
        let out = py.detach(|| self.run_solve(&obs_v, &r_prev_m, w0, use_prev, &params, n_levels));
        Ok(SolveResult {
            w: out.w,
            r: out.r,
            cost: out.stats.cost,
            rms: out.stats.rms,
            inlier_frac: out.stats.inlier_frac,
            overlap: out.stats.overlap,
            iters: out.iters,
            converged: out.converged,
            hessian: out.hessian,
            steps: out.steps,
        })
    }

    /// Relocalise against the accumulated map: evaluate `n_candidates` orientations spread
    /// over SO(3) at the coarsest level, refine the `top_k` best with the full pyramid and
    /// return the refined solution with the lowest cost (ties broken by overlap).
    #[pyo3(signature = (obs, n_candidates=2000, top_k=5, max_iter=10, tol=3e-4, huber=1.345,
                        tukey=4.685, w_min=0.1, w_sat=3.0, min_overlap=0.3))]
    #[allow(clippy::too_many_arguments)]
    fn global_search(
        &mut self,
        py: Python<'_>,
        obs: PyReadonlyArray2<f32>,
        n_candidates: usize,
        top_k: usize,
        max_iter: u32,
        tol: f64,
        huber: f32,
        tukey: f32,
        w_min: f32,
        w_sat: f32,
        min_overlap: f64,
    ) -> PyResult<SolveResult> {
        let obs_v = self.read_obs(&obs)?;
        let params = SolveParams {
            max_iter,
            tol,
            huber,
            tukey,
            w_min,
            w_sat,
            damping: 1e-6,
            level_tol_factor: 4.0,
            coarse_max_iter: 3,
            reweight_iters: 1000,
        };
        let out = py.detach(|| {
            let n_levels = self.subsets.len();
            let coarse = Self::coarse_maps(&self.map, n_levels);
            let obs_levels = self.observation_levels(&obs_v, n_levels);
            let top = n_levels - 1;
            let level = Level {
                map: if top == 0 {
                    &self.map
                } else {
                    &coarse[top - 1]
                },
                obs: &obs_levels[top],
                subset: &self.subsets[top],
            };
            // Deterministic quasi-uniform rotations: unit quaternions from a Halton-like set.
            let mut scored: Vec<(f64, f64, Mat3)> = Vec::with_capacity(n_candidates);
            let mut weights = vec![1.0f32; level.subset.len()];
            for i in 0..n_candidates {
                let q = halton_quaternion(i as u32 + 1);
                let r = quat_to_mat(q);
                let (_, _, stats) = solve::accumulate(
                    &self.surface,
                    &level,
                    &r,
                    &params,
                    &mut weights,
                    true,
                    f32::INFINITY,
                );
                if stats.overlap >= min_overlap && stats.cost.is_finite() {
                    scored.push((stats.cost, stats.overlap, r));
                }
            }
            // Low-overlap candidates can have spuriously low costs: only rank the ones that
            // see most of what the best candidate sees.
            let max_overlap = scored.iter().map(|s| s.1).fold(0.0, f64::max);
            scored.retain(|s| s.1 >= 0.75 * max_overlap);
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut best: Option<solve::SolveOutput> = None;
            for (_, _, r0) in scored.iter().take(top_k) {
                let refined = self.run_solve(&obs_v, r0, [0.0; 3], false, &params, n_levels);
                let better = match &best {
                    None => true,
                    Some(b) => {
                        refined.stats.cost < b.stats.cost && refined.stats.overlap >= min_overlap
                    }
                };
                if better {
                    best = Some(refined);
                }
            }
            best
        });
        let out = out.ok_or_else(|| PyValueError::new_err("global search found no candidate"))?;
        Ok(SolveResult {
            w: out.w,
            r: out.r,
            cost: out.stats.cost,
            rms: out.stats.rms,
            inlier_frac: out.stats.inlier_frac,
            overlap: out.stats.overlap,
            iters: out.iters,
            converged: out.converged,
            hessian: out.hessian,
            steps: out.steps,
        })
    }

    /// Photometric cost and overlap of `obs` at orientation `r` (no optimisation).
    #[pyo3(signature = (obs, r, use_prev=false, level=0, huber=1.345, tukey=4.685, w_min=0.1, w_sat=3.0))]
    #[allow(clippy::too_many_arguments)]
    fn cost(
        &mut self,
        obs: PyReadonlyArray2<f32>,
        r: PyReadonlyArray2<f64>,
        use_prev: bool,
        level: usize,
        huber: f32,
        tukey: f32,
        w_min: f32,
        w_sat: f32,
    ) -> PyResult<(f64, f64)> {
        let obs_v = self.read_obs(&obs)?;
        let r_m = read_mat3(&r)?;
        if use_prev {
            self.ensure_prev_map(w_sat.max(1.0));
        }
        let base = if use_prev { &self.prev_map } else { &self.map };
        let level = level.min(self.subsets.len() - 1);
        let map_l = if level == 0 {
            base.clone()
        } else {
            Self::coarse_maps(base, level + 1)
                .pop()
                .expect("coarse level")
        };
        let obs_levels = self.observation_levels(&obs_v, level + 1);
        let params = SolveParams {
            max_iter: 0,
            tol: 0.0,
            huber,
            tukey,
            w_min,
            w_sat,
            damping: 0.0,
            level_tol_factor: 1.0,
            coarse_max_iter: 0,
            reweight_iters: 1,
        };
        let lvl = Level {
            map: &map_l,
            obs: &obs_levels[level],
            subset: &self.subsets[level],
        };
        let mut weights = vec![1.0f32; lvl.subset.len()];
        let (_, _, stats) = solve::accumulate(
            &self.surface,
            &lvl,
            &r_m,
            &params,
            &mut weights,
            true,
            f32::INFINITY,
        );
        Ok((stats.cost, stats.overlap))
    }

    /// Normal equations `(H, b, cost, overlap)` of the finest level at orientation `r`,
    /// for diagnostics and tests (numerical gradient checks).
    #[pyo3(signature = (obs, r, use_prev=false, huber=1.345, tukey=4.685, w_min=0.1, w_sat=3.0))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn normal_equations(
        &mut self,
        obs: PyReadonlyArray2<f32>,
        r: PyReadonlyArray2<f64>,
        use_prev: bool,
        huber: f32,
        tukey: f32,
        w_min: f32,
        w_sat: f32,
    ) -> PyResult<([[f64; 3]; 3], [f64; 3], f64, f64)> {
        let obs_v = self.read_obs(&obs)?;
        let r_m = read_mat3(&r)?;
        if use_prev {
            self.ensure_prev_map(w_sat.max(1.0));
        }
        let base = if use_prev { &self.prev_map } else { &self.map };
        let obs_levels = self.observation_levels(&obs_v, 1);
        let params = SolveParams {
            max_iter: 0,
            tol: 0.0,
            huber,
            tukey,
            w_min,
            w_sat,
            damping: 0.0,
            level_tol_factor: 1.0,
            coarse_max_iter: 0,
            reweight_iters: 1,
        };
        let lvl = Level {
            map: base,
            obs: &obs_levels[0],
            subset: &self.subsets[0],
        };
        let mut weights = vec![1.0f32; lvl.subset.len()];
        let (h, b, stats) = solve::accumulate(
            &self.surface,
            &lvl,
            &r_m,
            &params,
            &mut weights,
            true,
            f32::INFINITY,
        );
        Ok((h, b, stats.cost, stats.overlap))
    }

    /// Splat `obs` into the maps at orientation `r`. The previous-frame map is replaced;
    /// the accumulated map is updated with forgetting factor `lambda_` (1 = none) and
    /// optionally cleared outside a `margin`-cell dilation of the current view.
    #[pyo3(signature = (obs, r, lambda_=1.0, w_max=50.0, forget_outside=false, margin=1, update_main=true))]
    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        py: Python<'_>,
        obs: PyReadonlyArray2<f32>,
        r: PyReadonlyArray2<f64>,
        lambda_: f32,
        w_max: f32,
        forget_outside: bool,
        margin: usize,
        update_main: bool,
    ) -> PyResult<()> {
        let obs_v = self.read_obs(&obs)?;
        let r_m = read_mat3(&r)?;
        py.detach(|| {
            let rt = geom::transpose(&r_m);
            let mut rt32 = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    rt32[i][j] = rt[i][j] as f32;
                }
            }
            // Remember this frame for the (lazily built) previous-frame map.
            self.prev_obs.clear();
            self.prev_obs
                .extend(self.index.iter().map(|&i| obs_v[i as usize]));
            self.prev_r = Some(r_m);
            self.prev_map_valid = false;
            if update_main {
                self.touched.clear();
                for (k, &i) in self.index.iter().enumerate() {
                    let v = self.surface[k];
                    let p = [
                        rt32[0][0] * v[0] + rt32[0][1] * v[1] + rt32[0][2] * v[2],
                        rt32[1][0] * v[0] + rt32[1][1] * v[1] + rt32[1][2] * v[2],
                        rt32[2][0] * v[0] + rt32[2][1] * v[1] + rt32[2][2] * v[2],
                    ];
                    let (u, vv) = self.map.project(p);
                    // Touched cells are only needed to forget the rest of the map.
                    let touched = if forget_outside {
                        Some(&mut self.touched)
                    } else {
                        None
                    };
                    self.map
                        .splat(u, vv, obs_v[i as usize], lambda_, w_max, touched);
                }
                if forget_outside {
                    self.map.forget_outside(&self.touched.mask, margin);
                }
            }
        });
        Ok(())
    }

    /// Splat the remembered last frame into `prev_map` if it is stale.
    fn ensure_prev_map(&mut self, w_max: f32) {
        if self.prev_map_valid {
            return;
        }
        self.prev_map.clear();
        if let Some(r) = self.prev_r {
            let rt = geom::transpose(&r);
            for (k, &value) in self.prev_obs.iter().enumerate() {
                let v = self.surface[k];
                let p = [
                    (rt[0][0] as f32) * v[0] + (rt[0][1] as f32) * v[1] + (rt[0][2] as f32) * v[2],
                    (rt[1][0] as f32) * v[0] + (rt[1][1] as f32) * v[1] + (rt[1][2] as f32) * v[2],
                    (rt[2][0] as f32) * v[0] + (rt[2][1] as f32) * v[1] + (rt[2][2] as f32) * v[2],
                ];
                let (u, vv) = self.prev_map.project(p);
                self.prev_map.splat(u, vv, value, 1.0, w_max, None);
            }
        }
        self.prev_map_valid = true;
    }

    fn reset(&mut self) {
        self.map.clear();
        self.prev_map.clear();
        self.prev_r = None;
        self.prev_obs.clear();
        self.prev_map_valid = false;
    }

    fn map_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        Array2::from_shape_vec((self.map.h, self.map.w), self.map.mean.clone())
            .expect("shape")
            .into_pyarray(py)
    }

    fn map_weight<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        Array2::from_shape_vec((self.map.h, self.map.w), self.map.weight.clone())
            .expect("shape")
            .into_pyarray(py)
    }

    /// Replace the accumulated map (e.g. with a saved template).
    fn set_map(
        &mut self,
        mean: PyReadonlyArray2<f32>,
        weight: PyReadonlyArray2<f32>,
    ) -> PyResult<()> {
        let m = mean.as_array();
        let w = weight.as_array();
        if m.shape() != [self.map.h, self.map.w] || w.shape() != [self.map.h, self.map.w] {
            return Err(PyValueError::new_err("map arrays must be (map_h, map_w)"));
        }
        self.map.mean = m.iter().copied().collect();
        self.map.weight = w.iter().copied().collect();
        Ok(())
    }
}

/// Radical-inverse (van der Corput) sequence in the given base.
fn radical_inverse(mut i: u32, base: u32) -> f64 {
    let mut f = 1.0 / base as f64;
    let mut r = 0.0;
    while i > 0 {
        r += f * (i % base) as f64;
        i /= base;
        f /= base as f64;
    }
    r
}

/// Uniformly distributed unit quaternion from a 3-D Halton point (Shoemake's method).
fn halton_quaternion(i: u32) -> [f64; 4] {
    let u1 = radical_inverse(i, 2);
    let u2 = radical_inverse(i, 3);
    let u3 = radical_inverse(i, 5);
    let a = (1.0 - u1).sqrt();
    let b = u1.sqrt();
    let t2 = 2.0 * std::f64::consts::PI * u2;
    let t3 = 2.0 * std::f64::consts::PI * u3;
    [a * t2.sin(), a * t2.cos(), b * t3.sin(), b * t3.cos()]
}

fn quat_to_mat(q: [f64; 4]) -> Mat3 {
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_class::<Engine>()?;
    m.add_class::<SolveResult>()?;
    Ok(())
}
