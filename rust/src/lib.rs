//! Compiled core of spintrack: the per-pixel kernels behind trackball rotation tracking.
//!
//! Everything that loops over window pixels or map cells lives here; configuration,
//! geometry setup, the per-frame state machine and I/O stay in Python.

// Index loops read best for the dense 3x3 algebra used throughout.
#![allow(clippy::needless_range_loop)]

mod geom;
mod map;
mod solve;

use geom::{Mat3, mul3_f32, to_f32, transpose};
use map::{Map, Touched, W_MAX, W_MIN, W_SAT, box_blur};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use solve::{Level, MAX_ITER, SolveOutput};

/// Pyramid depth: the finest level plus the coarse ones the fallback solve and the global
/// search start from.
const LEVELS: usize = 3;
/// Orientations the global search scores at the coarsest level, and how many of the best
/// it refines.
const GLOBAL_CANDIDATES: usize = 2000;
const GLOBAL_TOP_K: usize = 5;

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
    /// Fraction of overlapping pixels that are inliers.
    pub inlier_frac: f64,
    /// Fraction of window pixels landing on seen map cells.
    pub overlap: f64,
    /// Gauss-Newton iterations spent (all levels).
    pub iters: u32,
    pub converged: bool,
    /// Norm (rad) of the last Gauss-Newton step, NaN if none was taken.
    pub last_step: f64,
}

impl From<SolveOutput> for SolveResult {
    fn from(out: SolveOutput) -> Self {
        SolveResult {
            w: out.w,
            r: out.r,
            cost: out.stats.cost,
            inlier_frac: out.stats.inlier_frac,
            overlap: out.stats.overlap,
            iters: out.iters,
            converged: out.converged,
            last_step: out.last_step,
        }
    }
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

fn to_pyarray<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    v: Vec<f32>,
) -> Bound<'py, PyArray2<f32>> {
    Array2::from_shape_vec(shape, v)
        .expect("shape")
        .into_pyarray(py)
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
            let num = box_blur(&masked, n, n, radius);
            let den = box_blur(valid, n, n, radius);
            let img: Vec<f32> = num
                .iter()
                .zip(&den)
                .map(|(a, d)| if *d > 1e-6 { a / d } else { 0.0 })
                .collect();
            out.push(self.index.iter().map(|&i| img[i as usize]).collect());
        }
        out
    }

    fn run_solve(
        &self,
        obs: &[f32],
        r_prev: &Mat3,
        w0: [f64; 3],
        use_prev: bool,
        n_levels: usize,
        max_iter: u32,
    ) -> SolveOutput {
        let base = if use_prev { &self.prev_map } else { &self.map };
        let n_levels = n_levels.clamp(1, LEVELS);
        // Coarse copies of the map for levels 1.. (level 0 is the map itself).
        let coarse: Vec<Map> = (1..n_levels).map(|l| base.coarse(l)).collect();
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
        solve::solve(&self.surface, &levels, r_prev, w0, max_iter)
    }

    /// Splat the remembered last frame into `prev_map` if it is stale.
    fn ensure_prev_map(&mut self) {
        if self.prev_map_valid {
            return;
        }
        self.prev_map.clear();
        if let Some(r) = self.prev_r {
            let rt = to_f32(&transpose(&r));
            for (k, &value) in self.prev_obs.iter().enumerate() {
                let (u, v) = self.prev_map.project(mul3_f32(&rt, self.surface[k]));
                self.prev_map.splat(u, v, value, 1.0, W_SAT, None);
            }
        }
        self.prev_map_valid = true;
    }
}

#[pymethods]
impl Engine {
    /// `surface`: (N, 3) unit vectors; `index`: (N,) flat row-major pixel indices in the
    /// `window_size` x `window_size` window; `face`: map cells along a cube face side;
    /// `max_pixels`: if set, the finest level uses a spatial stride so that at most about
    /// this many pixels enter the solve (all pixels still update the map).
    #[new]
    #[pyo3(signature = (surface, index, window_size, face, max_pixels=None))]
    fn new(
        surface: PyReadonlyArray2<f32>,
        index: PyReadonlyArray1<i64>,
        window_size: usize,
        face: usize,
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
        let base_stride = match max_pixels {
            Some(m) if m > 0 && index_v.len() > m => {
                ((index_v.len() as f64 / m as f64).sqrt().ceil() as usize).max(1)
            }
            _ => 1,
        };
        let subsets = (0..LEVELS)
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
            map: Map::new(face),
            prev_map: Map::new(face),
            prev_obs: Vec::new(),
            prev_r: None,
            prev_map_valid: false,
            touched: Touched::new(6 * face * face),
        })
    }

    /// Align `obs` (window_size x window_size float32, normalized) starting from
    /// `exp(w0) r_prev`, against the accumulated map or the previous-frame map, on the
    /// finest `levels` pyramid levels (all by default).
    #[pyo3(signature = (obs, r_prev, w0, use_prev=false, levels=None, max_iter=MAX_ITER))]
    // A PyO3 method's arity is its Python signature's.
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
    ) -> PyResult<SolveResult> {
        let obs_v = self.read_obs(&obs)?;
        let r_prev_m = read_mat3(&r_prev)?;
        if use_prev {
            self.ensure_prev_map();
        }
        let n_levels = levels.unwrap_or(LEVELS);
        let out = py.detach(|| self.run_solve(&obs_v, &r_prev_m, w0, use_prev, n_levels, max_iter));
        Ok(out.into())
    }

    /// Relocalize against the accumulated map: score orientations spread over SO(3) at the
    /// coarsest level, refine the best few with the full pyramid and return the refined
    /// solution with the lowest cost among those overlapping at least `min_overlap`.
    fn global_search(
        &mut self,
        py: Python<'_>,
        obs: PyReadonlyArray2<f32>,
        min_overlap: f64,
    ) -> PyResult<SolveResult> {
        let obs_v = self.read_obs(&obs)?;
        let out = py.detach(|| {
            let top = LEVELS - 1;
            let coarse = self.map.coarse(top);
            let obs_levels = self.observation_levels(&obs_v, LEVELS);
            let level = Level {
                map: &coarse,
                obs: &obs_levels[top],
                subset: &self.subsets[top],
            };
            // Deterministic quasi-uniform rotations: unit quaternions from a Halton set.
            let mut scored: Vec<(f64, f64, Mat3)> = Vec::with_capacity(GLOBAL_CANDIDATES);
            for i in 0..GLOBAL_CANDIDATES {
                let r = quat_to_mat(halton_quaternion(i as u32 + 1));
                let (_, _, stats) = solve::accumulate(&self.surface, &level, &r, f32::INFINITY);
                if stats.overlap >= min_overlap && stats.cost.is_finite() {
                    scored.push((stats.cost, stats.overlap, r));
                }
            }
            // Low-overlap candidates can have spuriously low costs: only rank the ones that
            // see most of what the best candidate sees.
            let max_overlap = scored.iter().map(|s| s.1).fold(0.0, f64::max);
            scored.retain(|s| s.1 >= 0.75 * max_overlap);
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut best: Option<SolveOutput> = None;
            for (_, _, r0) in scored.iter().take(GLOBAL_TOP_K) {
                let refined = self.run_solve(&obs_v, r0, [0.0; 3], false, LEVELS, MAX_ITER);
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
        Ok(out.into())
    }

    /// Splat `obs` into the maps at orientation `r`. The previous-frame map is replaced;
    /// the accumulated map, unless `update_main` is off, is updated, each pixel with its
    /// `weight` if given, and optionally cleared outside a small dilation of the view.
    #[pyo3(signature = (obs, r, forget_outside=false, update_main=true, weight=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        obs: PyReadonlyArray2<f32>,
        r: PyReadonlyArray2<f64>,
        forget_outside: bool,
        update_main: bool,
        weight: Option<PyReadonlyArray2<f32>>,
    ) -> PyResult<()> {
        let obs_v = self.read_obs(&obs)?;
        let weight_v = weight.as_ref().map(|w| self.read_obs(w)).transpose()?;
        let r_m = read_mat3(&r)?;
        py.detach(|| {
            // Remember this frame for the (lazily built) previous-frame map.
            self.prev_obs.clear();
            self.prev_obs
                .extend(self.index.iter().map(|&i| obs_v[i as usize]));
            self.prev_r = Some(r_m);
            self.prev_map_valid = false;
            if !update_main {
                return;
            }
            self.touched.clear();
            let rt = to_f32(&transpose(&r_m));
            for (k, &i) in self.index.iter().enumerate() {
                let (u, v) = self.map.project(mul3_f32(&rt, self.surface[k]));
                // Touched cells are only needed to forget the rest of the map.
                let touched = forget_outside.then_some(&mut self.touched);
                let w = weight_v.as_ref().map_or(1.0, |w| w[i as usize]);
                if w > 0.0 {
                    self.map.splat(u, v, obs_v[i as usize], w, W_MAX, touched);
                }
            }
            if forget_outside {
                self.map.forget_outside(&self.touched.mask);
            }
        });
        Ok(())
    }

    fn reset(&mut self) {
        self.map.clear();
        self.prev_map.clear();
        self.prev_r = None;
        self.prev_obs.clear();
        self.prev_map_valid = false;
    }

    fn map_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        to_pyarray(py, (6 * self.map.n, self.map.n), self.map.mean.clone())
    }

    fn map_weight<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        to_pyarray(py, (6 * self.map.n, self.map.n), self.map.weight.clone())
    }

    /// Render the accumulated map into the window at orientation `r`: the model the
    /// residual is taken against. Returns `(value, confidence)`, both `window_size` x
    /// `window_size` float32 and zero where the map has not been seen.
    #[allow(clippy::type_complexity)]
    fn render<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray2<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        let r_m = read_mat3(&r)?;
        let n = self.n;
        let (value, conf) = py.detach(|| {
            let mut value = vec![0.0f32; n * n];
            let mut conf = vec![0.0f32; n * n];
            let rt = to_f32(&transpose(&r_m));
            for (k, &i) in self.index.iter().enumerate() {
                let (u, v) = self.map.project(mul3_f32(&rt, self.surface[k]));
                if let Some(s) = self.map.sample(u, v) {
                    value[i as usize] = s.value;
                    conf[i as usize] = s.confidence;
                }
            }
            (value, conf)
        });
        Ok((to_pyarray(py, (n, n), value), to_pyarray(py, (n, n), conf)))
    }

    /// Replace the accumulated map (e.g. with a saved template).
    fn set_map(
        &mut self,
        mean: PyReadonlyArray2<f32>,
        weight: PyReadonlyArray2<f32>,
    ) -> PyResult<()> {
        let m = mean.as_array();
        let w = weight.as_array();
        let shape = [6 * self.map.n, self.map.n];
        if m.shape() != shape || w.shape() != shape {
            return Err(PyValueError::new_err("map arrays must be (6 face, face)"));
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
    // The solver constants the Python state machine reasons with.
    m.add("W_MIN", W_MIN)?;
    m.add("MAX_ITER", MAX_ITER)?;
    m.add("TOL", solve::TOL)?;
    Ok(())
}
