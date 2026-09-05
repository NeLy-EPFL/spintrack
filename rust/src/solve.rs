//! Iteratively reweighted Gauss-Newton alignment of a normalized window against the map.
//!
//! The unknown is the ball orientation `R` (window frame -> map frame is `R^T`). Each
//! iteration solves for a left-multiplicative increment `R <- exp([d]x) R` from the
//! linearised photometric residuals `obs_k - M(project(R^T v_k))`.

use crate::geom::{Mat3, Vec3, cross, exp_so3, log_so3, mat_mul, norm, solve_spd3, transpose};
use crate::map::Map;

pub struct SolveParams {
    pub max_iter: u32,
    pub tol: f64,
    /// Huber and Tukey constants in units of the robust residual scale (sigma).
    pub huber: f32,
    pub tukey: f32,
    pub w_min: f32,
    pub w_sat: f32,
    pub damping: f64,
    /// Tolerance multiplier per coarser level (coarse levels stop earlier).
    pub level_tol_factor: f64,
    pub coarse_max_iter: u32,
    /// Iterations (per level) during which the robust weights are recomputed; afterwards
    /// they are frozen so that Gauss-Newton converges instead of chattering.
    pub reweight_iters: u32,
}

pub struct Level<'a> {
    pub map: &'a Map,
    pub obs: &'a [f32],
    pub subset: &'a [u32],
}

#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub cost: f64,
    pub rms: f64,
    pub inlier_frac: f64,
    pub overlap: f64,
    pub n_overlap: usize,
}

pub struct SolveOutput {
    pub r: Mat3,
    pub w: Vec3,
    pub stats: Stats,
    pub iters: u32,
    pub converged: bool,
    pub hessian: Mat3,
    /// Norm of every Gauss-Newton step taken, in order (diagnostics).
    pub steps: Vec<f64>,
}

#[inline]
fn to_f32(m: &Mat3) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[i][j] as f32;
        }
    }
    out
}

#[inline]
fn mul3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// One pass over a level: normal equations, gradient and residual statistics at `r_cur`.
///
/// `weights` holds one robust weight per subset pixel; it is recomputed from the residuals
/// when `reweight` is set and reused otherwise.
pub fn accumulate(
    surface: &[[f32; 3]],
    level: &Level,
    r_cur: &Mat3,
    p: &SolveParams,
    weights: &mut [f32],
    reweight: bool,
    scale: f32,
) -> (Mat3, Vec3, Stats) {
    let r = to_f32(r_cur);
    let rt = to_f32(&transpose(r_cur));
    let map = level.map;
    let mut h = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    let mut cost = 0.0f64;
    let mut sum_w = 0.0f64;
    let mut sum_sq_in = 0.0f64;
    let mut n_overlap = 0usize;
    let mut n_inlier = 0usize;
    // `scale` is the residual scale (sigma) multiplying the robust constants; an infinite
    // scale disables robust weighting.
    let huber = p.huber * scale;
    let tukey = if p.tukey > 0.0 { p.tukey * scale } else { 0.0 };
    let inlier_cut = if tukey > 0.0 { tukey } else { 2.0 * huber };
    for (idx, &k) in level.subset.iter().enumerate() {
        // Once the weights are frozen, so is the pixel set: pixels that were unseen or
        // rejected stay out, which keeps the iteration from chattering at the map frontier.
        if !reweight && weights[idx] <= 0.0 {
            continue;
        }
        let v = surface[k as usize];
        let pp = mul3(&rt, v);
        let (u, vv) = map.project(pp);
        let Some(s) = map.sample(u, vv, p.w_min, p.w_sat) else {
            if reweight {
                weights[idx] = 0.0;
            }
            continue;
        };
        n_overlap += 1;
        let res = level.obs[k as usize] - s.value;
        let a = res.abs();
        if reweight {
            let mut rho = if a <= huber { 1.0 } else { huber / a };
            if tukey > 0.0 {
                if a < tukey {
                    let t = 1.0 - (a / tukey) * (a / tukey);
                    rho *= t * t;
                } else {
                    rho = 0.0;
                }
            }
            weights[idx] = rho;
        }
        let rho = weights[idx];
        if a < inlier_cut {
            n_inlier += 1;
            sum_sq_in += (res * res) as f64;
        }
        let wgt = rho * s.confidence;
        if wgt <= 0.0 {
            continue;
        }
        let (du, dv) = map.projection_jacobian(pp);
        let g = [
            s.du * du[0] + s.dv * dv[0],
            s.du * du[1] + s.dv * dv[1],
            s.du * du[2] + s.dv * dv[2],
        ];
        let gc = mul3(&r, g);
        let c = cross(
            [gc[0] as f64, gc[1] as f64, gc[2] as f64],
            [v[0] as f64, v[1] as f64, v[2] as f64],
        );
        let j = [-c[0], -c[1], -c[2]];
        let wgt = wgt as f64;
        let res = res as f64;
        for i in 0..3 {
            b[i] += wgt * j[i] * res;
            for jj in i..3 {
                h[i][jj] += wgt * j[i] * j[jj];
            }
        }
        cost += wgt * res * res;
        sum_w += wgt;
    }
    for i in 0..3 {
        for jj in 0..i {
            h[i][jj] = h[jj][i];
        }
    }
    let stats = Stats {
        cost: if sum_w > 0.0 { cost / sum_w } else { f64::NAN },
        rms: if n_inlier > 0 {
            (sum_sq_in / n_inlier as f64).sqrt()
        } else {
            f64::NAN
        },
        inlier_frac: if n_overlap > 0 {
            n_inlier as f64 / n_overlap as f64
        } else {
            0.0
        },
        overlap: n_overlap as f64 / level.subset.len().max(1) as f64,
        n_overlap,
    };
    (h, b, stats)
}

/// Coarse-to-fine Gauss-Newton from the initial guess `exp(w0) r_prev`.
pub fn solve(
    surface: &[[f32; 3]],
    levels: &[Level],
    r_prev: &Mat3,
    w0: Vec3,
    p: &SolveParams,
) -> SolveOutput {
    let mut r_cur = mat_mul(&exp_so3(w0), r_prev);
    let mut iters = 0u32;
    let mut converged = false;
    let mut steps = Vec::new();
    let n_levels = levels.len();
    let mut weights: Vec<f32> = Vec::new();
    // Normal matrix and statistics from the last pass of the finest level; the final step
    // is below tolerance, so they describe the solution closely enough.
    let mut last: Option<(Mat3, Stats)> = None;
    for (li, level) in levels.iter().enumerate() {
        let depth = (n_levels - 1 - li) as i32; // 0 at the finest level
        let tol = p.tol * p.level_tol_factor.powi(depth);
        let max_iter = if depth == 0 {
            p.max_iter
        } else {
            p.coarse_max_iter
        };
        weights.clear();
        weights.resize(level.subset.len(), 1.0);
        converged = false;
        // Residual scale for the robust weights: unknown (no weighting) on the first pass,
        // then the RMS of the previous pass's inlier residuals.
        let mut scale = f32::INFINITY;
        for it in 0..max_iter {
            let reweight = it < p.reweight_iters;
            let (h, b, stats) =
                accumulate(surface, level, &r_cur, p, &mut weights, reweight, scale);
            if reweight && stats.rms.is_finite() {
                scale = (stats.rms as f32).max(1e-3);
            }
            iters += 1;
            if depth == 0 {
                last = Some((h, stats));
            }
            if stats.n_overlap < 10 {
                break;
            }
            let Some(delta) = solve_spd3(&h, [-b[0], -b[1], -b[2]], p.damping) else {
                break;
            };
            r_cur = mat_mul(&exp_so3(delta), &r_cur);
            steps.push(norm(delta));
            if norm(delta) < tol {
                converged = true;
                break;
            }
        }
    }
    let (hessian, stats) = match last {
        Some(v) => v,
        None => {
            let finest = levels.last().expect("at least one level");
            weights.clear();
            weights.resize(finest.subset.len(), 1.0);
            let (h, _, s) = accumulate(
                surface,
                finest,
                &r_cur,
                p,
                &mut weights,
                true,
                f32::INFINITY,
            );
            (h, s)
        }
    };
    let w = log_so3(&mat_mul(&r_cur, &transpose(r_prev)));
    SolveOutput {
        r: r_cur,
        w,
        stats,
        iters,
        converged,
        hessian,
        steps,
    }
}
