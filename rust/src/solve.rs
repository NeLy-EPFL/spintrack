//! Iteratively reweighted Gauss-Newton alignment of a normalized window against the map.
//!
//! The unknown is the ball orientation `R` (window frame -> map frame is `R^T`). Each
//! iteration solves for a left-multiplicative increment `R <- exp([d]x) R` from the
//! linearized photometric residuals `obs_k - M(project(R^T v_k))`.

use crate::geom::{
    Mat3, Vec3, cross, exp_so3, log_so3, mat_mul, mul3_f32, norm, solve_spd3, to_f32, transpose,
};
use crate::map::Map;

/// Gauss-Newton iterations at the finest level.
pub const MAX_ITER: u32 = 10;
/// Gauss-Newton iterations at each coarser level, which only has to get close.
pub const COARSE_MAX_ITER: u32 = 3;
/// Step norm (rad) below which the finest level has converged.
pub const TOL: f64 = 3e-4;
/// Tolerance multiplier per coarser level.
const LEVEL_TOL_FACTOR: f64 = 4.0;
/// Huber and Tukey constants, in units of the robust residual scale.
const HUBER: f32 = 1.345;
const TUKEY: f32 = 4.685;
/// Levenberg-Marquardt damping that keeps a near-degenerate system solvable.
const DAMPING: f64 = 1e-6;

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
    /// Norm (rad) of the last Gauss-Newton step, NaN if none was taken.
    pub last_step: f64,
}

/// One pass over a level: normal equations, gradient and residual statistics at `r_cur`.
///
/// `scale` is the residual scale the robust constants are multiplied by; an infinite
/// scale disables robust weighting.
pub fn accumulate(
    surface: &[[f32; 3]],
    level: &Level,
    r_cur: &Mat3,
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
    let huber = HUBER * scale;
    let tukey = TUKEY * scale;
    for &k in level.subset {
        let v = surface[k as usize];
        let pp = mul3_f32(&rt, v);
        let (u, vv) = map.project(pp);
        let Some(s) = map.sample(u, vv) else {
            continue;
        };
        n_overlap += 1;
        let res = level.obs[k as usize] - s.value;
        let a = res.abs();
        let mut rho = if a <= huber { 1.0 } else { huber / a };
        if a < tukey {
            let t = 1.0 - (a / tukey) * (a / tukey);
            rho *= t * t;
            n_inlier += 1;
            sum_sq_in += (res * res) as f64;
        } else {
            rho = 0.0;
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
        let gc = mul3_f32(&r, g);
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

/// Coarse-to-fine Gauss-Newton from the initial guess `exp(w0) r_prev`, with at most
/// `max_iter` iterations at the finest level (at least one).
pub fn solve(
    surface: &[[f32; 3]],
    levels: &[Level],
    r_prev: &Mat3,
    w0: Vec3,
    max_iter: u32,
) -> SolveOutput {
    let mut r_cur = mat_mul(&exp_so3(w0), r_prev);
    let mut iters = 0u32;
    let mut converged = false;
    let mut last_step = f64::NAN;
    let n_levels = levels.len();
    // Statistics of the last pass at the finest level; its step is below tolerance when
    // converged, so they describe the solution closely enough.
    let mut stats = Stats::default();
    for (li, level) in levels.iter().enumerate() {
        let depth = (n_levels - 1 - li) as i32; // 0 at the finest level
        let tol = TOL * LEVEL_TOL_FACTOR.powi(depth);
        let max_iter = if depth == 0 {
            max_iter.max(1)
        } else {
            COARSE_MAX_ITER
        };
        converged = false;
        // Residual scale for the robust weights: unknown (no weighting) on the first pass,
        // then the RMS of the previous pass's inlier residuals.
        let mut scale = f32::INFINITY;
        for _ in 0..max_iter {
            let (h, b, pass) = accumulate(surface, level, &r_cur, scale);
            if pass.rms.is_finite() {
                scale = (pass.rms as f32).max(1e-3);
            }
            iters += 1;
            if depth == 0 {
                stats = pass;
            }
            if pass.n_overlap < 10 {
                break;
            }
            let Some(delta) = solve_spd3(&h, [-b[0], -b[1], -b[2]], DAMPING) else {
                break;
            };
            r_cur = mat_mul(&exp_so3(delta), &r_cur);
            last_step = norm(delta);
            if last_step < tol {
                converged = true;
                break;
            }
        }
    }
    let w = log_so3(&mat_mul(&r_cur, &transpose(r_prev)));
    SolveOutput {
        r: r_cur,
        w,
        stats,
        iters,
        converged,
        last_step,
    }
}
