//! The ball surface map: an equal-area grid of normalized intensities with confidences.
//!
//! Map coordinates: `u = W * (atan2(px, pz) + pi) / (2 pi)` wraps around in longitude and
//! `v = H * (1 - py) / 2` runs from the +y pole (top) to the -y pole (bottom). Cell `(i, j)`
//! is centred at `(j + 0.5, i + 0.5)`.

use std::f32::consts::{FRAC_PI_2, PI};

/// Polynomial `atan2` (max error ~1e-5 rad). Used for both splatting and sampling, so any
/// approximation error cancels out; it is several times faster than the libm call.
#[inline]
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    let swap = ay > ax;
    let (num, den) = if swap { (ax, ay) } else { (ay, ax) };
    let z = if den > 0.0 { num / den } else { 0.0 };
    let z2 = z * z;
    let mut a = z
        * (0.999_977_26
            + z2 * (-0.332_623_47
                + z2 * (0.193_543_46
                    + z2 * (-0.116_432_87 + z2 * (0.052_653_32 + z2 * (-0.011_721_2))))));
    if swap {
        a = FRAC_PI_2 - a;
    }
    if x < 0.0 {
        a = PI - a;
    }
    if y < 0.0 { -a } else { a }
}

/// Cells touched by the current frame's splat: a mask plus the list of indices, so that
/// per-frame bookkeeping stays proportional to the view rather than to the whole map.
pub struct Touched {
    pub mask: Vec<u8>,
    pub list: Vec<u32>,
}

impl Touched {
    pub fn new(n: usize) -> Touched {
        Touched {
            mask: vec![0; n],
            list: Vec::new(),
        }
    }

    #[inline]
    pub fn mark(&mut self, idx: usize) {
        if self.mask[idx] == 0 {
            self.mask[idx] = 1;
            self.list.push(idx as u32);
        }
    }

    pub fn clear(&mut self) {
        for &i in &self.list {
            self.mask[i as usize] = 0;
        }
        self.list.clear();
    }
}

#[derive(Clone)]
pub struct Map {
    pub w: usize,
    pub h: usize,
    pub mean: Vec<f32>,
    pub weight: Vec<f32>,
}

pub struct Sample {
    pub value: f32,
    pub du: f32,
    pub dv: f32,
    pub confidence: f32,
}

impl Map {
    pub fn new(w: usize, h: usize) -> Map {
        Map {
            w,
            h,
            mean: vec![0.0; w * h],
            weight: vec![0.0; w * h],
        }
    }

    pub fn clear(&mut self) {
        self.mean.fill(0.0);
        self.weight.fill(0.0);
    }

    /// Continuous map coordinates of a unit direction.
    #[inline]
    pub fn project(&self, p: [f32; 3]) -> (f32, f32) {
        let lon = fast_atan2(p[0], p[2]);
        let u = (lon + PI) * (self.w as f32) / (2.0 * PI);
        let v = (1.0 - p[1]) * 0.5 * (self.h as f32);
        (u, v)
    }

    /// Gradient of `(u, v)` with respect to the direction `p` (unit), rows `du/dp`, `dv/dp`.
    #[inline]
    pub fn projection_jacobian(&self, p: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let rxz = (p[0] * p[0] + p[2] * p[2]).max(1e-12);
        let ku = self.w as f32 / (2.0 * PI) / rxz;
        let du = [ku * p[2], 0.0, -ku * p[0]];
        let dv = [0.0, -0.5 * self.h as f32, 0.0];
        (du, dv)
    }

    #[inline]
    fn cell_indices(&self, u: f32, v: f32) -> (usize, usize, usize, usize, f32, f32) {
        let uc = u - 0.5;
        let vc = v - 0.5;
        let u0f = uc.floor();
        let v0f = vc.floor();
        let fu = uc - u0f;
        let fv = vc - v0f;
        let w = self.w as i32;
        let h = self.h as i32;
        // u lies in [0, W], so u0 is in [-1, W]: one conditional wrap each side suffices.
        let mut u0 = u0f as i32;
        if u0 < 0 {
            u0 += w;
        } else if u0 >= w {
            u0 -= w;
        }
        let mut u1 = u0 + 1;
        if u1 >= w {
            u1 -= w;
        }
        let v0 = (v0f as i32).clamp(0, h - 1) as usize;
        let v1 = (v0f as i32 + 1).clamp(0, h - 1) as usize;
        (u0 as usize, u1 as usize, v0, v1, fu, fv)
    }

    /// Central-difference gradient of the mean at cell `(x, y)`, one-sided at unseen
    /// neighbours; the cell itself is known to be seen.
    #[inline]
    fn cell_gradient(&self, x: usize, y: usize, w_min: f32) -> (f32, f32) {
        let w = self.w;
        let i = y * w + x;
        let xl = y * w + if x == 0 { w - 1 } else { x - 1 };
        let xr = y * w + if x + 1 == w { 0 } else { x + 1 };
        let m = self.mean[i];
        let ml = if self.weight[xl] >= w_min {
            self.mean[xl]
        } else {
            m
        };
        let mr = if self.weight[xr] >= w_min {
            self.mean[xr]
        } else {
            m
        };
        let gx = if ml == m || mr == m {
            mr - ml
        } else {
            0.5 * (mr - ml)
        };
        let mu = if y > 0 && self.weight[i - w] >= w_min {
            self.mean[i - w]
        } else {
            m
        };
        let md = if y + 1 < self.h && self.weight[i + w] >= w_min {
            self.mean[i + w]
        } else {
            m
        };
        let gy = if mu == m || md == m {
            md - mu
        } else {
            0.5 * (md - mu)
        };
        (gx, gy)
    }

    /// Bilinear sample with a smooth gradient; `None` unless all four neighbouring cells
    /// have weight >= `w_min`. The gradient interpolates central differences of the
    /// neighbouring cells, which keeps the alignment Jacobian continuous across cells.
    #[inline]
    pub fn sample(&self, u: f32, v: f32, w_min: f32, w_sat: f32) -> Option<Sample> {
        let (u0, u1, v0, v1, fu, fv) = self.cell_indices(u, v);
        let i00 = v0 * self.w + u0;
        let i10 = v0 * self.w + u1;
        let i01 = v1 * self.w + u0;
        let i11 = v1 * self.w + u1;
        let (w00, w10, w01, w11) = (
            self.weight[i00],
            self.weight[i10],
            self.weight[i01],
            self.weight[i11],
        );
        if w00 < w_min || w10 < w_min || w01 < w_min || w11 < w_min {
            return None;
        }
        let (m00, m10, m01, m11) = (
            self.mean[i00],
            self.mean[i10],
            self.mean[i01],
            self.mean[i11],
        );
        let top = m00 + fu * (m10 - m00);
        let bot = m01 + fu * (m11 - m01);
        let value = top + fv * (bot - top);
        let (gx00, gy00) = self.cell_gradient(u0, v0, w_min);
        let (gx10, gy10) = self.cell_gradient(u1, v0, w_min);
        let (gx01, gy01) = self.cell_gradient(u0, v1, w_min);
        let (gx11, gy11) = self.cell_gradient(u1, v1, w_min);
        let gxt = gx00 + fu * (gx10 - gx00);
        let gxb = gx01 + fu * (gx11 - gx01);
        let gyt = gy00 + fu * (gy10 - gy00);
        let gyb = gy01 + fu * (gy11 - gy01);
        let du = gxt + fv * (gxb - gxt);
        let dv = gyt + fv * (gyb - gyt);
        let wt = w00 + fu * (w10 - w00);
        let wb = w01 + fu * (w11 - w01);
        let confidence = ((wt + fv * (wb - wt)) / w_sat).min(1.0);
        Some(Sample {
            value,
            du,
            dv,
            confidence,
        })
    }

    /// Add one observation at `(u, v)` with bilinear footprint, decaying old evidence by
    /// `lambda` and capping the weight at `w_max`; marks the cells in `touched` if given.
    #[inline]
    pub fn splat(
        &mut self,
        u: f32,
        v: f32,
        value: f32,
        lambda: f32,
        w_max: f32,
        touched: Option<&mut Touched>,
    ) {
        let (u0, u1, v0, v1, fu, fv) = self.cell_indices(u, v);
        let cells = [
            (v0 * self.w + u0, (1.0 - fu) * (1.0 - fv)),
            (v0 * self.w + u1, fu * (1.0 - fv)),
            (v1 * self.w + u0, (1.0 - fu) * fv),
            (v1 * self.w + u1, fu * fv),
        ];
        let mut touched = touched;
        for (idx, b) in cells {
            if b <= 0.0 {
                continue;
            }
            let w_old = self.weight[idx] * lambda;
            let w_new = w_old + b;
            self.mean[idx] += b * (value - self.mean[idx]) / w_new;
            self.weight[idx] = w_new.min(w_max);
            if let Some(t) = touched.as_deref_mut() {
                t.mark(idx);
            }
        }
    }

    /// Coarse pyramid level: weight-aware box blur of radius `2^level - 1` followed by
    /// decimation by `2^level`. The observation is blurred with the same box at the same
    /// level, so both sides of the alignment see the same smoothing.
    pub fn coarse(&self, level: usize) -> Map {
        let factor = 1usize << level;
        let radius = factor - 1;
        let wm: Vec<f32> = self
            .mean
            .iter()
            .zip(&self.weight)
            .map(|(m, w)| m * w)
            .collect();
        let sum_wm = box_blur(&wm, self.w, self.h, radius, true);
        let sum_w = box_blur(&self.weight, self.w, self.h, radius, true);
        let (w2, h2) = ((self.w / factor).max(1), (self.h / factor).max(1));
        let mut out = Map::new(w2, h2);
        for y in 0..h2 {
            for x in 0..w2 {
                // Sample the blurred field at the centre of each decimated block.
                let yy = (y * factor + factor / 2).min(self.h - 1);
                let xx = (x * factor + factor / 2).min(self.w - 1);
                let i = yy * self.w + xx;
                let o = y * w2 + x;
                out.weight[o] = sum_w[i];
                out.mean[o] = if sum_w[i] > 1e-6 {
                    sum_wm[i] / sum_w[i]
                } else {
                    0.0
                };
            }
        }
        out
    }

    /// Drop everything outside a `margin`-cell dilation of the `touched` cells.
    pub fn forget_outside(&mut self, touched: &[u8], margin: usize) {
        let keep = dilate(touched, self.w, self.h, margin);
        for (i, k) in keep.iter().enumerate() {
            if *k == 0 {
                self.weight[i] = 0.0;
                self.mean[i] = 0.0;
            }
        }
    }
}

/// Mean over a `(2r+1)^2` window; longitude wraps when `wrap_u`, rows are clamped.
pub fn box_blur(src: &[f32], w: usize, h: usize, radius: usize, wrap_u: bool) -> Vec<f32> {
    let r = radius as i64;
    let mut tmp = vec![0.0f32; w * h];
    let norm = 1.0 / ((2 * r + 1) as f32);
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out = &mut tmp[y * w..(y + 1) * w];
        for x in 0..w as i64 {
            let mut s = 0.0f32;
            for d in -r..=r {
                let xx = if wrap_u {
                    (x + d).rem_euclid(w as i64)
                } else {
                    (x + d).clamp(0, w as i64 - 1)
                };
                s += row[xx as usize];
            }
            out[x as usize] = s * norm;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h as i64 {
        for x in 0..w {
            let mut s = 0.0f32;
            for d in -r..=r {
                let yy = (y + d).clamp(0, h as i64 - 1) as usize;
                s += tmp[yy * w + x];
            }
            out[y as usize * w + x] = s * norm;
        }
    }
    out
}

/// Binary dilation with a square structuring element, wrapping in u and clamping in v.
pub fn dilate(mask: &[u8], w: usize, h: usize, margin: usize) -> Vec<u8> {
    let r = margin as i64;
    let mut out = vec![0u8; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut hit = 0u8;
            'outer: for dy in -r..=r {
                let yy = (y + dy).clamp(0, h as i64 - 1) as usize;
                for dx in -r..=r {
                    let xx = (x + dx).rem_euclid(w as i64) as usize;
                    if mask[yy * w + xx] != 0 {
                        hit = 1;
                        break 'outer;
                    }
                }
            }
            out[y as usize * w + x as usize] = hit;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splat_then_sample_recovers_value() {
        let mut m = Map::new(64, 32);
        for du in [-0.4f32, 0.0, 0.4] {
            for dv in [-0.4f32, 0.0, 0.4] {
                m.splat(10.5 + du, 7.5 + dv, 2.0, 1.0, 100.0, None);
            }
        }
        let s = m.sample(10.5, 7.5, 0.1, 1.0).expect("seen");
        assert!((s.value - 2.0).abs() < 1e-5);
        assert!(m.sample(40.0, 7.5, 0.1, 1.0).is_none());
    }

    #[test]
    fn fast_atan2_is_accurate() {
        for &(y, x) in &[
            (0.3f32, 0.9f32),
            (-0.7, 0.2),
            (0.5, -0.5),
            (-0.1, -0.99),
            (1.0, 0.0),
            (0.0, -1.0),
        ] {
            assert!((fast_atan2(y, x) - y.atan2(x)).abs() < 2e-5, "{y} {x}");
        }
    }

    #[test]
    fn projection_is_consistent_with_jacobian() {
        let m = Map::new(360, 180);
        let raw = [0.3f32, 0.2, 0.9];
        let n = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        let p = [raw[0] / n, raw[1] / n, raw[2] / n];
        let (u, v) = m.project(p);
        let (du, dv) = m.projection_jacobian(p);
        let eps = 1e-3f32;
        for k in 0..3 {
            // Finite difference along the tangent direction e_k - (e_k . p) p.
            let mut t = [0.0f32; 3];
            t[k] = 1.0;
            let dot = t[0] * p[0] + t[1] * p[1] + t[2] * p[2];
            for i in 0..3 {
                t[i] -= dot * p[i];
            }
            let q = [p[0] + eps * t[0], p[1] + eps * t[1], p[2] + eps * t[2]];
            let (u2, v2) = m.project(q);
            let du_t = du[0] * t[0] + du[1] * t[1] + du[2] * t[2];
            let dv_t = dv[0] * t[0] + dv[1] * t[1] + dv[2] * t[2];
            assert!(
                ((u2 - u) / eps - du_t).abs() < 0.05 * du_t.abs().max(1.0),
                "du[{k}]"
            );
            assert!(
                ((v2 - v) / eps - dv_t).abs() < 0.05 * dv_t.abs().max(1.0),
                "dv[{k}]"
            );
        }
    }
}
