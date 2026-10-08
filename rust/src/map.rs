//! The ball surface map: an equi-angular cubemap of normalized intensities with
//! confidences.
//!
//! Six square faces of `n x n` cells are stacked into one row-major `(6 n, n)` array in the
//! order `-x, +z, +x, -z, +y, -y`, addressed by a continuous `(u, v)`: cell `(i, j)` is
//! centered at `(j + 0.5, i + 0.5)`. Face coordinates are equi-angular
//! (`s' = tan(pi s / 4)`), which keeps every cell close to square and the solid angle per
//! cell within 1.41:1 over the sphere. Faces meet at seams, which `cube_cell` resolves by
//! re-projecting the direction of the cell it was asked for.

use std::f32::consts::PI;

/// Weight below which a cell counts as unseen.
pub const W_MIN: f32 = 0.1;
/// Weight at which a cell counts as fully trusted (sample confidence 1).
pub const W_SAT: f32 = 3.0;
/// Cap on the accumulated map's weights, so that it keeps adapting.
pub const W_MAX: f32 = 50.0;
/// Cells kept around the current view when forgetting the rest of the map.
pub const FORGET_MARGIN: usize = 1;

/// Cube faces as `(forward, right, up)`, in the order they are stacked. Mirrored in
/// `python/spintrack/maps.py`, which has to agree with this cell for cell.
const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
    ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
];

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The face `p` belongs to: the one whose `forward` it is closest to.
#[inline]
fn face_of(p: [f32; 3]) -> usize {
    let (ax, ay, az) = (p[0].abs(), p[1].abs(), p[2].abs());
    if ax >= ay && ax >= az {
        if p[0] > 0.0 { 2 } else { 0 }
    } else if ay >= az {
        if p[1] > 0.0 { 4 } else { 5 }
    } else if p[2] > 0.0 {
        1
    } else {
        3
    }
}

/// Direction at gnomonic face coordinates `(s, t)`; not normalized, every use is a ratio.
#[inline]
fn face_direction(face: usize, s: f32, t: f32) -> [f32; 3] {
    let (f, r, u) = FACES[face];
    [
        f[0] + s * r[0] + t * u[0],
        f[1] + s * r[1] + t * u[1],
        f[2] + s * r[2] + t * u[2],
    ]
}

/// Equi-angular face coordinate from the gnomonic one, and back.
///
/// `s` is always in `[-1, 1]` on the way in, since a direction lies on the face it is
/// closest to, which is exactly the range `atan_unit` covers.
#[inline]
fn eac(s: f32) -> f32 {
    (4.0 / PI) * atan_unit(s)
}

#[inline]
fn eac_inv(s: f32) -> f32 {
    (0.25 * PI * s).tan()
}

/// Polynomial `atan` on `[-1, 1]` (max error ~1e-5 rad), several times faster than the
/// libm call. Splatting and sampling both go through it, so its error cancels.
#[inline]
fn atan_unit(z: f32) -> f32 {
    let a = z.abs();
    let z2 = a * a;
    let r = a
        * (0.999_977_26
            + z2 * (-0.332_623_47
                + z2 * (0.193_543_46
                    + z2 * (-0.116_432_87 + z2 * (0.052_653_32 + z2 * (-0.011_721_2))))));
    if z < 0.0 { -r } else { r }
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
    /// Cells along a face side; the arrays are `(6 n, n)`.
    pub n: usize,
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
    pub fn new(n: usize) -> Map {
        Map {
            n,
            mean: vec![0.0; 6 * n * n],
            weight: vec![0.0; 6 * n * n],
        }
    }

    pub fn clear(&mut self) {
        self.mean.fill(0.0);
        self.weight.fill(0.0);
    }

    /// Continuous map coordinates of a unit direction.
    #[inline]
    pub fn project(&self, p: [f32; 3]) -> (f32, f32) {
        let face = face_of(p);
        let (f, r, up) = FACES[face];
        let d = dot(p, f);
        let n = self.n as f32;
        let s = eac(dot(p, r) / d);
        let t = eac(dot(p, up) / d);
        (0.5 * (s + 1.0) * n, (face as f32) * n + 0.5 * (1.0 - t) * n)
    }

    /// Gradient of `(u, v)` with respect to the direction `p` (unit), rows `du/dp`, `dv/dp`.
    /// Bounded everywhere: `p` lies on the face it is closest to, so
    /// `dot(p, forward) >= 1 / sqrt(3)`.
    #[inline]
    pub fn projection_jacobian(&self, p: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let face = face_of(p);
        let (f, r, up) = FACES[face];
        let d = dot(p, f);
        let (s, t) = (dot(p, r) / d, dot(p, up) / d);
        let k = 0.5 * (self.n as f32) * (4.0 / PI) / d;
        let ku = k / (1.0 + s * s);
        let kv = -k / (1.0 + t * t);
        let du = [
            ku * (r[0] - s * f[0]),
            ku * (r[1] - s * f[1]),
            ku * (r[2] - s * f[2]),
        ];
        let dv = [
            kv * (up[0] - t * f[0]),
            kv * (up[1] - t * f[1]),
            kv * (up[2] - t * f[2]),
        ];
        (du, dv)
    }

    /// Flat index of cell `(col, row)` of `face`, crossing the seam when either runs off
    /// the face.
    ///
    /// Off the face it re-projects the direction of the requested cell rather than
    /// consulting an edge table, so it cannot disagree with `project`. At a cube corner the
    /// fourth neighbor does not exist and this lands on one of the other three.
    #[inline]
    fn cube_cell(&self, face: usize, col: i32, row: i32) -> usize {
        let n = self.n as i32;
        if col >= 0 && col < n && row >= 0 && row < n {
            return (face * self.n + row as usize) * self.n + col as usize;
        }
        let size = n as f32;
        let s = eac_inv(2.0 * (col as f32 + 0.5) / size - 1.0);
        let t = eac_inv(1.0 - 2.0 * (row as f32 + 0.5) / size);
        let (u, v) = self.project(face_direction(face, s, t));
        let f2 = ((v as i32) / n).clamp(0, 5);
        let c2 = (u as i32).clamp(0, n - 1);
        let r2 = (v as i32 - f2 * n).clamp(0, n - 1);
        ((f2 * n + r2) * n + c2) as usize
    }

    /// Face, top-left tap `(col, row)` and fractional position of a bilinear read at
    /// `(u, v)`.
    #[inline]
    fn locate(&self, u: f32, v: f32) -> (usize, i32, i32, f32, f32) {
        let n = self.n as i32;
        let face = ((v as i32) / n).clamp(0, 5);
        let uc = u - 0.5;
        let vc = v - 0.5 - (face * n) as f32;
        let (u0, v0) = (uc.floor(), vc.floor());
        (face as usize, u0 as i32, v0 as i32, uc - u0, vc - v0)
    }

    /// The four cells a bilinear read at `(u, v)` draws on, and its fractional position:
    /// `[(u0, v0), (u1, v0), (u0, v1), (u1, v1)]`.
    #[inline]
    fn cell_indices(&self, u: f32, v: f32) -> ([usize; 4], f32, f32) {
        let (face, c0, r0, fu, fv) = self.locate(u, v);
        (
            [
                self.cube_cell(face, c0, r0),
                self.cube_cell(face, c0 + 1, r0),
                self.cube_cell(face, c0, r0 + 1),
                self.cube_cell(face, c0 + 1, r0 + 1),
            ],
            fu,
            fv,
        )
    }

    /// The 4x4 cells around a bilinear read at `(u, v)`, rows then columns, in the
    /// coordinates of the face the read lands on, and the read's fractional position
    /// within the middle 2x2.
    ///
    /// Gradients are differenced within this patch, so a cell across a seam is
    /// differenced along the reading face's axes rather than its own, rotated ones.
    #[inline]
    fn patch(&self, u: f32, v: f32) -> ([usize; 16], f32, f32) {
        let (face, c0, r0, fu, fv) = self.locate(u, v);
        let (c0, r0, n) = (c0 - 1, r0 - 1, self.n as i32);
        let mut out = [0usize; 16];
        if c0 >= 0 && c0 + 3 < n && r0 >= 0 && r0 + 3 < n {
            let base = (face * self.n + r0 as usize) * self.n + c0 as usize;
            for i in 0..4 {
                for j in 0..4 {
                    out[4 * i + j] = base + i * self.n + j;
                }
            }
        } else {
            for i in 0..4 {
                for j in 0..4 {
                    out[4 * i + j] = self.cube_cell(face, c0 + j as i32, r0 + i as i32);
                }
            }
        }
        (out, fu, fv)
    }

    /// Bilinear sample with a smooth gradient; `None` unless all four neighboring cells
    /// are seen. The gradient interpolates central differences of those cells (one-sided
    /// at unseen neighbors), which keeps the alignment Jacobian continuous across cells.
    #[inline]
    pub fn sample(&self, u: f32, v: f32) -> Option<Sample> {
        let (p, fu, fv) = self.patch(u, v);
        let (w00, w10, w01, w11) = (
            self.weight[p[5]],
            self.weight[p[6]],
            self.weight[p[9]],
            self.weight[p[10]],
        );
        if w00 < W_MIN || w10 < W_MIN || w01 < W_MIN || w11 < W_MIN {
            return None;
        }
        let (m00, m10, m01, m11) = (
            self.mean[p[5]],
            self.mean[p[6]],
            self.mean[p[9]],
            self.mean[p[10]],
        );
        let top = m00 + fu * (m10 - m00);
        let bot = m01 + fu * (m11 - m01);
        let value = top + fv * (bot - top);
        let seen = |k: usize, m: f32| {
            if self.weight[p[k]] >= W_MIN {
                self.mean[p[k]]
            } else {
                m
            }
        };
        let diff = |a: f32, b: f32, m: f32| {
            if a == m || b == m {
                b - a
            } else {
                0.5 * (b - a)
            }
        };
        let grad = |k: usize, m: f32| {
            (
                diff(seen(k - 1, m), seen(k + 1, m), m),
                diff(seen(k - 4, m), seen(k + 4, m), m),
            )
        };
        let (gx00, gy00) = grad(5, m00);
        let (gx10, gy10) = grad(6, m10);
        let (gx01, gy01) = grad(9, m01);
        let (gx11, gy11) = grad(10, m11);
        let gxt = gx00 + fu * (gx10 - gx00);
        let gxb = gx01 + fu * (gx11 - gx01);
        let gyt = gy00 + fu * (gy10 - gy00);
        let gyb = gy01 + fu * (gy11 - gy01);
        let du = gxt + fv * (gxb - gxt);
        let dv = gyt + fv * (gyb - gyt);
        let wt = w00 + fu * (w10 - w00);
        let wb = w01 + fu * (w11 - w01);
        let confidence = ((wt + fv * (wb - wt)) / W_SAT).min(1.0);
        Some(Sample {
            value,
            du,
            dv,
            confidence,
        })
    }

    /// Add one observation of weight `w` at `(u, v)` with a bilinear footprint, capping
    /// the cell weights at `w_max`; marks the cells in `touched` if given.
    #[inline]
    pub fn splat(
        &mut self,
        u: f32,
        v: f32,
        value: f32,
        w: f32,
        w_max: f32,
        mut touched: Option<&mut Touched>,
    ) {
        let ([i00, i10, i01, i11], fu, fv) = self.cell_indices(u, v);
        let cells = [
            (i00, w * (1.0 - fu) * (1.0 - fv)),
            (i10, w * fu * (1.0 - fv)),
            (i01, w * (1.0 - fu) * fv),
            (i11, w * fu * fv),
        ];
        for (idx, b) in cells {
            if b <= 0.0 {
                continue;
            }
            let w_new = self.weight[idx] + b;
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
        // Faces are blurred one at a time, clamped at their edges: the pyramid only widens
        // the basin of attraction, which a one-cell artifact along the seams does not hurt.
        let sum_wm = self.per_face(&wm, |src, n| box_blur(src, n, n, radius));
        let sum_w = self.per_face(&self.weight, |src, n| box_blur(src, n, n, radius));
        let n2 = (self.n / factor).max(1);
        let mut out = Map::new(n2);
        for y in 0..6 * n2 {
            for x in 0..n2 {
                // The center of each decimated block, row taken within its face.
                let yy = (y / n2) * self.n + ((y % n2) * factor + factor / 2).min(self.n - 1);
                let xx = (x * factor + factor / 2).min(self.n - 1);
                let i = yy * self.n + xx;
                let o = y * n2 + x;
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

    /// Apply a whole-image operation to each face in turn.
    fn per_face<T: Copy + Default, F: Fn(&[T], usize) -> Vec<T>>(
        &self,
        src: &[T],
        op: F,
    ) -> Vec<T> {
        let n = self.n;
        let mut out = vec![T::default(); src.len()];
        for f in 0..6 {
            let span = f * n * n..(f + 1) * n * n;
            out[span.clone()].copy_from_slice(&op(&src[span], n));
        }
        out
    }

    /// Drop everything outside a `FORGET_MARGIN`-cell dilation of the `touched` cells.
    pub fn forget_outside(&mut self, touched: &[u8]) {
        let keep = self.per_face(touched, |src, n| dilate(src, n, n, FORGET_MARGIN));
        for (i, k) in keep.iter().enumerate() {
            if *k == 0 {
                self.weight[i] = 0.0;
                self.mean[i] = 0.0;
            }
        }
    }
}

/// Mean over a `(2r+1)^2` window, clamped at the edges.
pub fn box_blur(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let r = radius as i64;
    let mut tmp = vec![0.0f32; w * h];
    let norm = 1.0 / ((2 * r + 1) as f32);
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out = &mut tmp[y * w..(y + 1) * w];
        for x in 0..w as i64 {
            let mut s = 0.0f32;
            for d in -r..=r {
                s += row[(x + d).clamp(0, w as i64 - 1) as usize];
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

/// Binary dilation with a square structuring element, clamped at the edges.
pub fn dilate(mask: &[u8], w: usize, h: usize, margin: usize) -> Vec<u8> {
    let r = margin as i64;
    let mut out = vec![0u8; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut hit = 0u8;
            'outer: for dy in -r..=r {
                let yy = (y + dy).clamp(0, h as i64 - 1) as usize;
                for dx in -r..=r {
                    let xx = (x + dx).clamp(0, w as i64 - 1) as usize;
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
    use std::f32::consts::FRAC_PI_2;

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    /// Direction of the point at map coordinates `(u, v)`; the inverse of `project`.
    fn unproject(m: &Map, u: f32, v: f32) -> [f32; 3] {
        let n = m.n as f32;
        let face = ((v / n) as usize).min(5);
        let s = eac_inv(2.0 * u / n - 1.0);
        let t = eac_inv(1.0 - 2.0 * (v - face as f32 * n) / n);
        unit(face_direction(face, s, t))
    }

    /// Deterministic pseudo-random numbers in `[-1, 1)`.
    fn lcg(mut seed: u32) -> impl FnMut() -> f32 {
        move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16777216.0 * 2.0 - 1.0
        }
    }

    #[test]
    fn splat_then_sample_recovers_value() {
        let mut m = Map::new(32);
        for du in [-0.4f32, 0.0, 0.4] {
            for dv in [-0.4f32, 0.0, 0.4] {
                m.splat(10.5 + du, 7.5 + dv, 2.0, 1.0, 100.0, None);
            }
        }
        let s = m.sample(10.5, 7.5).expect("seen");
        assert!((s.value - 2.0).abs() < 1e-5);
        assert!(m.sample(25.0, 7.5).is_none());
    }

    /// The Jacobian must be the derivative of the projection, on every face.
    fn check_jacobian(m: &Map, raw: [f32; 3]) {
        let p = unit(raw);
        let (u, v) = m.project(p);
        let (du, dv) = m.projection_jacobian(p);
        let eps = 1e-3f32;
        for k in 0..3 {
            // Finite difference along the tangent direction e_k - (e_k . p) p.
            let mut t = [0.0f32; 3];
            t[k] = 1.0;
            let d = dot(t, p);
            for i in 0..3 {
                t[i] -= d * p[i];
            }
            let q = unit([p[0] + eps * t[0], p[1] + eps * t[1], p[2] + eps * t[2]]);
            let (u2, v2) = m.project(q);
            let du_t = dot(du, t);
            let dv_t = dot(dv, t);
            assert!(
                ((u2 - u) / eps - du_t).abs() < 0.05 * du_t.abs().max(1.0),
                "du[{k}] at {p:?}"
            );
            assert!(
                ((v2 - v) / eps - dv_t).abs() < 0.05 * dv_t.abs().max(1.0),
                "dv[{k}] at {p:?}"
            );
        }
    }

    #[test]
    fn projection_is_consistent_with_jacobian() {
        let m = Map::new(64);
        // The center of each face, an edge, a corner, and a point either side of a seam.
        for face in 0..6 {
            check_jacobian(&m, FACES[face].0);
            check_jacobian(&m, face_direction(face, 0.4, -0.7));
            check_jacobian(&m, face_direction(face, 0.97, 0.93));
        }
    }

    #[test]
    fn projection_round_trips() {
        let m = Map::new(48);
        let mut next = lcg(12345);
        for _ in 0..2000 {
            let p = unit([next(), next(), next()]);
            let (u, v) = m.project(p);
            assert!((0.0..m.n as f32).contains(&u), "u {u}");
            assert!((0.0..(6 * m.n) as f32).contains(&v), "v {v}");
            let q = unproject(&m, u, v);
            assert!(dot(p, q) > 1.0 - 1e-6, "{p:?} -> {q:?}");
        }
    }

    #[test]
    fn seams_join_neighboring_faces() {
        let n = 32;
        let m = Map::new(n);
        // A texel at the face center spans pi/2 / n; the seam neighbor of an edge cell
        // must be its actual neighbor on the sphere, not a clamp back onto the same cell.
        let limit = 1.5 * FRAC_PI_2 / n as f32;
        for face in 0..6 {
            for k in 0..n as i32 {
                let last = n as i32 - 1;
                for (col, row, dcol, drow) in [
                    (0, k, -1, 0),
                    (last, k, 1, 0),
                    (k, 0, 0, -1),
                    (k, last, 0, 1),
                ] {
                    let idx = (face * n + row as usize) * n + col as usize;
                    let other = m.cube_cell(face, col + dcol, row + drow);
                    assert_ne!(other, idx, "face {face} cell ({col}, {row})");
                    let a = unproject(&m, col as f32 + 0.5, (face * n) as f32 + row as f32 + 0.5);
                    let b = unproject(&m, (other % n) as f32 + 0.5, (other / n) as f32 + 0.5);
                    let gap = dot(a, b).clamp(-1.0, 1.0).acos();
                    assert!(gap < limit, "face {face} ({col}, {row}) gap {gap}");
                }
            }
        }
    }

    #[test]
    fn splat_conserves_weight_over_a_seam() {
        for (u, v) in [
            (8.5f32, 8.5f32),
            (0.0, 8.5),
            (16.0, 8.5),
            (8.5, 16.0),
            (0.0, 0.0),
        ] {
            let mut m = Map::new(16);
            m.splat(u, v, 1.0, 1.0, 1e6, None);
            let total: f32 = m.weight.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "({u}, {v}) deposited {total}");
        }
    }

    /// On a texture linear in the direction, the sampled gradient must match the true
    /// tangent gradient where the taps straddle a seam too, whose neighboring face has
    /// rotated axes.
    #[test]
    fn gradient_is_right_across_seams() {
        let mut m = Map::new(104);
        let a = [0.3f32, -0.8, 0.5];
        for idx in 0..m.mean.len() {
            let (row, col) = (idx / m.n, idx % m.n);
            m.mean[idx] = dot(a, unproject(&m, col as f32 + 0.5, row as f32 + 0.5));
            m.weight[idx] = 1.0;
        }
        let mut next = lcg(777);
        let mut errors = Vec::new();
        while errors.len() < 2000 {
            let p = unit([next(), next(), next()]);
            let (u, v) = m.project(p);
            let (cells, _, _) = m.cell_indices(u, v);
            let ap = dot(a, p);
            let t = [a[0] - ap * p[0], a[1] - ap * p[1], a[2] - ap * p[2]];
            let t_norm = dot(t, t).sqrt();
            if cells.iter().all(|&c| c / (m.n * m.n) == face_of(p)) || t_norm < 0.2 {
                continue;
            }
            let s = m.sample(u, v).expect("seen");
            let (du, dv) = m.projection_jacobian(p);
            let g: Vec<f32> = (0..3).map(|i| s.du * du[i] + s.dv * dv[i]).collect();
            let gp = dot([g[0], g[1], g[2]], p);
            let e: Vec<f32> = (0..3).map(|i| g[i] - gp * p[i] - t[i]).collect();
            errors.push(dot([e[0], e[1], e[2]], [e[0], e[1], e[2]]).sqrt() / t_norm);
        }
        errors.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let (median, max) = (errors[errors.len() / 2], errors[errors.len() - 1]);
        // Differencing each tap along its own face's axes fails this by a wide margin.
        assert!(median < 0.12 && max < 0.4, "median {median}, max {max}");
    }
}
