//! The ball surface map: a grid of normalized intensities with confidences.
//!
//! Two tessellations, both addressed by a continuous `(u, v)` into one row-major `(H, W)`
//! array, so everything above this file sees the same thing. Cell `(i, j)` is centred at
//! `(j + 0.5, i + 0.5)`.
//!
//! `EqualArea`: `u = W * (atan2(px, pz) + pi) / (2 pi)` wraps around in longitude and
//! `v = H * (1 - py) / 2` runs from the +y pole (top) to the -y pole (bottom).
//!
//! `Cube`: six square faces stacked into a `(6 n, n)` array, `v` running through them in
//! the order `-x, +z, +x, -z, +y, -y`, `u` and the within-face part of `v` equi-angular
//! across the face. Faces meet at seams rather than wrapping, which `cube_cell` resolves
//! by re-projecting the direction of the cell it was asked for.

use std::f32::consts::{FRAC_PI_2, PI};

/// How the map tiles the sphere.
///
/// `EqualArea` gives every cell the same solid angle but not the same shape: cell extents
/// are `(2 pi / W) cos(lat)` east-west and `(2 / H) / cos(lat)` north-south, so at 180x360
/// an equatorial cell is 0.64 x 1.0 degrees while the polar row is 8.6 x 0.1, and
/// `projection_jacobian`'s `du/dp` grows as `1/cos^2(lat)` along with it.
///
/// `Cube` is equi-angular: `s' = tan(pi s / 4)` on each face, one `tan` more than a plain
/// gnomonic cube, which brings the solid angle per cell from a 5.2:1 spread between face
/// centre and corner down to 1.41:1 and makes every cell 0.87 degrees square at the same
/// texel budget. `benchmarks/spintrack_bench/map_grid_sweep.py` measures what that buys.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Projection {
    EqualArea,
    Cube,
}

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
/// libm call. Both projections go through it, on both the splatting and the sampling side,
/// so the approximation error cancels out rather than accumulating.
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

/// `atan2` built on `atan_unit` by range reduction.
#[inline]
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    let swap = ay > ax;
    let (num, den) = if swap { (ax, ay) } else { (ay, ax) };
    let mut a = atan_unit(if den > 0.0 { num / den } else { 0.0 });
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
    pub projection: Projection,
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
    pub fn new_with(projection: Projection, w: usize, h: usize) -> Map {
        Map {
            w,
            h,
            projection,
            mean: vec![0.0; w * h],
            weight: vec![0.0; w * h],
        }
    }

    pub fn new(w: usize, h: usize) -> Map {
        Map::new_with(Projection::EqualArea, w, h)
    }

    /// A cube map of `face x face` cells per face, stacked into `(6 face, face)`.
    pub fn cube(face: usize) -> Map {
        Map::new_with(Projection::Cube, face, 6 * face)
    }

    pub fn clear(&mut self) {
        self.mean.fill(0.0);
        self.weight.fill(0.0);
    }

    /// Continuous map coordinates of a unit direction.
    #[inline]
    pub fn project(&self, p: [f32; 3]) -> (f32, f32) {
        match self.projection {
            Projection::EqualArea => {
                let lon = fast_atan2(p[0], p[2]);
                let u = (lon + PI) * (self.w as f32) / (2.0 * PI);
                let v = (1.0 - p[1]) * 0.5 * (self.h as f32);
                (u, v)
            }
            Projection::Cube => {
                let face = face_of(p);
                let (f, r, up) = FACES[face];
                let d = dot(p, f);
                let n = self.w as f32;
                let s = eac(dot(p, r) / d);
                let t = eac(dot(p, up) / d);
                (
                    0.5 * (s + 1.0) * n,
                    (face as f32) * n + 0.5 * (1.0 - t) * n,
                )
            }
        }
    }

    /// Gradient of `(u, v)` with respect to the direction `p` (unit), rows `du/dp`, `dv/dp`.
    ///
    /// The equal-area `du/dp` carries a `1 / cos^2(lat)` that is only nominally bounded, by
    /// the floor on `rxz`. The cube has no such term: `p` always lies on the face it is
    /// closest to, so `dot(p, forward) >= 1 / sqrt(3)`.
    #[inline]
    pub fn projection_jacobian(&self, p: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        match self.projection {
            Projection::EqualArea => {
                let rxz = (p[0] * p[0] + p[2] * p[2]).max(1e-12);
                let ku = self.w as f32 / (2.0 * PI) / rxz;
                let du = [ku * p[2], 0.0, -ku * p[0]];
                let dv = [0.0, -0.5 * self.h as f32, 0.0];
                (du, dv)
            }
            Projection::Cube => {
                let face = face_of(p);
                let (f, r, up) = FACES[face];
                let d = dot(p, f);
                let (s, t) = (dot(p, r) / d, dot(p, up) / d);
                let k = 0.5 * (self.w as f32) * (4.0 / PI) / d;
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
        }
    }

    /// Flat index of cube cell `(col, row)` of `face`, crossing the seam when either runs
    /// off the face.
    ///
    /// The out-of-range case re-projects the direction of the cell it was asked for instead
    /// of consulting an edge table: it cannot disagree with `project`, and it costs a `tan`
    /// and an `atan2` on the few per cent of taps that fall on a seam. At a cube corner the
    /// fourth neighbour does not exist and this lands on one of the other three, which is
    /// the usual way to handle the eight cells where that happens.
    #[inline]
    fn cube_cell(&self, face: usize, col: i32, row: i32) -> usize {
        let n = self.w as i32;
        if col >= 0 && col < n && row >= 0 && row < n {
            return (face * self.w + row as usize) * self.w + col as usize;
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

    /// The four cells a bilinear read at `(u, v)` draws on, and its fractional position:
    /// `[(u0, v0), (u1, v0), (u0, v1), (u1, v1)]`.
    #[inline]
    fn cell_indices(&self, u: f32, v: f32) -> ([usize; 4], f32, f32) {
        let uc = u - 0.5;
        match self.projection {
            Projection::EqualArea => {
                let vc = v - 0.5;
                let u0f = uc.floor();
                let v0f = vc.floor();
                let (fu, fv) = (uc - u0f, vc - v0f);
                let w = self.w as i32;
                let h = self.h as i32;
                // u lies in [0, W], so u0 is in [-1, W]: one conditional wrap each side.
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
                let (u0, u1) = (u0 as usize, u1 as usize);
                let v0 = (v0f as i32).clamp(0, h - 1) as usize * self.w;
                let v1 = (v0f as i32 + 1).clamp(0, h - 1) as usize * self.w;
                ([v0 + u0, v0 + u1, v1 + u0, v1 + u1], fu, fv)
            }
            Projection::Cube => {
                let n = self.w as i32;
                let face = ((v as i32) / n).clamp(0, 5);
                let vc = v - 0.5 - (face * n) as f32;
                let u0f = uc.floor();
                let v0f = vc.floor();
                let (fu, fv) = (uc - u0f, vc - v0f);
                let face = face as usize;
                let (c0, r0) = (u0f as i32, v0f as i32);
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
        }
    }

    /// The cell `(dcol, drow)` away, or `None` where the grid genuinely ends (the two
    /// polar rows of the equal-area grid; the cube has no edges).
    #[inline]
    fn neighbour(&self, idx: usize, dcol: i32, drow: i32) -> Option<usize> {
        match self.projection {
            Projection::EqualArea => {
                let row = (idx / self.w) as i32 + drow;
                if row < 0 || row >= self.h as i32 {
                    return None;
                }
                let col = ((idx % self.w) as i32 + dcol).rem_euclid(self.w as i32);
                Some(row as usize * self.w + col as usize)
            }
            Projection::Cube => {
                let n = self.w;
                let (face, rem) = (idx / (n * n), idx % (n * n));
                Some(self.cube_cell(face, (rem % n) as i32 + dcol, (rem / n) as i32 + drow))
            }
        }
    }

    /// Central-difference gradient of the mean at `idx`, one-sided at unseen neighbours;
    /// the cell itself is known to be seen.
    #[inline]
    fn cell_gradient(&self, idx: usize, w_min: f32) -> (f32, f32) {
        let m = self.mean[idx];
        let seen = |n: Option<usize>| match n {
            Some(i) if self.weight[i] >= w_min => self.mean[i],
            _ => m,
        };
        let ml = seen(self.neighbour(idx, -1, 0));
        let mr = seen(self.neighbour(idx, 1, 0));
        let gx = if ml == m || mr == m {
            mr - ml
        } else {
            0.5 * (mr - ml)
        };
        let mu = seen(self.neighbour(idx, 0, -1));
        let md = seen(self.neighbour(idx, 0, 1));
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
        let ([i00, i10, i01, i11], fu, fv) = self.cell_indices(u, v);
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
        let (gx00, gy00) = self.cell_gradient(i00, w_min);
        let (gx10, gy10) = self.cell_gradient(i10, w_min);
        let (gx01, gy01) = self.cell_gradient(i01, w_min);
        let (gx11, gy11) = self.cell_gradient(i11, w_min);
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
    /// `scale` multiplies the footprint, so a less trustworthy pixel counts for less.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn splat(
        &mut self,
        u: f32,
        v: f32,
        value: f32,
        lambda: f32,
        w_max: f32,
        scale: f32,
        touched: Option<&mut Touched>,
    ) {
        if scale <= 0.0 {
            return;
        }
        let ([i00, i10, i01, i11], fu, fv) = self.cell_indices(u, v);
        let cells = [
            (i00, scale * (1.0 - fu) * (1.0 - fv)),
            (i10, scale * fu * (1.0 - fv)),
            (i01, scale * (1.0 - fu) * fv),
            (i11, scale * fu * fv),
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
        let cube = self.projection == Projection::Cube;
        // A cube blurs face by face, clamping at their edges. A seam-correct blur would
        // mean resampling the whole map, for a pyramid whose only job is to widen the
        // basin of attraction; a one-cell artefact along twelve edges does not affect that.
        let (sum_wm, sum_w) = if cube {
            (
                self.per_face(&wm, |src, n| box_blur(src, n, n, radius, false)),
                self.per_face(&self.weight, |src, n| box_blur(src, n, n, radius, false)),
            )
        } else {
            (
                box_blur(&wm, self.w, self.h, radius, true),
                box_blur(&self.weight, self.w, self.h, radius, true),
            )
        };
        let w2 = (self.w / factor).max(1);
        let mut out = if cube {
            Map::cube(w2)
        } else {
            Map::new(w2, (self.h / factor).max(1))
        };
        let h2 = out.h;
        for y in 0..h2 {
            for x in 0..w2 {
                // Sample the blurred field at the centre of each decimated block; on a cube
                // each face decimates on its own, so the row is taken within the face.
                let yy = if cube {
                    (y / w2) * self.w + ((y % w2) * factor + factor / 2).min(self.w - 1)
                } else {
                    (y * factor + factor / 2).min(self.h - 1)
                };
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

    /// Apply a whole-image operation to each cube face in turn.
    fn per_face<T: Copy + Default, F: Fn(&[T], usize) -> Vec<T>>(
        &self,
        src: &[T],
        op: F,
    ) -> Vec<T> {
        let n = self.w;
        let mut out = vec![T::default(); src.len()];
        for f in 0..6 {
            let span = f * n * n..(f + 1) * n * n;
            out[span.clone()].copy_from_slice(&op(&src[span], n));
        }
        out
    }

    /// Drop everything outside a `margin`-cell dilation of the `touched` cells.
    pub fn forget_outside(&mut self, touched: &[u8], margin: usize) {
        let keep = if self.projection == Projection::Cube {
            self.per_face(touched, |src, n| dilate(src, n, n, margin, false))
        } else {
            dilate(touched, self.w, self.h, margin, true)
        };
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

/// Binary dilation with a square structuring element; `wrap_u` wraps in u, else clamps.
pub fn dilate(mask: &[u8], w: usize, h: usize, margin: usize, wrap_u: bool) -> Vec<u8> {
    let r = margin as i64;
    let mut out = vec![0u8; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut hit = 0u8;
            'outer: for dy in -r..=r {
                let yy = (y + dy).clamp(0, h as i64 - 1) as usize;
                for dx in -r..=r {
                    let xx = if wrap_u {
                        (x + dx).rem_euclid(w as i64)
                    } else {
                        (x + dx).clamp(0, w as i64 - 1)
                    } as usize;
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
                m.splat(10.5 + du, 7.5 + dv, 2.0, 1.0, 100.0, 1.0, None);
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

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    /// Direction of the point at map coordinates `(u, v)`; the inverse of `project`.
    fn unproject(m: &Map, u: f32, v: f32) -> [f32; 3] {
        match m.projection {
            Projection::EqualArea => {
                let lon = u / (m.w as f32) * 2.0 * PI - PI;
                let y = 1.0 - 2.0 * v / (m.h as f32);
                let r = (1.0 - y * y).max(0.0).sqrt();
                [r * lon.sin(), y, r * lon.cos()]
            }
            Projection::Cube => {
                let n = m.w as f32;
                let face = ((v / n) as usize).min(5);
                let s = eac_inv(2.0 * u / n - 1.0);
                let t = eac_inv(1.0 - 2.0 * (v - face as f32 * n) / n);
                unit(face_direction(face, s, t))
            }
        }
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
        check_jacobian(&Map::new(360, 180), [0.3, 0.2, 0.9]);
    }

    #[test]
    fn cube_projection_is_consistent_with_jacobian() {
        let m = Map::cube(64);
        // The centre of each face, an edge, a corner, and a point either side of a seam.
        for (f, r, u) in FACES {
            check_jacobian(&m, f);
            check_jacobian(&m, face_direction_of(f, r, u, 0.4, -0.7));
            check_jacobian(&m, face_direction_of(f, r, u, 0.97, 0.93));
        }
    }

    fn face_direction_of(f: [f32; 3], r: [f32; 3], u: [f32; 3], s: f32, t: f32) -> [f32; 3] {
        [
            f[0] + s * r[0] + t * u[0],
            f[1] + s * r[1] + t * u[1],
            f[2] + s * r[2] + t * u[2],
        ]
    }

    #[test]
    fn cube_projection_round_trips() {
        let m = Map::cube(48);
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16777216.0 * 2.0 - 1.0
        };
        for _ in 0..2000 {
            let p = unit([next(), next(), next()]);
            let (u, v) = m.project(p);
            assert!((0.0..m.w as f32).contains(&u), "u {u}");
            assert!((0.0..m.h as f32).contains(&v), "v {v}");
            let q = unproject(&m, u, v);
            assert!(dot(p, q) > 1.0 - 1e-6, "{p:?} -> {q:?}");
        }
    }

    #[test]
    fn cube_seams_join_neighbouring_faces() {
        let n = 32;
        let m = Map::cube(n);
        // A texel at the face centre spans pi/2 / n; the seam neighbour of an edge cell
        // must be its actual neighbour on the sphere, not a clamp back onto the same cell.
        let limit = 1.5 * FRAC_PI_2 / n as f32;
        for face in 0..6 {
            for k in 0..n {
                for (col, row, dcol, drow) in [
                    (0i32, k as i32, -1i32, 0i32),
                    (n as i32 - 1, k as i32, 1, 0),
                    (k as i32, 0, 0, -1),
                    (k as i32, n as i32 - 1, 0, 1),
                ] {
                    let idx = (face * n + row as usize) * n + col as usize;
                    let other = m.neighbour(idx, dcol, drow).expect("cube has no edges");
                    assert_ne!(other, idx, "face {face} cell ({col}, {row})");
                    let a = unproject(&m, col as f32 + 0.5, (face * n) as f32 + row as f32 + 0.5);
                    let b = unproject(
                        &m,
                        (other % n) as f32 + 0.5,
                        (other / n) as f32 + 0.5,
                    );
                    let gap = dot(a, b).clamp(-1.0, 1.0).acos();
                    assert!(gap < limit, "face {face} ({col}, {row}) gap {gap}");
                }
            }
        }
    }

    #[test]
    fn cube_splat_conserves_weight_over_a_seam() {
        let n = 16;
        for (u, v) in [(8.5f32, 8.5f32), (0.0, 8.5), (16.0, 8.5), (8.5, 16.0), (0.0, 0.0)] {
            let mut m = Map::cube(n);
            m.splat(u, v, 1.0, 1.0, 1e6, 1.0, None);
            let total: f32 = m.weight.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "({u}, {v}) deposited {total}");
        }
    }
}
