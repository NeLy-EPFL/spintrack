//! Small dense 3x3 linear algebra and the SO(3) exponential/logarithm maps.

pub type Mat3 = [[f64; 3]; 3];
pub type Vec3 = [f64; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

pub fn transpose(a: &Mat3) -> Mat3 {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn mat_vec(a: &Mat3, v: Vec3) -> Vec3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm(v: Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Rodrigues' formula, `exp([w]x)`, accurate for small angles.
pub fn exp_so3(w: Vec3) -> Mat3 {
    let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let (a, b) = if theta2 < 1e-8 {
        (1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0)
    } else {
        let theta = theta2.sqrt();
        (theta.sin() / theta, (1.0 - theta.cos()) / theta2)
    };
    let k = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    let k2 = mat_mul(&k, &k);
    let mut out = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] += a * k[i][j] + b * k2[i][j];
        }
    }
    out
}

/// Logarithm map; the returned angle lies in `[0, pi]`.
pub fn log_so3(r: &Mat3) -> Vec3 {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let vee = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    if theta < 1e-6 {
        return [0.5 * vee[0], 0.5 * vee[1], 0.5 * vee[2]];
    }
    if std::f64::consts::PI - theta < 1e-6 {
        // Antisymmetric part vanishes near pi; take the axis from the diagonal of R + I.
        let m = [
            [r[0][0] + 1.0, r[0][1], r[0][2]],
            [r[1][0], r[1][1] + 1.0, r[1][2]],
            [r[2][0], r[2][1], r[2][2] + 1.0],
        ];
        let i = if m[0][0] >= m[1][1] && m[0][0] >= m[2][2] {
            0
        } else if m[1][1] >= m[2][2] {
            1
        } else {
            2
        };
        let col = [m[0][i], m[1][i], m[2][i]];
        let n = norm(col);
        let mut axis = [col[0] / n, col[1] / n, col[2] / n];
        if axis[0] * vee[0] + axis[1] * vee[1] + axis[2] * vee[2] < 0.0 {
            axis = [-axis[0], -axis[1], -axis[2]];
        }
        return [theta * axis[0], theta * axis[1], theta * axis[2]];
    }
    let s = theta / (2.0 * theta.sin());
    [s * vee[0], s * vee[1], s * vee[2]]
}

/// Solve the symmetric positive (semi)definite 3x3 system `h x = b` by Cholesky with a
/// tiny diagonal damping; returns `None` when the system is degenerate.
pub fn solve_spd3(h: &Mat3, b: Vec3, damping: f64) -> Option<Vec3> {
    let mut a = *h;
    for i in 0..3 {
        a[i][i] += damping * (1.0 + a[i][i]);
    }
    // Cholesky: a = l l^T.
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i][i] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Forward then backward substitution.
    let mut y = [0.0; 3];
    for i in 0..3 {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        let mut sum = y[i];
        for k in (i + 1)..3 {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_log_round_trip() {
        for w in [[0.1, -0.2, 0.3], [1e-9, 0.0, 0.0], [0.0, 2.5, -1.0]] {
            let r = exp_so3(w);
            let back = log_so3(&r);
            for i in 0..3 {
                assert!((back[i] - w[i]).abs() < 1e-9, "{w:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn cholesky_solves() {
        let h = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let x = [1.0, -2.0, 0.5];
        let b = mat_vec(&h, x);
        let got = solve_spd3(&h, b, 0.0).unwrap();
        for i in 0..3 {
            assert!((got[i] - x[i]).abs() < 1e-12);
        }
    }
}
