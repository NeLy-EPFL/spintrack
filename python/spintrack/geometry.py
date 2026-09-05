"""Rotations in SO(3): rotation vectors (angle-axis), matrices and helpers.

Conventions: a rotation vector `w` has direction = axis and norm = angle in radians;
`rotvec_to_matrix(w) @ v` rotates `v` by that angle about that axis (right-hand rule).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix `[v]x` such that `skew(v) @ u == cross(v, u)`."""
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotvec_to_matrix(w: np.ndarray) -> np.ndarray:
    """Exponential map so(3) -> SO(3) (Rodrigues' formula), accurate for small angles."""
    w = np.asarray(w, dtype=np.float64)
    theta2 = float(w @ w)
    k = skew(w)
    if theta2 < 1e-8:
        # Taylor expansions of sin(t)/t and (1 - cos(t))/t^2.
        a = 1.0 - theta2 / 6.0
        b = 0.5 - theta2 / 24.0
    else:
        theta = np.sqrt(theta2)
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / theta2
    return np.eye(3) + a * k + b * (k @ k)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Logarithm map SO(3) -> so(3); the angle is in [0, pi]."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * vee
    if np.pi - theta < 1e-6:
        # Near pi the antisymmetric part vanishes; take the axis from R + I.
        m = R + np.eye(3)
        i = int(np.argmax(np.diag(m)))
        axis = m[:, i] / np.linalg.norm(m[:, i])
        # Pick the sign that agrees with the (small) antisymmetric part.
        if vee @ axis < 0:
            axis = -axis
        return theta * axis
    return (theta / (2.0 * np.sin(theta))) * vee


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation matrix taking unit vector `a` onto unit vector `b`."""
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(a @ b)
    if s < _EPS:
        if c > 0:
            return np.eye(3)
        # Antiparallel: rotate by pi about any axis orthogonal to a.
        helper = (
            np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        )
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        return rotvec_to_matrix(np.pi * axis)
    angle = np.arctan2(s, c)
    return rotvec_to_matrix(axis / s * angle)


def rotation_angle(R: np.ndarray) -> float:
    """Angle in radians of the rotation `R`."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def normalize(v: np.ndarray) -> np.ndarray:
    """Unit vectors along the last axis (zero vectors are returned unchanged)."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)
