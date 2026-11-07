"""Gaussian-beam optics: ABCD matrices and the complex beam parameter q.

Conventions
-----------
* q = z + i*z_R, where z is the distance *from* the waist (z < 0 before the waist).
* 1/q = 1/R - i * lambda / (pi * n * w^2).
* ABCD matrices act on (x, x') and are multiplied in reverse order of traversal:
  M_total = M_N @ ... @ M_2 @ M_1.
* All lengths are in meters.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# ABCD matrices
# ---------------------------------------------------------------------------


def free_space(d: float, n: float = 1.0) -> np.ndarray:
    """Propagation over a distance d in a medium of index n (reduced-angle form)."""
    return np.array([[1.0, d / n], [0.0, 1.0]])


def thin_lens(f: float) -> np.ndarray:
    """Thin lens of focal length f (f > 0 converging)."""
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def curved_mirror(roc: float) -> np.ndarray:
    """Reflection from a mirror with radius of curvature roc at normal incidence.

    roc > 0 for a concave (focusing) mirror; roc = np.inf for a flat mirror.
    """
    return np.array([[1.0, 0.0], [-2.0 / roc, 1.0]])


def flat_interface(n1: float, n2: float) -> np.ndarray:
    """Refraction at a flat interface from index n1 into index n2."""
    return np.array([[1.0, 0.0], [0.0, n1 / n2]])


def system_matrix(*elements: np.ndarray) -> np.ndarray:
    """Total ABCD matrix for elements given in the order the beam meets them."""
    m = np.eye(2)
    for el in elements:
        m = el @ m
    return m


# ---------------------------------------------------------------------------
# q-parameter transforms
# ---------------------------------------------------------------------------


def rayleigh_range(w0, wavelength: float, n: float = 1.0):
    return np.pi * n * np.asarray(w0) ** 2 / wavelength


def q_from_waist(w0: float, wavelength: float, z: float = 0.0, n: float = 1.0) -> complex:
    """q at a distance z from a waist of radius w0."""
    return z + 1j * rayleigh_range(w0, wavelength, n)


def propagate_q(q, m: np.ndarray):
    """Apply an ABCD matrix to q (works element-wise on arrays of q)."""
    (a, b), (c, d) = m
    return (a * q + b) / (c * q + d)


def q_through_lens(q, f):
    """q just after a thin lens (vectorized shortcut for propagate_q(q, thin_lens(f)))."""
    return q / (1.0 - q / f)


def beam_radius(q, wavelength: float, n: float = 1.0):
    """1/e^2 intensity radius w from q."""
    inv_q = 1.0 / np.asarray(q)
    return np.sqrt(-wavelength / (np.pi * n * inv_q.imag))


def radius_of_curvature(q):
    """Wavefront radius of curvature R (np.inf at a waist)."""
    re = np.real(1.0 / np.asarray(q))
    with np.errstate(divide="ignore"):
        return np.where(re == 0, np.inf, 1.0 / re)


def waist_from_q(q, wavelength: float, n: float = 1.0):
    """Return (w0, z_waist) where z_waist is the waist position relative to the current plane.

    z_waist > 0 means the waist lies downstream of the current plane.
    """
    q = np.asarray(q)
    w0 = np.sqrt(q.imag * wavelength / (np.pi * n))
    return w0, -q.real


def gouy_phase(q):
    """Gouy phase arctan(z / z_R) at the plane described by q."""
    q = np.asarray(q)
    return np.arctan2(q.real, q.imag)


def mode_overlap(q1, q2):
    """Power coupling efficiency between two aligned TEM00 modes evaluated at the same plane.

    eta = 4 Im(q1) Im(q2) / |q1* - q2|^2   (1 for identical modes).
    For waists w1, w2 separated by dz: eta = 4 zR1 zR2 / ((zR1 + zR2)^2 + dz^2).
    Works element-wise on arrays.
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    return 4.0 * q1.imag * q2.imag / np.abs(np.conj(q1) - q2) ** 2


# ---------------------------------------------------------------------------
# Beam tracing through a 1D optical train
# ---------------------------------------------------------------------------


def trace_beam(q0: complex, wavelength: float, elements, points_per_meter: int = 2000):
    """Trace w(z) through a list of elements.

    elements: sequence of ("space", d) or ("lens", f) tuples, in beam order.
    Returns (z, w, q_final), with z starting at 0 at the plane where q0 is defined.
    """
    z_all, w_all = [np.array([0.0])], [np.array([beam_radius(q0, wavelength)])]
    q, z = complex(q0), 0.0
    for kind, value in elements:
        if kind == "space":
            n_pts = max(int(abs(value) * points_per_meter), 2)
            dz = np.linspace(0.0, value, n_pts)[1:]
            z_all.append(z + dz)
            w_all.append(beam_radius(q + dz, wavelength))
            q, z = q + value, z + value
        elif kind == "lens":
            q = q_through_lens(q, value)
        else:
            raise ValueError(f"Unknown element type: {kind!r}")
    return np.concatenate(z_all), np.concatenate(w_all), q


# ---------------------------------------------------------------------------
# Resonator eigenmode & mode matching
# ---------------------------------------------------------------------------


def eigenmode_q(m_roundtrip: np.ndarray) -> complex:
    """Self-consistent q of a resonator from its round-trip ABCD matrix.

    Solves C q^2 + (D - A) q - B = 0 and keeps the root with Im(q) > 0.
    Raises ValueError if the resonator is unstable (|A + D| >= 2).
    """
    (a, b), (c, d) = m_roundtrip
    if abs(a + d) >= 2.0:
        raise ValueError(f"Unstable resonator: |A + D| = {abs(a + d):.4f} >= 2")
    roots = np.roots([c, d - a, -b]).astype(complex)
    return complex(roots[np.argmax(roots.imag)])


def _two_lens_q(q_in, f1, x1, f2, x2, z_target):
    """q at z_target for input q_in at z=0 and lenses f1 at x1, f2 at x2 (vectorized)."""
    q = q_in + x1
    q = q_through_lens(q, f1)
    q = q + (x2 - x1)
    q = q_through_lens(q, f2)
    return q + (z_target - x2)


def optimize_two_lens(
    q_in: complex,
    q_target: complex,
    z_target: float,
    f1: float,
    f2: float,
    min_gap: float = 0.02,
    grid_points: int = 300,
):
    """Find lens positions (x1 < x2) that best map q_in (at z=0) onto q_target (at z_target).

    Brute-force grid search followed by Nelder-Mead refinement.
    Returns dict with keys x1, x2, eta, q_out.
    """
    xs = np.linspace(min_gap, z_target - min_gap, grid_points)
    x1, x2 = np.meshgrid(xs, xs, indexing="ij")
    valid = x2 - x1 >= min_gap
    eta = np.where(valid, mode_overlap(_two_lens_q(q_in, f1, x1, f2, x2, z_target), q_target), 0.0)
    i, j = np.unravel_index(np.argmax(eta), eta.shape)

    def cost(p):
        a, b = p
        if not (min_gap <= a and a + min_gap <= b <= z_target - min_gap):
            return 1.0
        return 1.0 - mode_overlap(_two_lens_q(q_in, f1, a, f2, b, z_target), q_target)

    res = minimize(cost, x0=[x1[i, j], x2[i, j]], method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-12})
    best_x1, best_x2 = res.x
    q_out = _two_lens_q(q_in, f1, best_x1, f2, best_x2, z_target)
    return {"x1": best_x1, "x2": best_x2, "eta": float(mode_overlap(q_out, q_target)), "q_out": q_out}
