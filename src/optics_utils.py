"""Gaussian-beam optics: complex beam parameter q and ABCD matrices.

Conventions
-----------
* q = z + i*z_R, where z is the distance *from* the waist (z < 0 before the waist).
* 1/q = 1/R - i * lambda / (pi * w^2).
* ABCD matrices act on (x, x') and are multiplied in reverse order of traversal:
  M_total = M_N @ ... @ M_2 @ M_1.
* All lengths are in meters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# ABCD matrices
# ---------------------------------------------------------------------------


def free_space(d: float) -> np.ndarray:
    """Propagation over a distance d in free space."""
    return np.array([[1.0, d], [0.0, 1.0]])


def thin_lens(f: float) -> np.ndarray:
    """Thin lens of focal length f (f > 0 converging)."""
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def system_matrix(*elements: np.ndarray) -> np.ndarray:
    """Total ABCD matrix for elements given in the order the beam meets them."""
    m = np.eye(2)
    for el in elements:
        m = el @ m
    return m


# ---------------------------------------------------------------------------
# q-parameter
# ---------------------------------------------------------------------------


def rayleigh_range(w0, wavelength: float):
    """z_R = pi w0^2 / lambda."""
    return np.pi * np.asarray(w0) ** 2 / wavelength


def q_from_waist(w0: float, wavelength: float, z: float = 0.0) -> complex:
    """q at a distance z after a waist of radius w0."""
    return z + 1j * rayleigh_range(w0, wavelength)


def propagate_q(q, m: np.ndarray):
    """Apply an ABCD matrix: q' = (A q + B) / (C q + D). Works element-wise on arrays."""
    (a, b), (c, d) = m
    return (a * q + b) / (c * q + d)


def beam_radius(q, wavelength: float):
    """1/e^2 intensity radius w from q."""
    inv_q = 1.0 / np.asarray(q)
    return np.sqrt(-wavelength / (np.pi * inv_q.imag))


def waist_from_q(q, wavelength: float):
    """Return (w0, z_waist): waist radius and waist position relative to the current plane.

    z_waist > 0 means the waist lies downstream of the current plane.
    """
    q = np.asarray(q)
    return np.sqrt(q.imag * wavelength / np.pi), -q.real


# ---------------------------------------------------------------------------
# Collimator -> free space -> focusing lens
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Collimator:
    """Fiber collimator: the output beam is described by its 1/e^2 beam diameter."""
    name: str
    focal_length: float     # effective focal length of the collimator [m]
    beam_diameter: float    # 1/e^2 output beam diameter [m]

    @property
    def waist(self) -> float:
        """Input waist radius w_in = D / 2, assumed to sit at the collimator output (z = 0)."""
        return self.beam_diameter / 2.0


def collimated_beam_diameter(wavelength: float, focal_length: float, mfd: float) -> float:
    """Theoretical collimated 1/e^2 diameter D = 4 lambda f / (pi MFD) (cross-check of datasheet)."""
    return 4.0 * wavelength * focal_length / (np.pi * mfd)


def focus_beam(w_in: float, wavelength: float, distance: float, focal_length):
    """Send a waist w_in (at z = 0) over `distance` and through a thin lens `focal_length`.

    Uses q_out = M q_in with M = L(f) @ S(d). Vectorized over focal_length and distance.
    Returns (z_out, w_out): new waist position after the lens and its radius.
    """
    q_in = q_from_waist(w_in, wavelength)
    q_at_lens = q_in + np.asarray(distance)          # free space: A=1, B=d, C=0, D=1
    f = np.asarray(focal_length, dtype=float)
    q_out = q_at_lens / (1.0 - q_at_lens / f)        # thin lens: A=1, B=0, C=-1/f, D=1
    w_out, z_out = waist_from_q(q_out, wavelength)
    return z_out, w_out


def focus_beam_analytic(w_in: float, wavelength: float, distance: float, focal_length):
    """Closed-form thin-lens waist transformation (Self, Appl. Opt. 22, 658 (1983)).

    Same result as focus_beam; used as an independent check of the ABCD calculation.
    """
    f = np.asarray(focal_length, dtype=float)
    zr = rayleigh_range(w_in, wavelength)
    s = np.asarray(distance) - f                      # waist position relative to front focus
    denom = s**2 + zr**2
    z_out = f + f**2 * s / denom
    w_out = w_in * np.abs(f) / np.sqrt(denom)
    return z_out, w_out


def trace_beam(w_in: float, wavelength: float, distance: float, focal_length: float,
               z_after: float, n_points: int = 1000):
    """w(z) from the collimator (z = 0) through the lens (z = distance) up to z_after past it."""
    q_in = q_from_waist(w_in, wavelength)
    z1 = np.linspace(0.0, distance, n_points)
    w1 = beam_radius(q_in + z1, wavelength)
    q_lens = propagate_q(q_in + distance, thin_lens(focal_length))
    z2 = np.linspace(0.0, z_after, n_points)
    w2 = beam_radius(q_lens + z2, wavelength)
    return np.concatenate([z1, distance + z2]), np.concatenate([w1, w2])
