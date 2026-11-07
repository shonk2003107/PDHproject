"""Two-mirror Fabry-Perot cavity: figures of merit, field response and PDH error signal.

Conventions
-----------
* R1, R2 are power reflectivities; mirrors are lossless unless `loss` is set
  (t_i^2 = 1 - R_i - loss_i).
* Detuning `df` is the laser frequency offset from a cavity resonance, in Hz.
* Reflection coefficient (E. Black, Am. J. Phys. 69, 79 (2001), generalized to r1 != r2):
      F(df) = (-r1 + (r1^2 + t1^2) r2 e^{i phi}) / (1 - r1 r2 e^{i phi}),  phi = 2 pi df / FSR
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import jv

from .optics_utils import curved_mirror, eigenmode_q, free_space, system_matrix

C_LIGHT = 299_792_458.0  # m/s


@dataclass
class FabryPerot:
    length: float                 # mirror separation [m]
    R1: float                     # input mirror power reflectivity
    R2: float                     # end mirror power reflectivity
    roc1: float = np.inf          # input mirror radius of curvature [m] (>0 concave)
    roc2: float = np.inf          # end mirror radius of curvature [m]
    wavelength: float = 1064e-9   # [m]
    loss1: float = 0.0            # scatter/absorption loss of mirror 1 (power fraction)
    loss2: float = 0.0            # scatter/absorption loss of mirror 2

    # ----- mirror amplitudes -------------------------------------------------
    @property
    def r1(self) -> float:
        return np.sqrt(self.R1)

    @property
    def r2(self) -> float:
        return np.sqrt(self.R2)

    @property
    def t1(self) -> float:
        return np.sqrt(1.0 - self.R1 - self.loss1)

    @property
    def t2(self) -> float:
        return np.sqrt(1.0 - self.R2 - self.loss2)

    # ----- longitudinal figures of merit ------------------------------------
    @property
    def fsr(self) -> float:
        """Free spectral range [Hz]."""
        return C_LIGHT / (2.0 * self.length)

    @property
    def finesse(self) -> float:
        rr = self.r1 * self.r2
        return np.pi * np.sqrt(rr) / (1.0 - rr)

    @property
    def linewidth(self) -> float:
        """Full width at half maximum of the resonance [Hz]."""
        return self.fsr / self.finesse

    @property
    def storage_time(self) -> float:
        """Field amplitude decay time tau = 1 / (pi * FWHM) [s]."""
        return 1.0 / (np.pi * self.linewidth)

    @property
    def quality_factor(self) -> float:
        return (C_LIGHT / self.wavelength) / self.linewidth

    # ----- transverse / stability -------------------------------------------
    @property
    def g1(self) -> float:
        return 1.0 - self.length / self.roc1

    @property
    def g2(self) -> float:
        return 1.0 - self.length / self.roc2

    @property
    def g_product(self) -> float:
        return self.g1 * self.g2

    @property
    def is_stable(self) -> bool:
        return 0.0 < self.g_product < 1.0

    def roundtrip_matrix(self) -> np.ndarray:
        """Round-trip ABCD matrix starting at mirror 1, heading toward mirror 2."""
        return system_matrix(
            free_space(self.length),
            curved_mirror(self.roc2),
            free_space(self.length),
            curved_mirror(self.roc1),
        )

    def eigenmode_q(self) -> complex:
        """q of the cavity TEM00 mode at mirror 1 (propagating toward mirror 2)."""
        return eigenmode_q(self.roundtrip_matrix())

    def waist(self) -> tuple[float, float]:
        """Analytic (w0, z1): waist radius and its distance from mirror 1."""
        if not self.is_stable:
            raise ValueError(f"Cavity is not stable (g1*g2 = {self.g_product:.4f})")
        g1, g2, L, lam = self.g1, self.g2, self.length, self.wavelength
        denom = g1 + g2 - 2.0 * g1 * g2
        w0_sq = (L * lam / np.pi) * np.sqrt(g1 * g2 * (1.0 - g1 * g2) / denom**2)
        z1 = L * g2 * (1.0 - g1) / denom
        return float(np.sqrt(w0_sq)), float(z1)

    def mirror_spot_sizes(self) -> tuple[float, float]:
        """Beam radii on mirror 1 and mirror 2."""
        g1, g2 = self.g1, self.g2
        base = self.length * self.wavelength / np.pi
        w1 = np.sqrt(base * np.sqrt(g2 / (g1 * (1.0 - g1 * g2))))
        w2 = np.sqrt(base * np.sqrt(g1 / (g2 * (1.0 - g1 * g2))))
        return float(w1), float(w2)

    def transverse_mode_spacing(self) -> float:
        """Frequency spacing between TEM_mn orders (m+n -> m+n+1) [Hz]."""
        root = np.sqrt(self.g_product)
        return self.fsr * np.arccos(np.sign(self.g1) * root) / np.pi

    # ----- field response -------------------------------------------------
    def _phase(self, df):
        return 2.0 * np.pi * np.asarray(df) / self.fsr

    def reflection(self, df):
        """Complex amplitude reflection coefficient F(df)."""
        e = np.exp(1j * self._phase(df))
        r1, r2, t1 = self.r1, self.r2, self.t1
        return (-r1 + (r1**2 + t1**2) * r2 * e) / (1.0 - r1 * r2 * e)

    def transmission(self, df):
        """Complex amplitude transmission coefficient."""
        phi = self._phase(df)
        return self.t1 * self.t2 * np.exp(1j * phi / 2.0) / (1.0 - self.r1 * self.r2 * np.exp(1j * phi))

    def intracavity_gain(self, df):
        """Circulating power / incident power."""
        phi = self._phase(df)
        return np.abs(self.t1 / (1.0 - self.r1 * self.r2 * np.exp(1j * phi))) ** 2

    # ----- Pound-Drever-Hall -------------------------------------------------
    def pdh_error_signal(self, df, mod_freq: float, beta: float, power: float = 1.0,
                         demod_phase: float = np.pi / 2):
        """PDH error signal after mixing and low-pass filtering [W].

        eps = 2 sqrt(Pc Ps) [Re(chi) cos(theta) + Im(chi) sin(theta)],
        chi = F(w) F*(w + W) - F*(w) F(w - W),
        Pc = J0(beta)^2 P0, Ps = J1(beta)^2 P0.
        demod_phase = pi/2 selects the dispersive quadrature used for locking.
        """
        df = np.asarray(df)
        pc = jv(0, beta) ** 2 * power
        ps = jv(1, beta) ** 2 * power
        f0 = self.reflection(df)
        chi = f0 * np.conj(self.reflection(df + mod_freq)) - np.conj(f0) * self.reflection(df - mod_freq)
        return 2.0 * np.sqrt(pc * ps) * (chi.real * np.cos(demod_phase) + chi.imag * np.sin(demod_phase))

    def pdh_slope(self, mod_freq: float, beta: float, power: float = 1.0, demod_phase: float = np.pi / 2,
                  step: float | None = None) -> float:
        """Numerical slope dEps/d(df) at resonance [W/Hz]."""
        h = step if step is not None else self.linewidth * 1e-4
        e = self.pdh_error_signal(np.array([-h, h]), mod_freq, beta, power, demod_phase)
        return float((e[1] - e[0]) / (2.0 * h))

    def summary(self) -> dict:
        out = {
            "length_m": self.length,
            "FSR_Hz": self.fsr,
            "finesse": self.finesse,
            "linewidth_FWHM_Hz": self.linewidth,
            "storage_time_s": self.storage_time,
            "Q": self.quality_factor,
            "g1": self.g1,
            "g2": self.g2,
            "g1g2": self.g_product,
            "stable": self.is_stable,
        }
        if self.is_stable:
            w0, z1 = self.waist()
            out.update({"waist_m": w0, "waist_from_M1_m": z1,
                        "TMS_Hz": self.transverse_mode_spacing()})
        return out
