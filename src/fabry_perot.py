"""Analytic model of a two-mirror Fabry-Perot cavity.

Everything here is a closed-form expression of the five system parameters
(ROC1, ROC2, R1, R2, L) plus the wavelength. Mirrors are lossless: T_i = 1 - R_i.

Conventions
-----------
* ROC > 0 for a concave mirror (seen from inside the cavity); np.inf for a flat mirror.
* R1, R2 are power reflectivities.
* Frequencies are in Hz; `df` is the laser detuning from a cavity resonance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


@dataclass
class FabryPerot:
    roc1: float                  # mirror 1 radius of curvature [m]
    roc2: float                  # mirror 2 radius of curvature [m]
    R1: float                    # mirror 1 power reflectivity
    R2: float                    # mirror 2 power reflectivity
    length: float                # mirror separation L [m]
    wavelength: float = 1550e-9  # [m]

    # ----- spatial: stability and TEM00 mode --------------------------------
    @property
    def g1(self) -> float:
        return 1.0 - self.length / self.roc1

    @property
    def g2(self) -> float:
        return 1.0 - self.length / self.roc2

    @property
    def is_stable(self) -> bool:
        """Stability condition 0 < g1 g2 < 1."""
        return 0.0 < self.g1 * self.g2 < 1.0

    def waist(self) -> float:
        """TEM00 waist radius w0 [m].

        w0^2 = (L lambda / pi) * sqrt( g1 g2 (1 - g1 g2) / (g1 + g2 - 2 g1 g2)^2 )
        """
        if not self.is_stable:
            raise ValueError(f"Cavity is not stable (g1*g2 = {self.g1 * self.g2:.4f})")
        g1, g2 = self.g1, self.g2
        w0_sq = (self.length * self.wavelength / np.pi) * np.sqrt(
            g1 * g2 * (1.0 - g1 * g2) / (g1 + g2 - 2.0 * g1 * g2) ** 2)
        return float(np.sqrt(w0_sq))

    def waist_position(self) -> float:
        """Distance of the waist from mirror 1 [m]: z1 = L g2 (1 - g1) / (g1 + g2 - 2 g1 g2)."""
        g1, g2 = self.g1, self.g2
        return float(self.length * g2 * (1.0 - g1) / (g1 + g2 - 2.0 * g1 * g2))

    def rayleigh_range(self) -> float:
        """z_R = pi w0^2 / lambda [m]."""
        return np.pi * self.waist() ** 2 / self.wavelength

    # ----- spectral ---------------------------------------------------------
    @property
    def fsr(self) -> float:
        """Free spectral range c / 2L [Hz]."""
        return C_LIGHT / (2.0 * self.length)

    @property
    def r_eff(self) -> float:
        """Effective round-trip amplitude factor sqrt(R1 R2)."""
        return np.sqrt(self.R1 * self.R2)

    @property
    def coefficient_of_finesse(self) -> float:
        """F = 4 sqrt(R1 R2) / (1 - sqrt(R1 R2))^2 (the 'F' in the Airy function)."""
        return 4.0 * self.r_eff / (1.0 - self.r_eff) ** 2

    @property
    def finesse(self) -> float:
        """Finesse = pi sqrt(F) / 2."""
        return np.pi * np.sqrt(self.coefficient_of_finesse) / 2.0

    @property
    def linewidth(self) -> float:
        """Resonance full width at half maximum, delta_nu = FSR / Finesse [Hz]."""
        return self.fsr / self.finesse

    def round_trip_phase(self, df):
        """delta = 2 pi df / FSR (round-trip phase relative to resonance)."""
        return 2.0 * np.pi * np.asarray(df) / self.fsr

    def transmission(self, df):
        """Airy transmission T = T_max / (1 + F sin^2(delta / 2)).

        T_max = (1 - R1)(1 - R2) / (1 - sqrt(R1 R2))^2  (= 1 for R1 = R2).
        """
        t_max = (1.0 - self.R1) * (1.0 - self.R2) / (1.0 - self.r_eff) ** 2
        delta = self.round_trip_phase(df)
        return t_max / (1.0 + self.coefficient_of_finesse * np.sin(delta / 2.0) ** 2)

    def reflection(self, df):
        """Reflection R = 1 - T (lossless mirrors)."""
        return 1.0 - self.transmission(df)

    def summary(self) -> dict:
        out = {
            "g1": self.g1,
            "g2": self.g2,
            "g1*g2": self.g1 * self.g2,
            "stable": self.is_stable,
            "FSR [Hz]": self.fsr,
            "coefficient of finesse F": self.coefficient_of_finesse,
            "finesse": self.finesse,
            "linewidth FWHM [Hz]": self.linewidth,
        }
        if self.is_stable:
            out.update({
                "waist w0 [m]": self.waist(),
                "waist from M1 [m]": self.waist_position(),
                "Rayleigh range zR [m]": self.rayleigh_range(),
            })
        return out
