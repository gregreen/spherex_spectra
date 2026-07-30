"""Physical constants for SPHEREx photometry.

All values are derived from ``astropy.constants`` and converted to
codebase-native units via ``.to()`` for clarity and maintainability.
JAX-traceable copies are provided alongside plain-Python floats.

----
Base units used throughout the codebase
----
  temperature    : kiloKelvin (kK)
  wavelength     : micron (um)
  angular        : arcsec
  flux density   : W / (m^2 um)
  photon rate    : s^{-1}
"""

import jax.numpy as jnp
import astropy.units as u
from astropy.constants import h, c, k_B

# ── constants in codebase-native units ───────────────────────────────────

# Temperature unit
TEMPERATURE_UNIT = "kK"
KB: float = k_B.to(u.J / u.Unit(TEMPERATURE_UNIT)).value     # J / kK

# Wavelength unit
WAVELENGTH_UNIT = "um"
HC: float = (h * c).to(u.J * u.Unit(WAVELENGTH_UNIT)).value  # J * um  (photon energy)

# Speed of light (for Planck prefactor  2 h c^2 / lambda^5)
C: float = c.to(u.m / u.s).value               # m / s

# Planck constant (kept for reference)
H: float = h.to(u.J * u.s).value               # J * s

# ── JAX-traceable copies ─────────────────────────────────────────────────

H_JAX = jnp.asarray(H)
C_JAX = jnp.asarray(C)
HC_JAX = jnp.asarray(HC)
KB_JAX = jnp.asarray(KB)
