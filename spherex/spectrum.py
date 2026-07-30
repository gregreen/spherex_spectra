"""Source spectrum models for SPHEREx photometry.

Each spectrum model is an ``equinox.Module`` that acts as a *parameterised
template*: it stores no per-source state, and its ``__call__`` method accepts
explicit parameter arrays so that gradients flow through them naturally.
"""

import equinox as eqx
import jax.numpy as jnp

from .constants import C_JAX, HC_JAX, KB_JAX


class BlackbodySpectrum(eqx.Module):
    """Blackbody spectrum template.

    The flux density at wavelength ``lambda`` for a source with temperature
    ``T`` and amplitude ``A`` is::

        f_lambda = A * B_lambda(lambda, T)

    where ``B_lambda`` is the Planck function (W m^-2 um^-1 sr^-1).  The
    amplitude absorbs the per-solid-angle factor, giving ``f_lambda`` in
    W m^-2 um^-1.

    Parameters
    ----------
    wavelength : jnp.ndarray, shape (...,)
        Wavelength(s) in microns.
    params : jnp.ndarray, shape (..., 2)
        Per-source parameters in **log-space**.
        ``params[..., 0]`` = log(temperature / kK),
        ``params[..., 1]`` = log(amplitude / (W m^-2 um^-1)).
    """

    n_params: int = 2

    def __call__(
        self,
        wavelength: jnp.ndarray,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        # Slice with a trailing dummy axis for broadcasting.
        log_temperature = params[..., 0:1]   # log(kK)       (..., 1)
        log_amplitude  = params[..., 1:2]   # log(W/m²/µm)  (..., 1)

        temperature = jnp.exp(log_temperature)               # kK
        amplitude  = jnp.exp(log_amplitude)                  # W/(m² µm)

        # Compute exponent x = hc / (lambda * k_B * T) carefully to avoid
        # float32 underflow in the backward pass.
        #
        # Naively, HC / (wavelength * KB * T) causes the gradient
        # -HC / (wavelength * KB * T^2) to compute (KB * T)^2 which
        # underflows float32 (~4.8e-39 at T=5 kK).  We instead
        # pre-compute inv_T_scale = HC / (wavelength * KB) so that the
        # gradient becomes -inv_T_scale / T^2 (neither factor underflows).
        inv_T_scale = HC_JAX / (wavelength * KB_JAX)          # K
        exponent = inv_T_scale / temperature                  # dimensionless

        exponent = jnp.clip(exponent, None, 50.0)   # guard exp overflow

        # Planck function  B_lambda = 2 h c^2 / lambda^5 * 1 / (e^x - 1)
        planck = (
            (2.0 * HC_JAX * C_JAX / wavelength**5)
            / (jnp.exp(exponent) - 1.0)
        )

        return amplitude * planck
