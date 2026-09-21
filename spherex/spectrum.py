"""Source spectrum models for SPHEREx photometry.

Each spectrum model is an ``equinox.Module`` that acts as a *parameterised
template* describing the **shape** of a source spectrum: it stores no
per-source state, and its ``__call__`` method accepts explicit parameter
arrays so that gradients flow through them naturally.

Amplitude convention
--------------------
The overall flux-density scale (``log_amplitude``) is deliberately *not*
part of a spectrum model.  It is carried as a separate per-source parameter,
always stored as the **first** column of ``source_params`` (see
:data:`LOG_AMPLITUDE_INDEX`).  The image generators apply it *outside* the
spectrum model::

    f_lambda = exp(source_params[:, 0]) * spectrum_model(lambda, source_params[:, 1:])

so ``spectrum_model`` only ever describes the spectral *shape*.  For
:class:`BlackbodySpectrum` the shape is normalised to ``f_lambda = 1`` at
:data:`LAMBDA_0`, so ``log_amplitude`` has a direct physical meaning: it is
the log of the source's flux density at ``LAMBDA_0``.

Splitting the amplitude out this way lets the optimiser solve for the optimal
amplitude of every source *directly*: the forward model is exactly linear in
amplitude, so it becomes a linear least-squares problem (see
``scripts.inference.solve_log_amplitudes``).
"""

import equinox as eqx
import jax.numpy as jnp

from .constants import HC_JAX, KB_JAX

# ---------------------------------------------------------------------------
# Amplitude / normalisation conventions
# ---------------------------------------------------------------------------

#: Reference wavelength (microns) at which every spectrum model's shape is
#: normalised to ``f_lambda = 1``.  This is a single GLOBAL constant shared by
#: all bands so that the modelled spectrum stays smooth in amplitude across
#: the whole wavelength range: changing band only changes the *sampling* of
#: the shape, never its normalisation.
LAMBDA_0 = 1.0

#: Index of the log-amplitude column in ``source_params``.  By convention the
#: amplitude is ALWAYS the first parameter; the remaining columns
#: (``source_params[:, 1:]``) are the ones passed to the spectrum model.
LOG_AMPLITUDE_INDEX = 0


def split_source_params(source_params: jnp.ndarray):
    """Split ``source_params`` into ``(log_amplitude, shape_params)``.

    Parameters
    ----------
    source_params : jnp.ndarray, shape (..., 1 + P)
        Per-source parameters with the log-amplitude in the first column.

    Returns
    -------
    log_amplitude : jnp.ndarray, shape (..., 1)
    shape_params : jnp.ndarray, shape (..., P)
    """
    start = LOG_AMPLITUDE_INDEX
    log_amplitude = source_params[..., start:start + 1]
    shape_params = source_params[..., start + 1:]
    return log_amplitude, shape_params


def join_source_params(log_amplitude: jnp.ndarray, shape_params: jnp.ndarray):
    """Inverse of :func:`split_source_params` (amplitude first)."""
    return jnp.concatenate([log_amplitude, shape_params], axis=-1)


class BlackbodySpectrum(eqx.Module):
    """Normalised blackbody spectrum template (shape only).

    Returns the Planck function normalised to ``f_lambda = 1`` at the
    reference wavelength ``lambda_0``::

        shape(lambda) = B_lambda(lambda, T) / B_lambda(lambda_0, T)

    The result is dimensionless; the source's physical amplitude is supplied
    separately as the first column of ``source_params`` (see the module
    docstring).  The 2hc^2 prefactor cancels in the ratio, so the returned
    shape is::

        (lambda_0 / lambda)^5 * (exp(x_0) - 1) / (exp(x) - 1)

    with ``x = hc / (lambda * k_B * T)``.

    Parameters
    ----------
    wavelength : jnp.ndarray, shape (...,)
        Wavelength(s) in microns.
    shape_params : jnp.ndarray, shape (..., 1)
        Per-source shape parameters in **log-space**.
        ``shape_params[..., 0]`` = log(temperature / kK).

    Attributes
    ----------
    lambda_0 : float
        Reference wavelength (microns).  Defaults to the global constant
        :data:`LAMBDA_0` and should be the same for every band, so that the
        normalisation does not introduce band-to-band discontinuities.
    n_params : int
        Number of spectrum-*shape* parameters (1: log-temperature).  The
        amplitude is not counted here (it is held outside the model).
    """

    lambda_0: float = LAMBDA_0
    n_params: int = 1

    def __call__(
        self,
        wavelength: jnp.ndarray,
        shape_params: jnp.ndarray,
    ) -> jnp.ndarray:
        # Slice with a trailing dummy axis for broadcasting.
        log_temperature = shape_params[..., 0:1]   # log(kK)  (..., 1)
        temperature = jnp.exp(log_temperature)     # kK

        # Exponent x = hc / (lambda * k_B * T), written via a pre-factored
        # inv_T_scale = hc / (lambda * k_B) so that its gradient w.r.t. T
        # becomes -inv_T_scale / T^2 - avoiding the (k_B * T)^2 term that
        # underflows float32 at T ~ 5 kK (see explanatory supplement 7.10).
        inv_T_scale = HC_JAX / (wavelength * KB_JAX)             # K
        inv_T_scale_0 = HC_JAX / (self.lambda_0 * KB_JAX)        # K

        exponent = inv_T_scale / temperature                      # dimensionless
        exponent_0 = inv_T_scale_0 / temperature                  # dimensionless

        # Guard exp overflow (short wavelengths / low temperatures).
        exponent = jnp.clip(exponent, None, 50.0)
        exponent_0 = jnp.clip(exponent_0, None, 50.0)

        # Ratio of Planck functions; the 2 h c^2 / lambda^5 prefactor cancels.
        shape = (
            (self.lambda_0 / wavelength) ** 5
            * jnp.expm1(exponent_0)
            / jnp.expm1(exponent)
        )
        return shape
