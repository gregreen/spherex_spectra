"""Optimised image generation for SPHEREx photometry (``image2``).

.. deprecated::
    This module is FROZEN and considered outdated/legacy.  It still uses
    the old fixed-``linspace`` + trapezoid wavelength quadrature, which
    ``spherex.image`` and ``spherex.image3`` have since replaced with a
    quantile / inverse-CDF importance-sampling estimator (far more
    accurate at low ``n_wavelength_samples`` - see
    ``spherex.image._one_subpixel_rate``).  It is not part of the main
    pipeline (``spherex.config`` builds generators from ``image``/
    ``image3`` only) and is kept only as a historical reference for its
    hoisting optimisations.  Do not use it for new work; it has not been
    updated to the new ``transmission.quantile`` /
    ``transmission.total_transmission`` interface and its per-stamp
    grid-hoisting trick is fundamentally incompatible with per-sub-pixel
    quantile sampling (each sub-pixel has a different ``omega_p``, hence
    a different quantile location).

    It also predates two later conventions and was NOT updated for them:
    the amplitude-first ``source_params`` layout (it never applies
    ``exp(source_params[..., 0])`` at all) and the log-flux spectrum-model
    contract of ``spherex.spectrum`` (it treats the model's output as a
    linear flux).  Its output amplitudes are therefore WRONG; use
    ``ImageGenerator`` or ``ImageGenerator3``.

Same interface as ``spherex.image`` but with two algorithmic improvements
that preserve mathematical correctness:

1. **Hoisted blackbody** – the source spectrum  (λ/hc)·f_λ(λ)  is
   evaluated *once per source* on a single wavelength grid, rather than
   once per sub-pixel.

2. **Single wavelength grid per stamp** – one ``linspace`` covers the
   full λ-range of the entire postage stamp (±5σ around the extremal
   λ_c values), shared across all sub-pixels.

References
----------
See ``spherex.image`` for the original (unoptimised) implementation.
"""

import equinox as eqx
import jax
from jax.scipy.integrate import trapezoid
import jax.numpy as jnp

from .constants import HC_JAX

# ---------------------------------------------------------------------------
# Helpers  (identical to image.py)
# ---------------------------------------------------------------------------

def _subpixel_centers(
    i: int,
    j: int,
    pixel_scale: float,
    oversampling: int,
) -> jnp.ndarray:
    """Return the angular positions of sub-pixel centres for pixel ``(i, j)``.

    Pixel ``(i, j)`` covers the angular region
    ``[j * scale, (j+1) * scale] x [i * scale, (i+1) * scale]``.
    """
    K = oversampling
    offsets = (jnp.arange(K) + 0.5) / K
    x_offsets, y_offsets = jnp.meshgrid(offsets, offsets)
    x = (j + x_offsets) * pixel_scale
    y = (i + y_offsets) * pixel_scale
    return jnp.stack([x.ravel(), y.ravel()], axis=-1)        # (K², 2)


# ---------------------------------------------------------------------------
# Optimised image generator
# ---------------------------------------------------------------------------

class ImageGenerator2(eqx.Module):
    """Optimised :class:`~spherex.image.ImageGenerator`.

    Same ``__call__`` signature, same mathematical result, but ~1.5–2×
    faster thanks to hoisted blackbody evaluation and a single
    wavelength grid shared across all sub-pixels in a postage stamp.
    """

    psf: eqx.Module
    transmission: eqx.Module
    spectrum_model: eqx.Module
    aperture: float

    def __call__(
        self,
        source_positions: jnp.ndarray,      # (S, 2)  [arcsec]
        source_params: jnp.ndarray,         # (S, P)
        image_width: int,
        image_height: int,
        pixel_scale: float,
        postage_stamp_half_size: int,
        n_wavelength_samples: int,
        oversampling: int = 1,
    ) -> jnp.ndarray:
        half = postage_stamp_half_size
        stamp_size = 2 * half + 1
        K = oversampling
        K2 = K * K

        # ---- static grids -------------------------------------------------
        di = jnp.arange(-half, half + 1)
        dj = jnp.arange(-half, half + 1)
        di_grid, dj_grid = jnp.meshgrid(di, dj, indexing="ij")

        sub_off = (jnp.arange(K) + 0.5) / K
        sx, sy = jnp.meshgrid(sub_off, sub_off)
        sx = sx.ravel()
        sy = sy.ravel()

        width = self.transmission.width
        lam_intercept = self.transmission.lambda_intercept
        lam_slope = self.transmission.lambda_slope

        # ---- per-source scan body -----------------------------------------
        def _one_stamp(image, src):
            omega_s, params_s = src

            jc = jnp.floor(omega_s[0] / pixel_scale).astype(jnp.int32)
            ic = jnp.floor(omega_s[1] / pixel_scale).astype(jnp.int32)

            j_idx = jc + dj_grid
            i_idx = ic + di_grid

            valid = (
                (j_idx >= 0) & (j_idx < image_width)
                & (i_idx >= 0) & (i_idx < image_height)
            )

            j_safe = jnp.clip(j_idx, 0, image_width - 1)
            i_safe = jnp.clip(i_idx, 0, image_height - 1)

            # ---- single λ grid covering the full stamp --------------------
            # λ_c at the top and bottom rows of the (unclipped) stamp
            y_bottom = (ic - half) * pixel_scale
            y_top = (ic + half) * pixel_scale
            lam_c_bottom = lam_intercept + lam_slope * y_bottom
            lam_c_top = lam_intercept + lam_slope * y_top

            lam_min = jnp.minimum(lam_c_bottom, lam_c_top) - 5.0 * width
            lam_max = jnp.maximum(lam_c_bottom, lam_c_top) + 5.0 * width

            lambdas = jnp.linspace(lam_min, lam_max,
                                   n_wavelength_samples)     # (N_λ,)

            # ---- precompute blackbody once per source ----------------------
            f_lam = self.spectrum_model(lambdas, params_s)    # (N_λ,)
            f_weighted = (lambdas / HC_JAX) * f_lam            # (N_λ,)

            # ---- sub-pixel positions (same as image.py) -------------------
            x = (j_idx[..., None] + sx) * pixel_scale         # (S, S, K2)
            y = (i_idx[..., None] + sy) * pixel_scale
            omega_p_all = jnp.stack([x, y], axis=-1)           # (S, S, K2, 2)
            omega_p_flat = omega_p_all.reshape(-1, 2)           # (S*S*K2, 2)

            # ---- per-subpixel: only PSF × T (blackbody is hoisted) --------
            # We vmap over sub-pixels; inside we only evaluate PSF & T.
            def _subpix_integrand(op):
                psf_val = self.psf(op, omega_s, lambdas)      # (N_λ,)
                t_val = self.transmission(lambdas, op)         # (N_λ,)
                return trapezoid(f_weighted * psf_val * t_val,
                                 lambdas)                      # scalar

            all_rates = jax.vmap(_subpix_integrand)(omega_p_flat)  # (S*S*K2,)

            # ---- average sub-pixels, scale, scatter-add -------------------
            rates = jnp.mean(
                all_rates.reshape(stamp_size * stamp_size, K2), axis=1
            )
            stamp = rates.reshape(stamp_size, stamp_size)
            stamp = stamp * (self.aperture * pixel_scale**2)

            stamp_masked = jnp.where(valid, stamp, 0.0)
            image = image.at[i_safe, j_safe].add(stamp_masked)

            return image, None

        # ---- run the scan over all sources ---------------------------------
        image = jnp.zeros((image_height, image_width))
        final_image, _ = jax.lax.scan(
            _one_stamp, image,
            (source_positions, source_params),
        )

        return final_image
