"""Faster generic image generation for SPHEREx photometry (``image3``).

Same mathematical model and public call signature as
:class:`~spherex.image.ImageGenerator`, but replaces the sequential
``jax.lax.scan`` over sources with a chunked batched-map
(``jax.lax.map(..., batch_size=...)``), followed by a single vectorised
scatter-add.  This lets XLA parallelise/fuse work across sources instead of
processing one postage stamp at a time, while keeping peak memory bounded by
the (tunable) ``source_batch_size``.

No assumptions are made about the functional form of the PSF, filter
transmission or spectrum models: they are called in exactly the same way as
in ``spherex.image``, so this module remains fully generic and compatible
with future (e.g. pixelized) PSF/transmission models.
"""

import equinox as eqx
import jax
from jax.scipy.integrate import trapezoid
import jax.numpy as jnp

from .constants import HC_JAX
from .image import _one_subpixel_rate


class ImageGenerator3(eqx.Module):
    """Chunked-batch reimplementation of :class:`~spherex.image.ImageGenerator`.

    Identical inputs/outputs to ``ImageGenerator``, plus an extra
    ``source_batch_size`` parameter on ``__call__`` that controls how many
    sources are processed in parallel at once (traded off against peak
    memory).  ``source_batch_size=None`` (the default) processes all sources
    in a single batch (full ``vmap``, no chunking).

    Parameters
    ----------
    psf : equinox.Module
        PSF model (shared across all sources).
    transmission : equinox.Module
        Filter transmission model (shared across all sources).
    spectrum_model : equinox.Module
        Spectrum template (shared; per-source parameters are passed at
        call time).
    aperture : float
        Telescope aperture in m².
    """

    psf: eqx.Module
    transmission: eqx.Module
    spectrum_model: eqx.Module
    aperture: float

    def __call__(
        self,
        source_positions: jnp.ndarray,      # (S, 2)  [pixel coords]
        source_params: jnp.ndarray,         # (S, P)
        image_width: int,
        image_height: int,
        pixel_scale: float,
        postage_stamp_half_size: int,
        n_wavelength_samples: int,
        oversampling: int = 1,
        source_batch_size: int | None = None,
    ) -> jnp.ndarray:
        """
        Returns
        -------
        jnp.ndarray, shape ``(image_height, image_width)``
            Photon detection rate per pixel in s⁻¹.
        """
        half = postage_stamp_half_size
        stamp_size = 2 * half + 1
        K = oversampling
        K2 = K * K
        S = source_positions.shape[0]

        # ---- static grids (known at trace time via Python ints) -----------
        di = jnp.arange(-half, half + 1)
        dj = jnp.arange(-half, half + 1)
        di_grid, dj_grid = jnp.meshgrid(di, dj, indexing="ij")

        sub_off = (jnp.arange(K) + 0.5) / K
        sx, sy = jnp.meshgrid(sub_off, sub_off)
        sx = sx.ravel()
        sy = sy.ravel()

        # ---- per-source stamp computation (no image carry) -----------------
        def _one_stamp(src):
            pixel_xy, params_s = src          # (2,) in pixel coordinates

            jc = jnp.floor(pixel_xy[0]).astype(jnp.int32)
            ic = jnp.floor(pixel_xy[1]).astype(jnp.int32)

            j_idx = jc + dj_grid
            i_idx = ic + di_grid

            valid = (
                (j_idx >= 0) & (j_idx < image_width)
                & (i_idx >= 0) & (i_idx < image_height)
            )

            j_safe = jnp.clip(j_idx, 0, image_width - 1)
            i_safe = jnp.clip(i_idx, 0, image_height - 1)

            x = (j_idx[..., None] + sx) * pixel_scale
            y = (i_idx[..., None] + sy) * pixel_scale
            omega_p_all = jnp.stack([x, y], axis=-1)
            omega_p_flat = omega_p_all.reshape(-1, 2)

            # Source position in arcsec (PSF needs angular units)
            omega_s = pixel_xy * pixel_scale

            all_rates = jax.vmap(
                _one_subpixel_rate,
                in_axes=(0, None, None, None, None, None, None),
            )(
                omega_p_flat, omega_s, params_s,
                self.spectrum_model, self.psf, self.transmission,
                n_wavelength_samples,
            )

            rates = jnp.mean(
                all_rates.reshape(stamp_size * stamp_size, K2), axis=1
            )
            stamp = rates.reshape(stamp_size, stamp_size)
            stamp = stamp * (self.aperture * pixel_scale**2)

            stamp_masked = jnp.where(valid, stamp, 0.0)
            return stamp_masked, i_safe, j_safe

        # ---- batched (chunked) map over all sources -------------------------
        batch_size = S if source_batch_size is None else min(source_batch_size, S)
        stamps, i_all, j_all = jax.lax.map(
            _one_stamp,
            (source_positions, source_params),
            batch_size=batch_size,
        )
        # stamps: (S, stamp_size, stamp_size)
        # i_all, j_all: (S, stamp_size, stamp_size)

        # ---- single vectorised scatter-add -----------------------------------
        image = jnp.zeros((image_height, image_width))
        image = image.at[i_all, j_all].add(stamps)

        return image
