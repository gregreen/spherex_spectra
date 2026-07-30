"""Image generation from point-source models for SPHEREx photometry.

Provides the numerical wavelength-integration of the photon-detection rate
for a single pixel (with oversampling) and an ``ImageGenerator`` that
composites postage stamps from multiple sources into a full detector image.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .constants import HC_JAX


# ---------------------------------------------------------------------------
# Helper
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

    Parameters
    ----------
    i : int
        Pixel row index (0-based, bottom of image).
    j : int
        Pixel column index (0-based, left of image).
    pixel_scale : float
        Pixel size in arcsec.
    oversampling : int
        Number of sub-divisions per pixel edge (K).  Produces K² sub-pixels.

    Returns
    -------
    jnp.ndarray, shape ``(oversampling**2, 2)``
        (x, y) positions of sub-pixel centres in arcsec.
    """
    K = oversampling
    # Offsets within the pixel: (u + 0.5) / K  for u = 0 .. K-1
    offsets = (jnp.arange(K) + 0.5) / K   # shape (K,)

    # Build a grid of (x, y) for this pixel
    x_offsets, y_offsets = jnp.meshgrid(offsets, offsets)  # both (K, K)
    x = (j + x_offsets) * pixel_scale                        # (K, K)
    y = (i + y_offsets) * pixel_scale                        # (K, K)

    return jnp.stack([x.ravel(), y.ravel()], axis=-1)        # (K², 2)


# ---------------------------------------------------------------------------
# Per-subpixel photon rate integrand (internal)
# ---------------------------------------------------------------------------

def _one_subpixel_rate(
    omega_p: jnp.ndarray,          # (2,)  – sub-pixel centre  [arcsec]
    omega_s: jnp.ndarray,          # (2,)  – source position   [arcsec]
    spectrum_params: jnp.ndarray,  # (P,)  – source parameters
    spectrum_model: eqx.Module,
    psf: eqx.Module,
    transmission: eqx.Module,
    n_wavelength_samples: int,
) -> jnp.ndarray:
    """Compute the wavelength integral for a single sub-pixel.

    Returns the *unscaled* integral (without the ``A * pixel_scale²``
    factor), i.e. ::

        ∫ dλ  (λ / hc) * f_λ(λ) * PSF(Ω_p) * T(λ | Ω_p)

    The integral is estimated via quantile / inverse-CDF importance
    sampling against the transmission profile itself, rather than fixed
    uniform-grid quadrature.  Writing ``norm(Ω_p) = ∫ T(λ|Ω_p) dλ`` and
    ``p(λ) = T(λ|Ω_p) / norm``, the integral becomes
    ``norm * E_{λ~p}[(λ/hc) f_λ(λ) PSF(λ)]``.  Sampling ``λ`` at evenly
    spaced quantiles of ``p`` (via ``transmission.quantile``) concentrates
    evaluations where the transmission profile actually has support,
    instead of wasting them in its tails - this is far more accurate than
    a fixed ``linspace`` + trapezoid rule at low sample counts, since the
    latter mostly samples the (near-zero) tails of a narrow bandpass.

    Requires ``transmission`` to implement ``quantile(q, omega_p)`` (the
    inverse CDF of its normalised profile) and ``total_transmission
    (omega_p)`` (the profile's integral over wavelength) - see
    ``spherex.transmission`` for the interface contract.
    """
    # Evenly spaced quantiles in the interior of (0, 1) (midpoint rule),
    # avoiding q=0/1 where the inverse CDF can diverge.
    q = (jnp.arange(n_wavelength_samples) + 0.5) / n_wavelength_samples
    lambdas = transmission.quantile(q, omega_p)               # (N_λ,)
    norm = transmission.total_transmission(omega_p)           # scalar

    f_lam = spectrum_model(lambdas, spectrum_params)          # (N_λ,)
    psf_val = psf(omega_p, omega_s, lambdas)                  # (N_λ,)

    integrand = (lambdas / HC_JAX) * f_lam * psf_val          # (N_λ,)
    return norm * jnp.mean(integrand)                         # scalar


# ---------------------------------------------------------------------------
# Photon rate for one pixel (with oversampling)
# ---------------------------------------------------------------------------

def photon_rate_per_pixel(
    omega_p_centers: jnp.ndarray,     # (K², 2)
    omega_s: jnp.ndarray,             # (2,)
    spectrum_params: jnp.ndarray,     # (P,)
    spectrum_model: eqx.Module,
    psf: eqx.Module,
    transmission: eqx.Module,
    aperture: float,
    pixel_scale: float,
    oversampling: int,
    n_wavelength_samples: int,
) -> jnp.ndarray:
    """Photon detection rate in one detector pixel for a single source.

    Approximates Eq. (2) of the SPHEREx photometry document with
    *oversampling* of the pixel area::

        Ndot ≈ A · (pixel_scale)² · (1/K²) Σ_uv ∫ dλ (λ/hc) f_λ PSF T

    Parameters
    ----------
    omega_p_centers : shape ``(K², 2)``
        Sub-pixel centre positions in arcsec (from ``_subpixel_centers``).
    omega_s : shape ``(2,)``
        Source position in arcsec.
    spectrum_params : shape ``(n_params,)``
        Parameters for the spectrum model (e.g. [T, amplitude]).
    spectrum_model : equinox.Module
        Spectrum template (e.g. ``BlackbodySpectrum``).
    psf : equinox.Module
        PSF model.
    transmission : equinox.Module
        Filter transmission model.
    aperture : float
        Telescope aperture in m².
    pixel_scale : float
        Pixel size in arcsec.
    oversampling : int
        Number of sub-divisions per pixel edge (K).
    n_wavelength_samples : int
        Number of wavelength grid points for the λ-integral.

    Returns
    -------
    jnp.ndarray (scalar)
        Photon detection rate in s⁻¹.
    """
    # Integrate over wavelength for each sub-pixel centre
    sub_rates = jax.vmap(
        _one_subpixel_rate,
        in_axes=(0, None, None, None, None, None, None),
    )(
        omega_p_centers,          # (K², 2) -> vmap over first axis
        omega_s,
        spectrum_params,
        spectrum_model,
        psf,
        transmission,
        n_wavelength_samples,
    )                                                         # (K²,)

    mean_sub_rate = jnp.mean(sub_rates)                       # scalar
    return aperture * pixel_scale**2 * mean_sub_rate


# ---------------------------------------------------------------------------
# Full-image generator
# ---------------------------------------------------------------------------

class ImageGenerator(eqx.Module):
    """Generate a SPHEREx detector image from a list of point sources.

    For each source a small *postage stamp* is computed and added into
    the full image.  The stamp size is controlled by
    ``postage_stamp_half_size`` (in pixels) and should be several times
    larger than the PSF FWHM to capture the full flux.

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

    Call signature
    --------------
    __call__(source_positions, source_params, image_width, image_height,
             pixel_scale, postage_stamp_half_size, n_wavelength_samples,
             oversampling=1) -> image (H, W) in s⁻¹
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

        # ---- static grids (known at trace time via Python ints) -----------
        di = jnp.arange(-half, half + 1)
        dj = jnp.arange(-half, half + 1)
        di_grid, dj_grid = jnp.meshgrid(di, dj, indexing="ij")

        sub_off = (jnp.arange(K) + 0.5) / K
        sx, sy = jnp.meshgrid(sub_off, sub_off)
        sx = sx.ravel()
        sy = sy.ravel()

        # ---- per-source scan body -----------------------------------------
        def _one_stamp(image, src):
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
            image = image.at[i_safe, j_safe].add(stamp_masked)

            return image, None

        # ---- run the scan over all sources ---------------------------------
        image = jnp.zeros((image_height, image_width))
        final_image, _ = jax.lax.scan(
            _one_stamp, image,
            (source_positions, source_params),
        )

        return final_image

