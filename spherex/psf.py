"""Point-spread function models for SPHEREx photometry.

All angular quantities are in **arcseconds** (positions) and
**arcsec^-2** (PSF value).  The PSF is normalised so that its integral
over the whole detector plane (:math:`\\mathbb{R}^2` in arcsec²) is 1.
"""

import equinox as eqx
import jax.numpy as jnp


class GaussianPSF(eqx.Module):
    """Circular Gaussian PSF whose FWHM is proportional to wavelength.

    Parameters
    ----------
    fwhm_ref : float
        FWHM at the reference wavelength, in arcsec.
    wavelength_ref : float
        Reference wavelength in microns.

    Call signature
    --------------
    __call__(omega_p, omega_s, wavelength) -> value in arcsec^-2

    omega_p : jnp.ndarray, shape (..., 2)
        Detector-plane positions in arcsec.
    omega_s : jnp.ndarray, shape (..., 2)
        Source positions in arcsec.
    wavelength : jnp.ndarray, shape (...,)
        Wavelength(s) in microns.  Broadcasts against the leading
        dimensions of ``omega_p`` and ``omega_s``.
    """

    fwhm_ref: float
    wavelength_ref: float

    def __call__(
        self,
        omega_p: jnp.ndarray,
        omega_s: jnp.ndarray,
        wavelength: jnp.ndarray,
    ) -> jnp.ndarray:
        # FWHM(lambda)  [arcsec]
        fwhm = self.fwhm_ref * (wavelength / self.wavelength_ref)
        # FWHM -> sigma:  sigma = FWHM / (2 * sqrt(2 * ln(2)))
        sigma = fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))   # arcsec

        # Squared Euclidean distance  [arcsec^2]
        d2 = jnp.sum((omega_p - omega_s) ** 2, axis=-1)

        # 2-D normalised Gaussian  [arcsec^-2]
        #   (1 / (2 pi sigma^2)) * exp(-d2 / (2 sigma^2))
        norm = 1.0 / (2.0 * jnp.pi * sigma**2)
        return norm * jnp.exp(-0.5 * d2 / sigma**2)
