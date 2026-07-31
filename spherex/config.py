"""Pre-configured SPHEREx bands for realistic image simulations.

Provides a ``SpherexImageGenerator`` subclass of
:class:`~spherex.image.ImageGenerator` that is pre-configured for one of
the six SPHEREx bands using published instrument parameters.

References
----------
* Instrument page: https://spherex.caltech.edu/page/instrument
* IRSA mission page: https://irsa.ipac.caltech.edu/Missions/spherex.html
"""

import jax.numpy as jnp

from .image import ImageGenerator
from .image3 import ImageGenerator3
from .psf import GaussianPSF
from .transmission import GaussianFilterTransmission
from .spectrum import BlackbodySpectrum


# ---------------------------------------------------------------------------
# SPHEREx instrument constants
# ---------------------------------------------------------------------------

# Telescope aperture  (20 cm diameter)
_APERTURE = jnp.pi * (0.10) ** 2   # m^2

# Detector: 2048 x 2048 pixels at 6.2 arcsec / pixel
_PIXEL_SCALE = 6.2                  # arcsec
_DETECTOR_PIXELS = 2048             # pixels along each axis
_DETECTOR_ARCSEC = _DETECTOR_PIXELS * _PIXEL_SCALE   # ~12698 arcsec

# PSF: ~6 arcsec FWHM at 1.0 um, proportional to wavelength
_PSF_FWHM_REF = 6.0                 # arcsec
_PSF_WAVELENGTH_REF = 1.0           # um

# ---------------------------------------------------------------------------
# Band definitions  (from spherex.caltech.edu/page/instrument)
# ---------------------------------------------------------------------------
# Each entry: (lambda_min, lambda_max, R, band_name)
# lambda_min / max in um, R = lambda / Delta_lambda

_BANDS = {
    1: (0.75, 1.09, 41, "Band 1"),
    2: (1.10, 1.62, 41, "Band 2"),
    3: (1.63, 2.41, 41, "Band 3"),
    4: (2.42, 3.82, 35, "Band 4"),
    5: (3.83, 4.41, 110, "Band 5"),
    6: (4.42, 5.00, 130, "Band 6"),
}


def _build_band(band, psf_scale=1.0, lambda_slope_scale=1.0):
    """Build PSF, transmission, spectrum model, and aperture for a band.

    Parameters
    ----------
    band : int
        SPHEREx band number (1-6).
    psf_scale : float, optional
        Factor by which to scale the PSF FWHM (default 1).
    lambda_slope_scale : float, optional
        Factor by which to scale the wavelength gradient (um/arcsec)
        of the linear variable filter (default 1).  Set to the
        downsampling factor when the detector image has been binned
        so that the wavelength range across the (fewer) pixels matches
        the full un-binned detector.

    Returns
    -------
    psf, transmission, spectrum_model, aperture
    """
    if band not in _BANDS:
        raise ValueError(f"Unknown SPHEREx band {band}.  Must be 1-6.")

    lam_min, lam_max, R, _name = _BANDS[band]

    # ---- PSF ---------------------------------------------------------------
    # FWHM proportional to wavelength, scaled for testing
    psf = GaussianPSF(
        fwhm_ref=_PSF_FWHM_REF * psf_scale,
        wavelength_ref=_PSF_WAVELENGTH_REF,
    )

    # ---- filter transmission -----------------------------------------------
    # LVF: lambda_c(y) = lambda_intercept + lambda_slope * y
    lambda_intercept = lam_min
    lambda_slope = (lam_max - lam_min) / _DETECTOR_ARCSEC * lambda_slope_scale   # um / arcsec

    # R = lambda / FWHM;  FWHM = 2*sqrt(2*ln(2))*sigma ~ 2.355*sigma
    lam_mid = 0.5 * (lam_min + lam_max)
    sigma = lam_mid / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)) * R)  # um

    transmission = GaussianFilterTransmission(
        lambda_intercept=lambda_intercept,
        lambda_slope=lambda_slope,
        width=sigma,
    )

    # ---- spectrum ----------------------------------------------------------
    spectrum_model = BlackbodySpectrum()

    return psf, transmission, spectrum_model, _APERTURE


# ---------------------------------------------------------------------------
# Pre-configured ImageGenerator subclass
# ---------------------------------------------------------------------------

class SpherexImageGenerator(ImageGenerator):
    """:class:`ImageGenerator` pre-configured for a specific SPHEREx band.

    Stores the detector dimensions and pixel scale so callers only need
    to supply source positions / parameters and the postage-stamp size.
    In the future WCS-related utilities can be added here as methods.

    Parameters
    ----------
    band : int
        SPHEREx band number (1-6).
    image_width : int, optional
        Detector width in pixels.  Default 2048.
    image_height : int, optional
        Detector height in pixels.  Default 2048.
    pixel_scale : float, optional
        Pixel size in arcsec.  Default 6.2.
    psf_scale : float, optional
        Factor scaling the PSF FWHM (default 1).  Larger values make
        the wavelength-dependent PSF effects more visually apparent.

    Call signature
    --------------
    __call__(source_positions, source_params, postage_stamp_half_size,
             n_wavelength_samples, oversampling=1) -> image (H, W) in s^-1
    """

    band: int
    image_width: int
    image_height: int
    pixel_scale: float

    def __init__(
        self,
        band,
        image_width=_DETECTOR_PIXELS,
        image_height=_DETECTOR_PIXELS,
        pixel_scale=_PIXEL_SCALE,
        psf_scale=1.0,
        lambda_slope_scale=1.0,
    ):
        psf, transmission, spectrum_model, aperture = _build_band(
            band, psf_scale=psf_scale,
            lambda_slope_scale=lambda_slope_scale,
        )

        super().__init__(
            psf=psf,
            transmission=transmission,
            spectrum_model=spectrum_model,
            aperture=aperture,
        )

        self.band = band
        self.image_width = image_width
        self.image_height = image_height
        self.pixel_scale = pixel_scale

    def __call__(
        self,
        source_positions,
        source_params,
        postage_stamp_half_size,
        n_wavelength_samples,
        oversampling=1,
    ):
        """Generate a SPHEREx detector image.

        Parameters
        ----------
        source_positions : (S, 2) array
            Source positions in arcsec (origin at bottom-left of pixel (0,0)).
        source_params : (S, P) array
            Per-source spectrum parameters.
        postage_stamp_half_size : int
            Half-size of the postage stamp in pixels.
        n_wavelength_samples : int
            Number of points for the wavelength integral.
        oversampling : int, optional
            Sub-pixel oversampling factor (default 1).

        Returns
        -------
        jnp.ndarray, shape (image_height, image_width)
            Photon detection rate per pixel in s^-1.
        """
        return super().__call__(
            source_positions,
            source_params,
            image_width=self.image_width,
            image_height=self.image_height,
            pixel_scale=self.pixel_scale,
            postage_stamp_half_size=postage_stamp_half_size,
            n_wavelength_samples=n_wavelength_samples,
            oversampling=oversampling,
        )


class SpherexImageGenerator3(ImageGenerator3):
    """:class:`ImageGenerator3` pre-configured for a specific SPHEREx band.

    Same as :class:`SpherexImageGenerator`, but based on the faster
    chunked-batch :class:`~spherex.image3.ImageGenerator3` implementation.
    Adds a ``source_batch_size`` parameter to ``__call__`` (see
    :class:`~spherex.image3.ImageGenerator3` for details).
    """

    band: int
    image_width: int
    image_height: int
    pixel_scale: float

    def __init__(
        self,
        band,
        image_width=_DETECTOR_PIXELS,
        image_height=_DETECTOR_PIXELS,
        pixel_scale=_PIXEL_SCALE,
        psf_scale=1.0,
        lambda_slope_scale=1.0,
    ):
        psf, transmission, spectrum_model, aperture = _build_band(
            band, psf_scale=psf_scale,
            lambda_slope_scale=lambda_slope_scale,
        )

        super().__init__(
            psf=psf,
            transmission=transmission,
            spectrum_model=spectrum_model,
            aperture=aperture,
        )

        self.band = band
        self.image_width = image_width
        self.image_height = image_height
        self.pixel_scale = pixel_scale

    def __call__(
        self,
        source_positions,
        source_params,
        postage_stamp_half_size,
        n_wavelength_samples,
        oversampling=1,
        source_batch_size=None,
    ):
        """Generate a SPHEREx detector image.

        Parameters
        ----------
        source_positions : (S, 2) array
            Source positions in arcsec (origin at bottom-left of pixel (0,0)).
        source_params : (S, P) array
            Per-source spectrum parameters.
        postage_stamp_half_size : int
            Half-size of the postage stamp in pixels.
        n_wavelength_samples : int
            Number of points for the wavelength integral.
        oversampling : int, optional
            Sub-pixel oversampling factor (default 1).
        source_batch_size : int, optional
            Number of sources processed per batch (default: all at once).

        Returns
        -------
        jnp.ndarray, shape (image_height, image_width)
            Photon detection rate per pixel in s^-1.
        """
        return super().__call__(
            source_positions,
            source_params,
            image_width=self.image_width,
            image_height=self.image_height,
            pixel_scale=self.pixel_scale,
            postage_stamp_half_size=postage_stamp_half_size,
            n_wavelength_samples=n_wavelength_samples,
            oversampling=oversampling,
            source_batch_size=source_batch_size,
        )
