"""SPHEREx PSF photometry – modular JAX + Equinox pipeline."""

from .constants import (H, C, HC, KB, H_JAX, C_JAX, HC_JAX, KB_JAX,
                         TEMPERATURE_UNIT, WAVELENGTH_UNIT)
from .spectrum import (
    BlackbodySpectrum,
    NeuralNetSpectrum,
    LAMBDA_0,
    LOG_AMPLITUDE_INDEX,
    split_source_params,
    join_source_params,
)
from .psf import GaussianPSF
from .transmission import GaussianFilterTransmission
from .image import ImageGenerator, photon_rate_per_pixel, _subpixel_centers
from .config import SpherexImageGenerator, SpherexImageGenerator3
from .image2 import ImageGenerator2
from .image3 import ImageGenerator3

__all__ = [
    "H", "C", "HC", "KB",
    "H_JAX", "C_JAX", "HC_JAX", "KB_JAX",
    "TEMPERATURE_UNIT", "WAVELENGTH_UNIT",
    "BlackbodySpectrum",
    "NeuralNetSpectrum",
    "LAMBDA_0",
    "LOG_AMPLITUDE_INDEX",
    "split_source_params",
    "join_source_params",
    "GaussianPSF",
    "GaussianFilterTransmission",
    "ImageGenerator",
    "SpherexImageGenerator",
    "ImageGenerator2",
    "ImageGenerator3",
    "SpherexImageGenerator3",
    "photon_rate_per_pixel",
    "_subpixel_centers",
]
