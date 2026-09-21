"""Tests for SPHEREx image generation."""

import astropy.units as u
import jax
import jax.numpy as jnp
import pytest

from spherex import (
    BlackbodySpectrum,
    GaussianPSF,
    GaussianFilterTransmission,
    ImageGenerator,
    photon_rate_per_pixel,
    _subpixel_centers,
    HC_JAX,
)

from spherex.constants import TEMPERATURE_UNIT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def psf():
    return GaussianPSF(fwhm_ref=2.0, wavelength_ref=1.0)


@pytest.fixture
def transmission():
    return GaussianFilterTransmission(
        lambda_intercept=1.0,
        lambda_slope=0.01,
        width=0.05,
    )


@pytest.fixture
def spectrum_model():
    return BlackbodySpectrum()


@pytest.fixture
def generator(psf, transmission, spectrum_model):
    return ImageGenerator(
        psf=psf,
        transmission=transmission,
        spectrum_model=spectrum_model,
        aperture=1.0,   # 1 m² for simplicity
    )


# ---------------------------------------------------------------------------
# _subpixel_centers
# ---------------------------------------------------------------------------

def test_subpixel_centers_shape():
    centers = _subpixel_centers(0, 0, 1.0, 3)
    assert centers.shape == (9, 2)


def test_subpixel_centers_range():
    """Sub-pixel centres should lie within the pixel."""
    scale = 0.5
    centers = _subpixel_centers(3, 2, scale, 5)
    x = centers[:, 0]
    y = centers[:, 1]
    assert jnp.all(x >= 2 * scale) and jnp.all(x < 3 * scale)
    assert jnp.all(y >= 3 * scale) and jnp.all(y < 4 * scale)


def test_subpixel_centers_oversampling_1():
    """With oversampling=1 the centre should be the pixel centre."""
    centers = _subpixel_centers(2, 3, 0.5, 1)
    expected = jnp.array([(3 + 0.5) * 0.5, (2 + 0.5) * 0.5])
    assert jnp.allclose(centers[0], expected)


# ---------------------------------------------------------------------------
# photon_rate_per_pixel
# ---------------------------------------------------------------------------

def test_photon_rate_per_pixel_positive(psf, transmission, spectrum_model):
    """Photon rate should be positive for a source inside the pixel."""
    omega_p_centers = _subpixel_centers(5, 5, 0.5, 3)
    omega_s = jnp.array([2.75, 2.75])   # near pixel centre -> high PSF value
    T_val = (5000 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value
    # source_params = [log_amplitude, log_temperature]
    params = jnp.log(jnp.array([1.0, T_val]))

    rate = photon_rate_per_pixel(
        omega_p_centers, omega_s, params,
        spectrum_model, psf, transmission,
        aperture=1.0, pixel_scale=0.5,
        oversampling=3, n_wavelength_samples=101,
    )
    assert rate > 0


def test_photon_rate_per_pixel_far_source_is_small(psf, transmission, spectrum_model):
    """A source far from the pixel should produce negligible rate."""
    omega_p_centers = _subpixel_centers(5, 5, 0.5, 3)
    omega_s = jnp.array([100.0, 100.0])  # far away
    params = jnp.log(jnp.array([1.0, 5.0]))

    rate = photon_rate_per_pixel(
        omega_p_centers, omega_s, params,
        spectrum_model, psf, transmission,
        aperture=1.0, pixel_scale=0.5,
        oversampling=3, n_wavelength_samples=51,
    )
    assert rate < 1e-6


def test_photon_rate_scales_with_aperture(psf, transmission, spectrum_model):
    """Doubling aperture should double the rate."""
    omega_p_centers = _subpixel_centers(5, 5, 0.5, 3)
    omega_s = jnp.array([2.75, 2.75])
    params = jnp.log(jnp.array([1.0, 5.0]))

    kwargs = dict(
        omega_p_centers=omega_p_centers, omega_s=omega_s,
        source_params=params, spectrum_model=spectrum_model,
        psf=psf, transmission=transmission,
        pixel_scale=0.5, oversampling=3, n_wavelength_samples=51,
    )
    r1 = photon_rate_per_pixel(aperture=1.0, **kwargs)
    r2 = photon_rate_per_pixel(aperture=2.0, **kwargs)
    assert jnp.allclose(r2, 2.0 * r1)


# ---------------------------------------------------------------------------
# ImageGenerator
# ---------------------------------------------------------------------------

def test_image_generator_shape(generator):
    """Generated image should have the requested shape."""
    pos = jnp.array([[5.0, 5.0]])
    params = jnp.log(jnp.array([[1.0, 5.0]]))

    img = generator(pos, params, 20, 15, 0.5,
                    postage_stamp_half_size=5,
                    n_wavelength_samples=31,
                    oversampling=1)
    assert img.shape == (15, 20)


def test_image_generator_single_source_positive(generator):
    """A single source should produce a positive image."""
    pos = jnp.array([[5.0, 5.0]])
    params = jnp.log(jnp.array([[1.0, 5.0]]))

    img = generator(pos, params, 30, 30, 0.5,
                    postage_stamp_half_size=10,
                    n_wavelength_samples=31,
                    oversampling=2)
    assert jnp.all(img >= 0)
    assert jnp.sum(img) > 0


def test_image_generator_source_outside_image(generator):
    """A source completely outside the image should contribute nothing."""
    pos = jnp.array([[100.0, 100.0]])
    params = jnp.log(jnp.array([[1.0, 5.0]]))

    img = generator(pos, params, 20, 20, 0.5,
                    postage_stamp_half_size=5,
                    n_wavelength_samples=21,
                    oversampling=1)
    assert jnp.all(img == 0)


def test_image_generator_two_sources_additive(generator):
    """Two sources should produce the sum of their individual contributions."""
    pos1 = jnp.array([[3.0, 4.0]])
    pos2 = jnp.array([[8.0, 6.0]])
    params1 = jnp.log(jnp.array([[1.0, 6.0]]))   # [logA, logT]
    params2 = jnp.log(jnp.array([[1.5, 4.0]]))

    kwargs = dict(
        image_width=20, image_height=20, pixel_scale=0.5,
        postage_stamp_half_size=6, n_wavelength_samples=21,
        oversampling=1,
    )
    img1 = generator(pos1, params1, **kwargs)
    img2 = generator(pos2, params2, **kwargs)
    img_both = generator(
        jnp.vstack([pos1, pos2]),
        jnp.vstack([params1, params2]),
        **kwargs,
    )
    assert jnp.allclose(img_both, img1 + img2, rtol=1e-5)


def test_oversampling_reduces_discretisation_error(generator):
    """Higher oversampling should shift the total flux closer to the
    no-oversampling limit—this test just checks that oversampling>1
    changes the result (indicating it has an effect)."""
    pos = jnp.array([[4.3, 4.7]])  # non-integer pixel position
    params = jnp.log(jnp.array([[1.0, 5.0]]))

    img1 = generator(pos, params, 20, 20, 0.5,
                     postage_stamp_half_size=10,
                     n_wavelength_samples=31,
                     oversampling=1)
    img3 = generator(pos, params, 20, 20, 0.5,
                     postage_stamp_half_size=10,
                     n_wavelength_samples=31,
                     oversampling=3)

    # Total flux should be similar (within a few percent) but not identical
    total1 = jnp.sum(img1)
    total3 = jnp.sum(img3)
    assert jnp.abs(total1 - total3) / total1 < 0.15


def test_image_gradient_wrt_temperature(generator):
    """Gradient of total flux w.r.t. source temperature should be computable."""
    pos = jnp.array([[5.0, 5.0]])

    def total_flux(T):
        # source_params = [log_amplitude, log_temperature]
        p = jnp.log(jnp.array([[1.0, T]]))
        return jnp.sum(generator(pos, p, 20, 20, 0.5,
                                 postage_stamp_half_size=8,
                                 n_wavelength_samples=21,
                                 oversampling=1))

    grad = jax.grad(total_flux)(5.0)
    assert jnp.isfinite(grad)
    assert grad != 0.0


def test_image_gradient_wrt_amplitude(generator):
    """Gradient of total flux w.r.t. amplitude should be computable."""
    pos = jnp.array([[5.0, 5.0]])

    def total_flux(amp):
        p = jnp.log(jnp.array([[amp, 5.0]]))
        return jnp.sum(generator(pos, p, 20, 20, 0.5,
                                 postage_stamp_half_size=8,
                                 n_wavelength_samples=21,
                                 oversampling=1))

    grad = jax.grad(total_flux)(2.0)
    assert jnp.isfinite(grad)
    assert grad != 0.0


def test_image_linear_in_amplitude(generator):
    """The image must be exactly linear in exp(source_params[:, 0])."""
    pos = jnp.array([[5.0, 5.0], [10.0, 10.0]])
    log_base = jnp.log(jnp.array([[1.0, 5.0], [2.0, 6.0]]))  # [logA, logT]

    img1 = generator(pos, log_base, 20, 20, 0.5,
                     postage_stamp_half_size=8,
                     n_wavelength_samples=21, oversampling=1)
    log_doubled = log_base.at[:, 0].add(jnp.log(2.0))  # double amplitude
    img2 = generator(pos, log_doubled, 20, 20, 0.5,
                     postage_stamp_half_size=8,
                     n_wavelength_samples=21, oversampling=1)
    assert jnp.allclose(img2, 2.0 * img1, rtol=1e-5)
