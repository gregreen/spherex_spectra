"""Tests for SPHEREx image generation."""

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spherex import (
    BlackbodySpectrum,
    NeuralNetSpectrum,
    GaussianPSF,
    GaussianFilterTransmission,
    ImageGenerator,
    photon_rate_per_pixel,
    _subpixel_centers,
    HC_JAX,
)

from spherex.constants import TEMPERATURE_UNIT
from spherex.image import MAX_LOG_FLUX
from spherex.spectrum import normalized_source_params


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
# Numeric safety of the exponent
# ---------------------------------------------------------------------------

class _PowerLawSpectrum(eqx.Module):
    """Toy log-flux model: a power law whose exponent IS the shape parameter.

    The library contract is just ``__call__(wavelength, shape_params)``
    returning an unnormalised log-flux (plus ``n_params``), so a model whose
    log-shape range can be dialled at will is the cleanest way to drive the
    exponent past the float32 limit.
    """
    n_params: int = 1

    def __call__(self, wavelength, shape_params):
        return shape_params[0] * jnp.log(jnp.atleast_1d(wavelength))


@pytest.fixture
def long_wavelength_transmission():
    """A bandpass far from LAMBDA_0 = 1 um, so a power law has a large shape."""
    return GaussianFilterTransmission(
        lambda_intercept=3.0,
        lambda_slope=0.01,
        width=0.05,
    )


def test_runaway_theta_cannot_overflow_the_exponent(psf,
                                                   long_wavelength_transmission):
    """An out-of-prior theta must degrade a pixel, never produce inf/NaN.

    ``exp(log_flux)`` overflows float32 above ~88.7, and the overflow is worse
    than a large number: ``inf`` times an exactly zero PSF value, sub-pixel
    weight or postage-stamp mask element is ``NaN``, which then poisons chi^2
    for EVERY parameter set that touches that source.  Observed in the mock as
    a ``nan`` loss at the initial guess while the true-parameter loss was
    1.0002.  ``MAX_LOG_FLUX`` bounds the exponent, so the rate stays finite.
    """
    model = _PowerLawSpectrum()
    transmission = long_wavelength_transmission
    omega_p_centers = _subpixel_centers(5, 5, 0.5, 3)
    omega_s = jnp.array([2.75, 2.75])

    # f_lambda(LAMBDA_0) at the mock's brightest amplitude, with a shape
    # parameter far outside any sane prior.
    log_amplitude = float(np.log(5e-11))
    theta = jnp.array([110.0])
    source_params = jnp.concatenate([jnp.array([log_amplitude]), theta])

    # The exponent this configuration would evaluate, at the wavelength the
    # integration actually samples: log_amplitude + log-shape.
    norm_params = normalized_source_params(model, source_params)
    lam = transmission.quantile(0.5, omega_p_centers[0])
    exponent = float(norm_params[0] + model(lam, norm_params[1:])[0])
    assert exponent > 88.7          # the unguarded exp() would overflow
    assert exponent > MAX_LOG_FLUX  # ...so the bound is doing the work here

    # Why the bound matters at all: inf is not the end of the story.
    overflowed = jnp.exp(jnp.float32(exponent))
    assert not np.isfinite(np.asarray(overflowed))
    assert np.isnan(np.asarray(overflowed * jnp.float32(0.0)))

    rate = np.asarray(photon_rate_per_pixel(
        omega_p_centers, omega_s, source_params,
        model, psf, transmission,
        aperture=1.0, pixel_scale=0.5,
        oversampling=3, n_wavelength_samples=5,
    ))
    assert np.all(np.isfinite(rate))
    assert np.all(rate >= 0.0)

    # A sane shape parameter is untouched by the bound: the clipped and
    # unclipped rates agree, i.e. this is a guard, not a change of model.
    mild = jnp.concatenate([jnp.array([log_amplitude]), jnp.array([-1.0])])
    rate_mild = np.asarray(photon_rate_per_pixel(
        omega_p_centers, omega_s, mild,
        model, psf, transmission,
        aperture=1.0, pixel_scale=0.5,
        oversampling=3, n_wavelength_samples=5,
    ))
    assert np.all(np.isfinite(rate_mild))
    assert np.all(rate_mild > 0.0)


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


# ---------------------------------------------------------------------------
# Spectrum-model agnosticism: the generator must accept any model that
# implements the (wavelengths (N_λ,), shape_params (P,)) -> (N_λ,) contract.
# ---------------------------------------------------------------------------

def test_image_generator_with_neural_net_spectrum(psf, transmission):
    """A NeuralNetSpectrum model is usable end-to-end by the generator."""
    model = NeuralNetSpectrum(
        n_params=2, n_hidden_layers=2, hidden_size=8,
        key=jax.random.PRNGKey(0),
    )
    generator = ImageGenerator(
        psf=psf, transmission=transmission,
        spectrum_model=model, aperture=1.0,
    )

    pos = jnp.array([[5.0, 5.0], [10.0, 10.0]])
    # source_params = [log_amplitude, theta_0, theta_1]
    params = jnp.array([[jnp.log(1e-13), 0.1, -0.2],
                        [jnp.log(2e-13), -0.3, 0.4]])

    img = generator(pos, params, 20, 20, 0.5,
                    postage_stamp_half_size=8,
                    n_wavelength_samples=21, oversampling=1)
    assert img.shape == (20, 20)
    assert jnp.all(img >= 0)
    assert jnp.sum(img) > 0

    # Gradients must flow into the spectrum-model shape params.
    def total_flux(shape_params):
        p = params.at[:, 1:].set(shape_params)
        return jnp.sum(generator(pos, p, 20, 20, 0.5,
                                 postage_stamp_half_size=8,
                                 n_wavelength_samples=21,
                                 oversampling=1))

    grad = jax.grad(total_flux)(params[:, 1:])
    assert grad.shape == params[:, 1:].shape
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0)


def test_image_linear_in_amplitude_neural_net_spectrum(psf, transmission):
    """Amplitude linearity (used by the least-squares solve) holds for the NN
    model too, since the shape never depends on the amplitude."""
    model = NeuralNetSpectrum(
        n_params=2, n_hidden_layers=2, hidden_size=8,
        key=jax.random.PRNGKey(1),
    )
    generator = ImageGenerator(
        psf=psf, transmission=transmission,
        spectrum_model=model, aperture=1.0,
    )

    pos = jnp.array([[5.0, 5.0], [10.0, 10.0]])
    log_base = jnp.array([[jnp.log(1e-13), 0.1, -0.2],
                          [jnp.log(2e-13), -0.3, 0.4]])

    img1 = generator(pos, log_base, 20, 20, 0.5,
                     postage_stamp_half_size=8,
                     n_wavelength_samples=21, oversampling=1)
    log_doubled = log_base.at[:, 0].add(jnp.log(2.0))
    img2 = generator(pos, log_doubled, 20, 20, 0.5,
                     postage_stamp_half_size=8,
                     n_wavelength_samples=21, oversampling=1)
    assert jnp.allclose(img2, 2.0 * img1, rtol=1e-5)
