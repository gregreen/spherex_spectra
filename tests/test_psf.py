"""Tests for SPHEREx PSF models."""

import jax
import jax.numpy as jnp
import pytest

from spherex.psf import GaussianPSF


@pytest.fixture
def psf():
    """Default PSF: 2 arcsec FWHM at 1 um, growing 0.1 arcsec/um."""
    return GaussianPSF(fwhm_ref=2.0, wavelength_ref=1.0)


def test_psf_output_scalar(psf):
    """Scalar inputs should return a scalar."""
    omega_p = jnp.array([0.0, 0.0])
    omega_s = jnp.array([0.0, 0.0])
    lam = jnp.array(1.0)
    result = psf(omega_p, omega_s, lam)
    assert result.shape == ()


def test_psf_peak_value(psf):
    """At the source position the PSF should equal 1 / (2 pi sigma^2)."""
    omega_s = jnp.array([0.0, 0.0])
    lam = jnp.array(1.0)   # reference wavelength -> FWHM = 2.0 arcsec

    # sigma = FWHM / (2 sqrt(2 ln 2)) = 2.0 / 2.35482... ≈ 0.8493
    # peak = 1 / (2 pi sigma^2)
    fwhm_to_sigma = 1.0 / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    sigma = 2.0 * fwhm_to_sigma
    expected_peak = 1.0 / (2.0 * jnp.pi * sigma**2)

    peak = psf(omega_s, omega_s, lam)
    assert jnp.allclose(peak, expected_peak, rtol=1e-5)


def test_psf_fwhm_grows_with_wavelength(psf):
    """At longer wavelengths the PSF should be broader (peak lower)."""
    omega_s = jnp.array([0.0, 0.0])
    peak_ref = psf(omega_s, omega_s, jnp.array(1.0))
    peak_long = psf(omega_s, omega_s, jnp.array(3.0))
    assert peak_long < peak_ref


def test_psf_normalisation(psf):
    """Integral of the PSF over a large grid should be ~1."""
    # Create a large grid centred on the source
    N = 401
    half_size = 20.0   # arcsec — much larger than FWHM
    x = jnp.linspace(-half_size, half_size, N)
    y = jnp.linspace(-half_size, half_size, N)
    dx = x[1] - x[0]
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    omega_p = jnp.stack([X.ravel(), Y.ravel()], axis=-1)   # (N*N, 2)

    omega_s = jnp.array([0.0, 0.0])
    lam = jnp.array(1.0)

    values = psf(omega_p, omega_s, lam)                     # (N*N,)
    integral = jnp.sum(values) * dx**2
    assert jnp.allclose(integral, 1.0, rtol=1e-3)


def test_psf_batched_positions(psf):
    """Batched omega_p should produce matching output shape."""
    omega_p = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # (3, 2)
    omega_s = jnp.array([0.0, 0.0])
    lam = jnp.array(1.0)
    result = psf(omega_p, omega_s, lam)
    assert result.shape == (3,)


def test_psf_batched_wavelengths(psf):
    """Batched wavelengths should produce matching output shape."""
    omega_p = jnp.array([0.0, 0.0])
    omega_s = jnp.array([0.0, 0.0])
    lam = jnp.array([0.5, 1.0, 1.5])
    result = psf(omega_p, omega_s, lam)
    assert result.shape == (3,)


def test_psf_radial_symmetry(psf):
    """PSF should be circularly symmetric."""
    omega_s = jnp.array([0.0, 0.0])
    lam = jnp.array(1.0)

    val_x = psf(jnp.array([2.0, 0.0]), omega_s, lam)
    val_y = psf(jnp.array([0.0, 2.0]), omega_s, lam)
    val_diag = psf(jnp.array([jnp.sqrt(2.0), jnp.sqrt(2.0)]), omega_s, lam)

    assert jnp.allclose(val_x, val_y, rtol=1e-5)
    assert jnp.allclose(val_x, val_diag, rtol=1e-5)


def test_psf_gradient(psf):
    """Gradient w.r.t. parameters should be computable."""
    omega_p = jnp.array([1.0, 0.0])
    omega_s = jnp.array([0.0, 0.0])
    lam = jnp.array(1.0)

    def loss(fwhm_ref):
        p = GaussianPSF(
            fwhm_ref=fwhm_ref,
            wavelength_ref=1.0,

        )
        return p(omega_p, omega_s, lam)

    grad = jax.grad(loss)(2.0)
    assert jnp.isfinite(grad)
    assert grad != 0.0
