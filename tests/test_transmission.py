"""Tests for SPHEREx filter transmission models."""

import jax.numpy as jnp
import pytest

from spherex.transmission import GaussianFilterTransmission


@pytest.fixture
def filt():
    """Default filter: lambda_c = 1.0 + 0.01 * y_p, width = 0.05 um."""
    return GaussianFilterTransmission(
        lambda_intercept=1.0,
        lambda_slope=0.01,
        width=0.05,
    )


def test_central_wavelength_scalar(filt):
    """Scalar omega_p should give scalar lambda_c."""
    omega_p = jnp.array([0.0, 10.0])
    lam_c = filt.central_wavelength(omega_p)
    assert lam_c.shape == ()
    assert jnp.allclose(lam_c, 1.0 + 0.01 * 10.0)


def test_central_wavelength_batched(filt):
    """Batched omega_p should give batched lambda_c."""
    omega_p = jnp.array([[0.0, 0.0], [10.0, 20.0], [5.0, 5.0]])
    lam_c = filt.central_wavelength(omega_p)
    assert lam_c.shape == (3,)
    expected = 1.0 + 0.01 * omega_p[:, 1]
    assert jnp.allclose(lam_c, expected)


def test_central_wavelength_x_independent(filt):
    """Central wavelength should not depend on x."""
    omega_p1 = jnp.array([0.0, 5.0])
    omega_p2 = jnp.array([100.0, 5.0])
    assert jnp.allclose(
        filt.central_wavelength(omega_p1),
        filt.central_wavelength(omega_p2),
    )


def test_transmission_peak(filt):
    """At lambda = lambda_c, transmission should be 1."""
    omega_p = jnp.array([0.0, 10.0])
    lam_c = filt.central_wavelength(omega_p)
    T = filt(lam_c, omega_p)
    assert jnp.allclose(T, 1.0, rtol=1e-5)


def test_transmission_range(filt):
    """Transmission should always be in [0, 1]."""
    omega_p = jnp.array([0.0, 5.0])
    lam_c = filt.central_wavelength(omega_p)
    lambdas = jnp.linspace(lam_c - 1.0, lam_c + 1.0, 1000)
    T = filt(lambdas, omega_p)
    assert jnp.all(T >= 0.0)
    assert jnp.all(T <= 1.0)


def test_transmission_gaussian_shape(filt):
    """Transmission should follow exp(-0.5 * ((lam-lam_c)/width)^2)."""
    omega_p = jnp.array([0.0, 5.0])
    lam_c = filt.central_wavelength(omega_p)
    lam = jnp.array([lam_c + 2.0 * filt.width])
    T = filt(lam, omega_p)
    expected = jnp.exp(-0.5 * (2.0)**2)   # exp(-2) ≈ 0.1353
    assert jnp.allclose(T, expected, rtol=1e-5)


def test_transmission_batched_wavelength(filt):
    """Batched wavelengths should produce matching output."""
    omega_p = jnp.array([0.0, 0.0])
    lam = jnp.array([0.9, 1.0, 1.1])
    T = filt(lam, omega_p)
    assert T.shape == (3,)


def test_transmission_batched_omega_p(filt):
    """Batched omega_p should produce matching output."""
    omega_p = jnp.array([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]])
    lam = jnp.array(1.0)
    T = filt(lam, omega_p)
    assert T.shape == (3,)


def test_transmission_far_from_center_is_small(filt):
    """Far from lambda_c the transmission should be negligible."""
    omega_p = jnp.array([0.0, 5.0])
    lam_c = filt.central_wavelength(omega_p)
    lam_far = lam_c + 10.0 * filt.width
    T = filt(lam_far, omega_p)
    assert T < 1e-20
