"""Tests for SPHEREx source spectrum models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spherex.spectrum import BlackbodySpectrum
from spherex.constants import TEMPERATURE_UNIT, WAVELENGTH_UNIT
import astropy.units as u


@pytest.fixture
def bb():
    return BlackbodySpectrum()


def test_blackbody_output_shape(bb):
    """Single set of params should produce matching output shape."""
    lam = jnp.array([0.5, 1.0, 2.0])
    params = jnp.log(jnp.array([5.0, 1.0]))
    result = bb(lam, params)
    assert result.shape == (3,)


def test_blackbody_batched_params(bb):
    """Multiple param sets should broadcast over wavelength."""
    lam = jnp.array([0.5, 1.0, 2.0])            # (3,)
    params = jnp.log(jnp.array([[5.0, 1.0],           # (2, 2)
                                [3.0, 2.0]]))
    result = bb(lam, params)
    assert result.shape == (2, 3)


def test_blackbody_batched_wavelength_and_params(bb):
    """Batched wavelength and params should broadcast."""
    lam = jnp.array([[0.5, 1.0], [1.5, 2.0]])   # (2, 2)
    params = jnp.log(jnp.array([[5.0, 1.0],
                                [3.0, 2.0]]))           # (2, 2)
    result = bb(lam, params)
    assert result.shape == (2, 2)


def test_blackbody_positivity(bb):
    """Flux should always be positive."""
    lam = jnp.logspace(-1, 2, 200)               # 0.1 – 100 um
    T_eff = (5778 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value  # Solar T
    params = jnp.log(jnp.array([T_eff, 1.0]))
    result = bb(lam, params)
    assert jnp.all(result > 0)


def test_blackbody_amplitude_scaling(bb):
    """Doubling amplitude should double flux."""
    lam = jnp.array([1.0])
    p1 = jnp.log(jnp.array([5.0, 1.0]))
    p2 = jnp.log(jnp.array([5.0, 2.0]))
    assert jnp.allclose(bb(lam, p2), 2.0 * bb(lam, p1))


def test_blackbody_peak_near_visible_for_solar_temperature(bb):
    """Solar-temperature blackbody peaks near 0.5 um (Wien's law: ~0.502 um)."""
    lam = (jnp.linspace(0.1, 3.0, 1000) * u.um).to(WAVELENGTH_UNIT).value
    T_eff = (5778 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value
    params = jnp.log(jnp.array([T_eff, 1.0]))  # log-space
    flux = bb(lam, params)
    idx_max = jnp.argmax(flux)
    lam_peak = lam[idx_max]
    # Wien's displacement: lambda_max * T ≈ 2.898 um*kK -> ~0.501 um
    limits = ([0.4, 0.7] * u.um).to(WAVELENGTH_UNIT).value
    assert limits[0] < lam_peak < limits[1]


def test_blackbody_hotter_is_brighter_at_short_wavelengths(bb):
    """At short wavelengths a hotter blackbody should be brighter."""
    lam = jnp.array([0.3])
    p_cool = jnp.log(jnp.array([4.0, 1.0]))
    p_hot = jnp.log(jnp.array([8.0, 1.0]))
    assert bb(lam, p_hot) > bb(lam, p_cool)


def test_blackbody_gradient(bb):
    """Gradient w.r.t. temperature should be computable and finite."""
    lam = jnp.array([1.0])
    params = jnp.log(jnp.array([5.0, 1.0]))

    grad_T = jax.grad(lambda p: jnp.sum(bb(lam, p)))(params)
    assert jnp.isfinite(grad_T[0])
    assert grad_T[0] != 0.0


def test_blackbody_gradient_amplitude(bb):
    """Gradient w.r.t. amplitude should be the Planck function itself."""
    lam = jnp.array([1.0])
    params = jnp.log(jnp.array([5.0, 2.0]))

    grad_amp = jax.grad(lambda p: jnp.sum(bb(lam, p)))(params)
    # d/d(logA) [exp(logA) * B] = exp(logA) * B = A * B
    planck_val = bb(lam, jnp.log(jnp.array([5.0, 1.0])))
    assert jnp.allclose(grad_amp[1], planck_val)
