"""Tests for SPHEREx source spectrum models (shape-only templates)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spherex.spectrum import (
    BlackbodySpectrum,
    NeuralNetSpectrum,
    LAMBDA_0,
    LOG_AMPLITUDE_INDEX,
    split_source_params,
    join_source_params,
)
from spherex.constants import TEMPERATURE_UNIT, WAVELENGTH_UNIT
import astropy.units as u


@pytest.fixture
def bb():
    return BlackbodySpectrum()


def test_blackbody_output_shape(bb):
    """Single set of shape params should produce matching output shape."""
    lam = jnp.array([0.5, 1.0, 2.0])
    params = jnp.log(jnp.array([5.0]))          # (1,) log-temperature
    result = bb(lam, params)
    assert result.shape == (3,)


def test_blackbody_batched_params(bb):
    """Multiple param sets should broadcast over wavelength."""
    lam = jnp.array([0.5, 1.0, 2.0])            # (3,)
    params = jnp.log(jnp.array([[5.0],          # (2, 1)
                                [3.0]]))
    result = bb(lam, params)
    assert result.shape == (2, 3)


def test_blackbody_batched_wavelength_and_params(bb):
    """Batched wavelength and params should broadcast."""
    lam = jnp.array([[0.5, 1.0], [1.5, 2.0]])   # (2, 2)
    params = jnp.log(jnp.array([[5.0],
                                [3.0]]))        # (2, 1)
    result = bb(lam, params)
    assert result.shape == (2, 2)


def test_blackbody_positivity(bb):
    """The shape should always be positive."""
    lam = jnp.logspace(-1, 2, 200)               # 0.1 – 100 um
    T_eff = (5778 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value  # Solar T
    params = jnp.log(jnp.array([T_eff]))
    result = bb(lam, params)
    assert jnp.all(result > 0)


def test_blackbody_normalised_at_lambda_0(bb):
    """The shape must be exactly 1 at the reference wavelength."""
    lam = jnp.array([LAMBDA_0])
    for T in (3.0, 5.0, 8.0):
        params = jnp.log(jnp.array([T]))
        assert jnp.allclose(bb(lam, params), 1.0, rtol=1e-5)


def test_blackbody_shape_is_pure_ratio(bb):
    """The shape is a pure ratio, independent of any amplitude."""
    lam = jnp.array([0.5, LAMBDA_0, 2.0])
    params = jnp.log(jnp.array([5.0]))
    shape = bb(lam, params)
    assert jnp.allclose(shape[1], 1.0, rtol=1e-5)
    # A 5 kK star is bluer than LAMBDA_0 on the blue side and redder on the
    # red side, so the ratio is >1 below and <1 above the reference.
    assert shape[0] > 1.0
    assert shape[2] < 1.0


def test_blackbody_peak_near_visible_for_solar_temperature(bb):
    """Solar-temperature blackbody peaks near 0.5 um (Wien's law: ~0.502 um)."""
    lam = (jnp.linspace(0.1, 3.0, 1000) * u.um).to(WAVELENGTH_UNIT).value
    T_eff = (5778 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value
    params = jnp.log(jnp.array([T_eff]))  # log-space
    flux = bb(lam, params)
    idx_max = jnp.argmax(flux)
    lam_peak = lam[idx_max]
    # Wien's displacement: lambda_max * T ≈ 2.898 um*kK -> ~0.501 um
    limits = ([0.4, 0.7] * u.um).to(WAVELENGTH_UNIT).value
    assert limits[0] < lam_peak < limits[1]


def test_blackbody_hotter_is_brighter_at_short_wavelengths(bb):
    """At short wavelengths a hotter blackbody should be brighter."""
    lam = jnp.array([0.3])
    p_cool = jnp.log(jnp.array([4.0]))
    p_hot = jnp.log(jnp.array([8.0]))
    assert bb(lam, p_hot) > bb(lam, p_cool)


def test_blackbody_gradient(bb):
    """Gradient w.r.t. log-temperature should be computable and finite.

    Note the wavelength must differ from LAMBDA_0 here: the shape is pinned
    to exactly 1 at LAMBDA_0 for every temperature, so d/dT vanishes there.
    """
    lam = jnp.array([2.0])
    params = jnp.log(jnp.array([5.0]))

    grad_T = jax.grad(lambda p: jnp.sum(bb(lam, p)))(params)
    assert jnp.isfinite(grad_T[0])
    assert grad_T[0] != 0.0


def test_blackbody_gradient_finite_across_bands(bb):
    """Gradient w.r.t. log-temperature is finite across all bands."""
    lam = jnp.array([0.75, 1.6, 2.4, 3.8, 4.4, 5.0])
    params = jnp.log(jnp.array([5.0]))
    grad = jax.grad(lambda p: jnp.sum(bb(lam, p)))(params)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0)


def test_blackbody_n_params(bb):
    """The spectrum model has exactly one shape parameter (log-temperature)."""
    assert bb.n_params == 1


def test_lambda_0_is_shared(bb):
    """Every instance defaults to the same global reference wavelength."""
    assert BlackbodySpectrum().lambda_0 == LAMBDA_0
    assert bb.lambda_0 == LAMBDA_0


def test_split_join_source_params():
    """The [log_amplitude, shape...] layout round-trips."""
    assert LOG_AMPLITUDE_INDEX == 0
    source_params = jnp.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
    log_amp, shape = split_source_params(source_params)
    assert log_amp.shape == (2, 1)
    assert shape.shape == (2, 2)
    assert jnp.allclose(log_amp[:, 0], jnp.array([1.0, 4.0]))
    assert jnp.allclose(shape, jnp.array([[2.0, 3.0], [5.0, 6.0]]))
    assert jnp.allclose(join_source_params(log_amp, shape), source_params)


# ---------------------------------------------------------------------------
# NeuralNetSpectrum
# ---------------------------------------------------------------------------
#
# The model must satisfy the same interface as BlackbodySpectrum: one source's
# shape params ``(n_params,)`` and many wavelengths ``(N_lambda,)`` in, a
# ``(N_lambda,)`` shape out, normalised to 1 at LAMBDA_0.  Batching over
# sources is the caller's job (the image generators already map over sources).


@pytest.fixture
def nn():
    return NeuralNetSpectrum(
        n_params=2, n_hidden_layers=2, hidden_size=8,
        key=jax.random.PRNGKey(0),
    )


def test_neural_net_output_shape(nn):
    """One source's params + many wavelengths -> (N_lambda,)."""
    lam = jnp.array([0.5, 1.0, 2.0])
    params = jnp.zeros(2)
    result = nn(lam, params)
    assert result.shape == (3,)


def test_neural_net_single_wavelength(nn):
    """A scalar wavelength is promoted to a length-1 vector."""
    result = nn(jnp.array(1.5), jnp.zeros(2))
    assert result.shape == (1,)


def test_neural_net_normalised_at_lambda_0(nn):
    """The shape must be exactly 1 at the reference wavelength."""
    for params in (jnp.zeros(2), jnp.array([1.0, -2.0])):
        assert jnp.allclose(
            nn(jnp.array([LAMBDA_0]), params), 1.0, rtol=1e-5
        )


def test_neural_net_positivity(nn):
    """The shape is an exponential, so it must be strictly positive."""
    lam = jnp.linspace(0.4, 5.0, 50)
    assert jnp.all(nn(lam, jnp.array([0.3, -0.7])) > 0)


def test_neural_net_gradient(nn):
    """Gradients w.r.t. shape params should be finite and non-zero."""
    lam = jnp.array([0.5, 1.0, 2.0])
    params = jnp.array([0.1, -0.2])

    grad = jax.grad(lambda p: jnp.sum(nn(lam, p)))(params)
    assert grad.shape == (2,)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(grad != 0.0)


def test_neural_net_consistent_under_vmap(nn):
    """Vmapping over sources must match looping over sources."""
    lam = jnp.array([0.5, 1.0, 2.0])
    thetas = jnp.array([[0.0, 0.0], [0.5, -0.5], [1.0, 0.25]])

    batched = jax.vmap(nn, in_axes=(None, 0))(lam, thetas)
    assert batched.shape == (3, 3)
    for i in range(thetas.shape[0]):
        assert jnp.allclose(batched[i], nn(lam, thetas[i]))


def test_neural_net_n_params(nn):
    """``n_params`` counts shape parameters (amplitude lives outside)."""
    assert nn.n_params == 2

