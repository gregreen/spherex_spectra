"""Tests for SPHEREx source spectrum models and the caller-side normalisation.

The models return an *unnormalised log-flux* (see ``spherex.spectrum``); it is
the CALLER's job to anchor it at ``LAMBDA_0``, via ``reference_log_flux`` /
``normalized_shape`` / ``normalized_source_params``.  Both sides are tested
here.
"""

import equinox as eqx
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
    reference_log_flux,
    normalized_log_shape,
    normalized_shape,
    normalized_source_params,
)
from spherex.constants import (
    HC_JAX, KB_JAX, TEMPERATURE_UNIT, WAVELENGTH_UNIT,
)
import astropy.units as u


@pytest.fixture
def bb():
    return BlackbodySpectrum()


class _OffsetSpectrum(eqx.Module):
    """A spectrum model plus a wavelength-independent offset.

    Models are only defined up to such an offset, so anything that consumes
    them must be blind to it.
    """

    base: eqx.Module
    offset: float

    def __call__(self, wavelength, shape_params):
        return self.base(wavelength, shape_params) + self.offset


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


def test_blackbody_log_kernel_is_finite(bb):
    """The raw log-flux kernel must be finite over a huge wavelength range."""
    lam = jnp.logspace(-1, 2, 200)               # 0.1 – 100 um
    for T in (3.0, 5.0, 8.0):
        params = jnp.log(jnp.array([T]))         # log(kK)
        assert jnp.all(jnp.isfinite(bb(lam, params)))


def test_blackbody_normalized_shape_is_positive(bb):
    """The normalised shape is an exponential, so it is strictly positive."""
    lam = jnp.logspace(-1, 2, 200)               # 0.1 – 100 um
    T_eff = (5778 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value  # Solar T
    params = jnp.log(jnp.array([T_eff]))
    shape = normalized_shape(bb, lam, params)
    assert jnp.all(jnp.isfinite(shape))
    assert jnp.all(shape > 0)


def test_blackbody_normalised_at_lambda_0(bb):
    """The normalised shape must be exactly 1 at the reference wavelength."""
    lam = jnp.array([LAMBDA_0])
    for T in (3.0, 5.0, 8.0):
        params = jnp.log(jnp.array([T]))
        assert jnp.allclose(
            normalized_shape(bb, lam, params), 1.0, rtol=1e-5
        )


def test_reference_log_flux_is_the_model_at_lambda_0(bb):
    """``reference_log_flux`` is precisely the model evaluated at LAMBDA_0."""
    params = jnp.log(jnp.array([5.0]))
    assert jnp.allclose(
        reference_log_flux(bb, params), bb(jnp.array([LAMBDA_0]), params)[0]
    )
    # ... and it defaults to the global LAMBDA_0.
    assert jnp.allclose(
        reference_log_flux(bb, params, LAMBDA_0),
        reference_log_flux(bb, params),
    )


def test_blackbody_normalized_shape_is_the_planck_ratio(bb):
    """normalized_shape reproduces the analytic Planck ratio exactly.

    ``(lambda_0/lambda)^5 * (exp(x_0) - 1) / (exp(x) - 1)`` - this is the
    formula the model used to return internally, now assembled by the caller.
    """
    lam = jnp.array([0.5, 1.0, 2.0, 4.0])
    T = 5.0
    params = jnp.log(jnp.array([T]))

    x = (HC_JAX / (lam * KB_JAX)) / T
    x_0 = (HC_JAX / (LAMBDA_0 * KB_JAX)) / T
    expected = (
        (LAMBDA_0 / lam) ** 5 * jnp.expm1(x_0) / jnp.expm1(x)
    )
    assert jnp.allclose(normalized_shape(bb, lam, params), expected, rtol=1e-5)


def test_blackbody_shape_is_pure_ratio(bb):
    """The shape is a pure ratio, independent of any amplitude."""
    lam = jnp.array([0.5, LAMBDA_0, 2.0])
    params = jnp.log(jnp.array([5.0]))
    shape = normalized_shape(bb, lam, params)
    assert jnp.allclose(shape[1], 1.0, rtol=1e-5)
    # A 5 kK star is bluer than LAMBDA_0 on the blue side and redder on the
    # red side, so the ratio is >1 below and <1 above the reference.
    assert shape[0] > 1.0
    assert shape[2] < 1.0


def test_blackbody_peak_near_visible_for_solar_temperature(bb):
    """Solar-temperature blackbody peaks near 0.5 um (Wien's law: ~0.502 um).

    The model returns a log-flux, but ``log`` is monotonic, so its argmax is
    still the Planck peak.
    """
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
    assert (normalized_shape(bb, lam, p_hot)
            > normalized_shape(bb, lam, p_cool))


def test_blackbody_gradient(bb):
    """Gradient w.r.t. log-temperature should be computable and finite."""
    lam = jnp.array([2.0])
    params = jnp.log(jnp.array([5.0]))

    grad_T = jax.grad(lambda p: jnp.sum(bb(lam, p)))(params)
    assert jnp.isfinite(grad_T[0])
    assert grad_T[0] != 0.0


def test_normalized_shape_has_zero_temperature_gradient_at_lambda_0(bb):
    """The normalised shape is PINNED to 1 at LAMBDA_0 for every temperature.

    Pinning is now enforced by the caller, so this is the property that would
    break first if ``normalized_shape`` stopped anchoring at LAMBDA_0.
    """
    lam = jnp.array([LAMBDA_0])
    params = jnp.log(jnp.array([5.0]))
    grad = jax.grad(
        lambda p: jnp.sum(normalized_shape(bb, lam, p))
    )(params)
    assert jnp.allclose(grad, 0.0, atol=1e-6)


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


def test_blackbody_is_anchor_free(bb):
    """Models must not carry their own reference wavelength any more.

    Normalisation at LAMBDA_0 is the caller's job (see the module docstring),
    so a model attribute would silently reintroduce the old convention.
    """
    assert not hasattr(bb, "lambda_0")


def test_normalized_source_params_folds_the_reference(bb):
    """Folding the reference into the amplitude is equivalent to dividing."""
    lam = jnp.array([0.5, 1.0, 2.0, 4.0])
    log_amp = -30.0
    theta = jnp.log(jnp.array([5.0]))
    source_params = join_source_params(jnp.array([log_amp]), theta)

    folded = normalized_source_params(bb, source_params)

    # The folded first column is NOT the physical log-amplitude any more ...
    assert not jnp.allclose(folded[0], log_amp)
    # ... but exp(folded[0]) * exp(model) must equal the physical flux.
    lhs = jnp.exp(folded[0] + bb(lam, folded[1:]))
    rhs = jnp.exp(log_amp) * normalized_shape(bb, lam, theta)
    assert jnp.allclose(lhs, rhs, rtol=1e-6)


def test_offset_invariance(bb):
    """Any wavelength-independent offset in a model must cancel out.

    This is the property the whole caller-side normalisation exists for: a
    model is only defined up to such an offset, which is exactly what a
    randomly initialised (and unconstrained) neural network supplies.

    Note the offsets below are large on purpose: in float32 the subtraction
    ``(a + c) - (b + c)`` loses precision proportional to ``|c| * eps``, so
    agreement is limited to ~1e-5 rather than to the last bit.
    """
    lam = jnp.array([0.5, 1.0, 2.0, 4.0])
    theta = jnp.log(jnp.array([5.0]))

    reference = normalized_shape(bb, lam, theta)
    for offset in (-37.5, 0.0, 12.25):
        shifted = normalized_shape(_OffsetSpectrum(bb, offset), lam, theta)
        np.testing.assert_allclose(shifted, reference, rtol=1e-5)

    # Same for the folded amplitude route used by the generators: the
    # physical flux must be unchanged by the offset.
    lam = jnp.array([0.5, 1.0, 2.0, 4.0])
    log_amp = -30.0
    source_params = join_source_params(jnp.array([log_amp]), theta)

    offset_model = _OffsetSpectrum(bb, 5.5)
    folded = normalized_source_params(offset_model, source_params)
    lhs = jnp.exp(folded[0] + offset_model(lam, folded[1:]))
    plain = normalized_source_params(bb, source_params)
    rhs = jnp.exp(plain[0] + bb(lam, plain[1:]))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-5)


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


def test_neural_net_log_kernel_is_finite(nn):
    """The log-flux kernel must be finite, with no exp/overflow in the model."""
    lam = jnp.linspace(0.4, 5.0, 50)
    for params in (jnp.zeros(2), jnp.array([5.0, -5.0])):
        assert jnp.all(jnp.isfinite(nn(lam, params)))


def test_neural_net_normalised_at_lambda_0(nn):
    """The caller-normalised shape must be exactly 1 at LAMBDA_0."""
    lam = jnp.array([LAMBDA_0])
    for params in (jnp.zeros(2), jnp.array([1.0, -2.0])):
        assert jnp.allclose(
            normalized_shape(nn, lam, params), 1.0, rtol=1e-5
        )


def test_neural_net_normalized_log_shape_is_zero_at_lambda_0(nn):
    """``normalized_log_shape`` is anchored to 0 at LAMBDA_0."""
    params = jnp.array([0.4, -0.3])
    assert jnp.allclose(
        normalized_log_shape(nn, jnp.array([LAMBDA_0]), params), 0.0,
        atol=1e-6,
    )


def test_neural_net_positivity(nn):
    """The normalised shape is an exponential, so it is strictly positive."""
    lam = jnp.linspace(0.4, 5.0, 50)
    shape = normalized_shape(nn, lam, jnp.array([0.3, -0.7]))
    assert jnp.all(shape > 0)
    assert jnp.all(jnp.isfinite(shape))


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


def test_neural_net_normalized_source_params_round_trip(nn):
    """The folded-amplitude route matches the explicit division, for the NN."""
    lam = jnp.linspace(0.4, 5.0, 17)
    log_amp = -31.0
    theta = jnp.array([0.6, -0.4])
    source_params = join_source_params(jnp.array([log_amp]), theta)

    folded = normalized_source_params(nn, source_params)
    lhs = jnp.exp(folded[0] + nn(lam, folded[1:]))
    rhs = jnp.exp(log_amp) * normalized_shape(nn, lam, theta)
    assert jnp.allclose(lhs, rhs, rtol=1e-6)


def test_neural_net_offset_invariance(nn):
    """A constant offset on the network's log-flux must cancel out.

    This is what makes a *random* network safe: its raw offset is neither
    meaningful nor constrained by the data, and the caller removes it by
    subtracting the value at LAMBDA_0.  The tolerance reflects float32
    cancellation in ``(a + c) - (b + c)``.
    """
    lam = jnp.linspace(0.4, 5.0, 17)
    theta = jnp.array([0.2, 0.8])

    reference = normalized_shape(nn, lam, theta)
    shifted = normalized_shape(_OffsetSpectrum(nn, 41.0), lam, theta)
    np.testing.assert_allclose(shifted, reference, rtol=1e-5)

