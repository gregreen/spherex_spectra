"""Tests for the mock simulation's pluggable spectrum model.

Covers the three things the neural-network mock relies on:

* the model factory (determinism, output rescaling, normalisation),
* generic ``(N, 1 + P)`` parameter bookkeeping for any ``P``,
* the injection of a single shared model into every band's generator, and the
  guarantee that its weights are FROZEN during inference.
"""

import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
))

from spherex import (  # noqa: E402
    BlackbodySpectrum,
    NeuralNetSpectrum,
    SpherexImageGenerator,
    SpherexImageGenerator3,
    LAMBDA_0,
    normalized_shape,
    normalized_log_shape,
)
from spherex.config import _build_band  # noqa: E402

import mock_spherex_images as mock  # noqa: E402
from inference import EXPOSURE_TIME, infer_parameters  # noqa: E402


def _small_nn(n_params=1, n_hidden_layers=1, hidden_size=8, seed=0):
    """A tiny neural-network spectrum model (fast to build and evaluate)."""
    return mock._build_spectrum_model(
        kind="nn", n_params=n_params, n_hidden_layers=n_hidden_layers,
        hidden_size=hidden_size, seed=seed, verbose=False,
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def test_build_spectrum_model_kinds():
    """The factory honours the requested kind, and defaults to SPECTRUM_KIND."""
    model = mock._build_spectrum_model(kind="blackbody", verbose=False)
    assert isinstance(model, BlackbodySpectrum)
    assert model.n_params == 1

    default = mock._build_spectrum_model(verbose=False)
    expected = (BlackbodySpectrum if mock.SPECTRUM_KIND == "blackbody"
                else NeuralNetSpectrum)
    assert isinstance(default, expected)


def test_build_spectrum_model_rejects_unknown_kind():
    with pytest.raises(ValueError):
        mock._build_spectrum_model(kind="magic")


def test_nn_factory_is_seed_deterministic():
    """Same seed -> identical weights; different seed -> different weights."""
    a = mock._model_fingerprint(_small_nn(seed=7))
    b = mock._model_fingerprint(_small_nn(seed=7))
    c = mock._model_fingerprint(_small_nn(seed=8))

    assert a == b
    assert a != c


def test_nn_rescale_hits_target_log_shape_std():
    """The output rescaling must bring the in-band log-shape to O(1).

    The measured spread uses a DIFFERENT key from the one used for the
    rescaling, so this is a genuine out-of-sample check of the target (a
    finite-sample std over the rescaling draw would be exact but circular).
    """
    model = _small_nn(seed=11)
    lambdas = mock._wavelength_grid(256)
    thetas = jax.random.normal(jax.random.PRNGKey(1234), (256, model.n_params))

    log_shape = jax.vmap(
        lambda th: normalized_log_shape(model, lambdas, th)
    )(thetas)
    measured = float(jnp.std(log_shape))

    assert np.isfinite(measured)
    assert measured == pytest.approx(mock.NN_TARGET_LOG_SHAPE_STD, rel=0.5)


def test_nn_shape_is_normalised_and_positive():
    """Caller-normalised NN shapes are finite, positive and 1 at LAMBDA_0."""
    model = _small_nn(seed=5)
    thetas = jax.random.normal(jax.random.PRNGKey(0), (8, model.n_params))
    lambdas = mock._wavelength_grid(64)

    log_shapes = jax.vmap(
        lambda th: normalized_log_shape(model, lambdas, th)
    )(thetas)
    assert np.all(np.isfinite(log_shapes))

    shapes = np.asarray(jnp.exp(log_shapes))
    assert np.all(shapes > 0.0)
    # Normalised to exactly 1 at the global reference wavelength.
    np.testing.assert_allclose(
        np.asarray(normalized_shape(model, jnp.array([LAMBDA_0]), thetas[0])),
        1.0, rtol=1e-5,
    )


def test_nn_supports_multiple_shape_parameters():
    """P > 1 must build, evaluate and be differentiable."""
    model = _small_nn(n_params=3)
    assert model.n_params == 3

    lambdas = mock._wavelength_grid(32)
    theta = jnp.zeros(3)
    shape = model(lambdas, theta)
    assert shape.shape == (32,)

    grad = jax.grad(lambda th: jnp.sum(model(lambdas, th)))(theta)
    assert grad.shape == (3,)
    assert np.all(np.isfinite(np.asarray(grad)))


# ---------------------------------------------------------------------------
# Generic (N, 1 + P) parameter bookkeeping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_params", [1, 3])
def test_shape_param_helpers_are_p_generic(n_params):
    model = _small_nn(n_params=n_params)
    rng = np.random.default_rng(0)
    n = 17

    prior = mock._draw_shape_params(rng, n, model)
    init = mock._draw_init_shape_params(rng, n, model)

    assert prior.shape == (n, n_params)
    assert init.shape == (n, n_params)

    log_amplitudes = rng.uniform(-30.0, -20.0, n)
    params = mock._make_params(log_amplitudes, prior)

    assert params.shape == (n, 1 + n_params)
    # Amplitude FIRST (see spherex.spectrum), shape columns after.
    np.testing.assert_allclose(np.asarray(params[:, 0]), log_amplitudes,
                               rtol=1e-6)

    # Perturbation keeps the (N, 1 + P) structure and only touches its block.
    pert_shape, pert_A = mock._perturb_params(prior, log_amplitudes, rng)
    assert pert_shape.shape == (n, n_params)
    assert pert_A.shape == (n,)
    assert not np.allclose(pert_shape, prior)


def test_blackbody_prior_and_init_differ():
    """The initial guess must not simply re-draw the prior for either model."""
    model = BlackbodySpectrum()
    rng = np.random.default_rng(0)

    prior = mock._draw_shape_params(rng, 4096, model)
    init = mock._draw_init_shape_params(rng, 4096, model)

    # Prior: log of a uniform draw (so not uniform in log T); init: uniform in
    # log T.  Their means differ substantially.
    assert abs(np.mean(init) - np.mean(prior)) > 0.05


def test_shape_param_labels_match_p():
    assert mock._shape_param_labels(BlackbodySpectrum()) == ["log(T / kK)"]
    assert mock._shape_param_labels(_small_nn(n_params=3)) == [
        "theta_0", "theta_1", "theta_2"
    ]


def test_filter_sources_builds_generic_params():
    """``_filter_sources`` must stack [log_amplitude, theta] for any P."""
    model = _small_nn(n_params=3)
    rng = np.random.default_rng(1)

    skycoords, shape_params, log_amplitudes = mock._generate_catalog(
        rng, radius_deg=0.01, model=model
    )
    assert shape_params.shape == (mock.N_SOURCES, 3)

    # A single synthetic exposure with a generous half-stamp so that most
    # sources land inside the tiny detector.
    wcs = mock._make_wcs(mock.CATALOG_CENTER.ra.deg,
                         mock.CATALOG_CENTER.dec.deg)
    exposures = [(1, wcs, 8)]

    _, _, _, per_exposure_data = mock._filter_sources(
        skycoords, shape_params, log_amplitudes, exposures
    )

    positions, params, band, _wcs, half_stamp, src_idx = per_exposure_data[0]
    assert params.shape == (positions.shape[0], 4)
    np.testing.assert_allclose(
        np.asarray(params[:, 1:]), shape_params[np.asarray(src_idx)],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(params[:, 0]), log_amplitudes[np.asarray(src_idx)],
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Library injection point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [SpherexImageGenerator,
                                 SpherexImageGenerator3])
def test_generators_accept_injected_spectrum_model(cls):
    """Passing a model must reach the generator, and change the image."""
    model = _small_nn(seed=3)

    gen_bb = cls(band=2, image_width=32, image_height=32)
    gen_nn = cls(band=2, image_width=32, image_height=32,
                 spectrum_model=model)

    assert isinstance(gen_bb.spectrum_model, BlackbodySpectrum)
    assert gen_nn.spectrum_model is model

    params = jnp.asarray([[np.log(1e-17), 0.0]], dtype=jnp.float32)
    kwargs = dict(postage_stamp_half_size=6, n_wavelength_samples=3,
                  oversampling=2)
    img_bb = np.asarray(gen_bb(jnp.asarray([[16.0, 16.0]]), params, **kwargs))
    img_nn = np.asarray(gen_nn(jnp.asarray([[16.0, 16.0]]), params, **kwargs))

    assert img_nn.shape == img_bb.shape
    assert np.all(np.isfinite(img_nn))
    assert not np.allclose(img_nn, img_bb)


def test_build_band_default_builds_a_blackbody():
    """The library default path still builds an (anchor-free) blackbody."""
    _psf, _tr, spectrum_model, _ap = _build_band(3)
    assert isinstance(spectrum_model, BlackbodySpectrum)
    assert spectrum_model.n_params == 1


class _OffsetSpectrum(eqx.Module):
    """A spectrum model plus a wavelength-independent offset.

    Models are only defined up to such an offset - a randomly initialised
    network supplies an arbitrary one - so the generators must be blind to it.
    """

    base: eqx.Module
    offset: float

    def __call__(self, wavelength, shape_params):
        return self.base(wavelength, shape_params) + self.offset


@pytest.mark.parametrize("cls", [SpherexImageGenerator,
                                 SpherexImageGenerator3])
def test_images_are_invariant_to_a_model_offset(cls):
    """Two models differing only by an offset must give the same image.

    This is the end-to-end guarantee of moving the LAMBDA_0 normalisation out
    of the models and into the generators.  (Agreement is limited to a few
    parts in 10^6 by float32 cancellation in ``(a + c) - (b + c)``, so this is
    not bit-exact.)
    """
    positions = jnp.asarray([[16.0, 16.0], [25.0, 13.0]], dtype=jnp.float32)
    params = jnp.asarray([[np.log(1e-17), np.log(5.0)],
                          [np.log(3e-18), np.log(4.0)]], dtype=jnp.float32)
    kwargs = dict(postage_stamp_half_size=6, n_wavelength_samples=3,
                  oversampling=2)

    bb = BlackbodySpectrum()
    gen_plain = cls(band=3, image_width=48, image_height=48,
                    spectrum_model=bb)
    gen_shifted = cls(band=3, image_width=48, image_height=48,
                      spectrum_model=_OffsetSpectrum(bb, 9.0))

    img_plain = np.asarray(gen_plain(positions, params, **kwargs))
    img_shifted = np.asarray(gen_shifted(positions, params, **kwargs))

    # Sanity: the sources actually put flux on the detector (pixels outside the
    # postage stamps are legitimately exactly zero, hence the tiny atol).
    assert img_plain.sum() > 0.0
    assert np.all(np.isfinite(img_shifted))
    np.testing.assert_allclose(img_shifted, img_plain, rtol=1e-5, atol=1e-20)


# ---------------------------------------------------------------------------
# Frozen weights
# ---------------------------------------------------------------------------

def test_inference_leaves_model_frozen():
    """Inference must not touch the spectrum model's weights.

    Freezing is automatic (only ``log_params`` and ``log_backgrounds`` are
    differentiated, and the generator is a closed-over constant); this test
    pins that property down so a future refactor cannot silently unfreeze it.
    """
    n_sources = 4
    half_stamp = 8
    model = _small_nn(seed=2)
    mock._set_spectrum_model(model)

    gen = SpherexImageGenerator3(
        band=3, image_width=32, image_height=32, pixel_scale=6.2,
        spectrum_model=model,
    )

    rng = np.random.default_rng(0)
    positions = jnp.asarray(
        rng.uniform(8.0, 24.0, size=(n_sources, 2)), dtype=jnp.float32
    )
    true_log_params = jnp.asarray(
        np.column_stack([
            np.log(rng.uniform(2e-21, 1e-16, size=n_sources)),
            rng.normal(0.0, 1.0, size=n_sources),
        ]), dtype=jnp.float32,
    )

    rate = gen(
        positions, true_log_params,
        postage_stamp_half_size=half_stamp,
        n_wavelength_samples=2, oversampling=2,
    )
    counts = (np.asarray(rate) + 1e-3) * EXPOSURE_TIME
    sigma = np.sqrt(np.maximum(counts, 0.0) + 1.0)

    exposures = [(
        jnp.asarray(counts, dtype=jnp.float32),
        jnp.asarray(sigma, dtype=jnp.float32),
        positions,
        gen,
        half_stamp,
        jnp.arange(n_sources, dtype=jnp.int32),
    )]
    log_backgrounds = jnp.array([np.log(1e-3)], dtype=jnp.float32)

    before = mock._model_fingerprint(model)

    rec_params, rec_backgrounds, _losses, _lrs = infer_parameters(
        exposures, true_log_params, n_steps=4, learning_rate=1e-2,
        warmup_steps=1, momentum=0.0, n_lambda=2, oversampling=2,
        amp_solve_every=2,
    )

    # The model is untouched...
    assert mock._model_fingerprint(model) == before
    # ...and only the source parameters (and backgrounds) came back.
    assert np.asarray(rec_params).shape == (n_sources, 1 + model.n_params)
    assert np.asarray(rec_backgrounds).shape == (1,)


def test_model_fingerprint_detects_change():
    """The fingerprint used to prove freezing must be sensitive."""
    model = _small_nn(seed=4)
    before = mock._model_fingerprint(model)

    # Nudging a single weight must change the fingerprint.  This also pins down
    # that ``eqx.tree_at`` can address the output layer, which is what
    # ``_rescale_nn_output`` relies on.
    last = len(model.layers) - 1
    nudged = eqx.tree_at(
        lambda m: m.layers[last].weight,
        model,
        model.layers[last].weight * 1.0001,
    )

    assert mock._model_fingerprint(nudged) != before
    assert mock._model_fingerprint(model) == before


def test_neural_net_spectrum_is_a_valid_library_model():
    """The injected instance is a plain NeuralNetSpectrum (library contract)."""
    model = _small_nn(n_params=2)
    assert isinstance(model, NeuralNetSpectrum)
    assert model.n_params == 2
    # Shape of a single source: (N_lambda,) - the model must NOT batch itself.
    assert model(mock._wavelength_grid(16), jnp.zeros(2)).shape == (16,)
