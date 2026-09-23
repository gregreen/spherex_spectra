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


def test_nn_rescale_targets_the_per_source_log_shape_std():
    """The output rescaling must bring a TYPICAL source's spectrum to O(1).

    The statistic matched is the PER-SOURCE spread (the std over wavelength of
    one drawn theta, median over draws), not the spread pooled over
    (theta, wavelength): the pooled number also contains the spread between
    sources, so a strongly theta-dependent model could hit it while every
    individual spectrum stayed nearly flat - which is the failure the
    rescaling exists to prevent.

    The measurement uses a DIFFERENT key from the one used for the rescaling,
    so this is a genuine out-of-sample check of the target (a finite-sample std
    over the rescaling draw would be exact but circular).
    """
    model = _small_nn(seed=11)
    lambdas = mock._wavelength_grid(256)
    thetas = jax.random.normal(jax.random.PRNGKey(1234), (256, model.n_params))

    log_shape = jax.vmap(
        lambda th: normalized_log_shape(model, lambdas, th)
    )(thetas)
    per_source = np.asarray(jnp.std(log_shape, axis=1))

    assert np.all(np.isfinite(np.asarray(log_shape)))
    assert np.median(per_source) == pytest.approx(
        mock.NN_TARGET_LOG_SHAPE_STD, rel=0.5)
    # Only the RATIO is controlled by the rescaling, so the two statistics must
    # be of the same order - if the pooled one ran away it would mean the
    # rescaling had been hijacked by the spread BETWEEN sources.
    pooled = float(jnp.std(log_shape))
    assert pooled > 0.0
    assert pooled < 4.0 * mock.NN_TARGET_LOG_SHAPE_STD


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

    # The FiLM branches are part of the same pytree, so they are fingerprinted
    # too - freezing must cover them, not just the main MLP.
    film_nudged = eqx.tree_at(
        lambda m: m.film[0].layers[-1].weight,
        model,
        model.film[0].layers[-1].weight * 1.0001,
    )
    assert mock._model_fingerprint(film_nudged) != before
    assert mock._model_fingerprint(model) == before


def test_layer_norm_is_passed_through_and_fingerprinted():
    """The factory honours ``layer_norm``, and freezing covers its parameters.

    The option is a per-wavelength feature normalisation (see
    ``NeuralNetSpectrum``), so it must not change any other weight for a given
    seed - only add its own, which the frozen-weights guarantee then covers.

    Both variants are requested EXPLICITLY, so this tests the flag rather than
    whatever ``NN_LAYER_NORM`` happens to be set to in the mock's config.
    """
    plain = mock._build_spectrum_model(kind="nn", layer_norm=False,
                                       verbose=False)
    normed = mock._build_spectrum_model(kind="nn", layer_norm=True,
                                       verbose=False)

    assert plain.layer_norm is False
    assert normed.layer_norm is True
    assert len(normed.norms) == len(normed.film)     # one per activated layer

    lambdas = mock._wavelength_grid(16)
    theta = jnp.zeros(normed.n_params)
    assert normed(lambdas, theta).shape == (16,)
    assert not np.allclose(np.asarray(normed(lambdas, theta)),
                           np.asarray(plain(lambdas, theta)))

    # Same key -> same weights, EXCEPT the output layer: turning the
    # normalisation on changes the raw log-shape spread, so
    # ``_rescale_nn_output`` deliberately re-tunes the output weight block to
    # hit the same target spread.  Everything upstream is bit-identical.
    for a, b in zip(plain.layers[:-1], normed.layers[:-1]):
        np.testing.assert_array_equal(np.asarray(a.weight),
                                      np.asarray(b.weight))
    for fa, fb in zip(plain.film, normed.film):
        for la, lb in zip(fa.layers, fb.layers):
            np.testing.assert_array_equal(np.asarray(la.weight),
                                          np.asarray(lb.weight))
    assert not np.allclose(np.asarray(plain.layers[-1].weight),
                           np.asarray(normed.layers[-1].weight))

    # The affine parameters are leaves of the model, so a nudged one must show
    # up in the fingerprint that proves nothing changed during inference.
    before = mock._model_fingerprint(normed)
    nudged = eqx.tree_at(
        lambda m: m.norms[0].weight,
        normed,
        normed.norms[0].weight * 1.0001,
    )
    assert mock._model_fingerprint(nudged) != before
    assert mock._model_fingerprint(normed) == before


def test_neural_net_spectrum_is_a_valid_library_model():
    """The injected instance is a plain NeuralNetSpectrum (library contract)."""
    model = _small_nn(n_params=2)
    assert isinstance(model, NeuralNetSpectrum)
    assert model.n_params == 2
    # Shape of a single source: (N_lambda,) - the model must NOT batch itself.
    assert model(mock._wavelength_grid(16), jnp.zeros(2)).shape == (16,)


# ---------------------------------------------------------------------------
# Per-source spectrum diagnostics
# ---------------------------------------------------------------------------

def _tiny_exposure(n_sources=3, band=3, half_stamp=8, image=32, seed=0,
                   model=None):
    """One synthetic exposure in the format the diagnostics expect."""
    model = mock.SPECTRUM_MODEL if model is None else model
    gen = SpherexImageGenerator3(
        band=band, image_width=image, image_height=image, pixel_scale=6.2,
        spectrum_model=model,
    )

    rng = np.random.default_rng(seed)
    positions = jnp.asarray(
        rng.uniform(8.0, image - 8.0, (n_sources, 2)), dtype=jnp.float32
    )
    log_amplitude = np.log(rng.uniform(3e-18, 3e-17, n_sources))
    if mock._is_blackbody(model):
        shape = np.log(rng.uniform(3.0, 8.0, (n_sources, 1)))
    else:
        shape = rng.normal(0.0, 1.0, (n_sources, model.n_params))
    params = jnp.asarray(np.column_stack([log_amplitude, shape]),
                         dtype=jnp.float32)

    rate = gen(positions, params, postage_stamp_half_size=half_stamp,
               n_wavelength_samples=2, oversampling=2)
    counts = (np.asarray(rate) + 1e-3) * EXPOSURE_TIME
    sigma = np.sqrt(np.maximum(counts, 0.0) + 1.0)

    src_idx = np.arange(n_sources, dtype=np.int32)
    per_exposure_data = [(positions, params, band, None, half_stamp, src_idx)]
    inf_exposures = [(
        jnp.asarray(counts, dtype=jnp.float32),
        jnp.asarray(sigma, dtype=jnp.float32),
        positions, gen, half_stamp, src_idx,
    )]
    return per_exposure_data, inf_exposures


def test_shared_snr_reproduces_the_bright_mask():
    """Sharing the per-exposure S/N sweep must not change the bright mask."""
    model = mock._build_spectrum_model(kind="blackbody", verbose=False)
    mock._set_spectrum_model(model)
    per_exposure_data, inf_exposures = _tiny_exposure(model=model, n_sources=4)

    precomputed = mock._per_exposure_source_snr(per_exposure_data,
                                                inf_exposures)
    with_shared = mock._compute_bright_mask(
        per_exposure_data, inf_exposures, 4, snr_per_exposure=precomputed)
    without = mock._compute_bright_mask(per_exposure_data, inf_exposures, 4)

    assert with_shared.tolist() == without.tolist()
    band, src_idx, snr = precomputed[0]
    assert band == 3
    np.testing.assert_array_equal(src_idx, np.arange(4))
    assert np.asarray(snr).shape == (4,)


def test_observed_central_wavelengths_follow_the_linear_variable_filter():
    """lambda_c = intercept + slope * y, in arcsec, and independent of x."""
    model = mock._build_spectrum_model(kind="blackbody", verbose=False)
    gen = SpherexImageGenerator3(
        band=3, image_width=32, image_height=32, pixel_scale=6.2,
        spectrum_model=model,
    )
    positions = np.array([[10.0, 0.0], [10.0, 16.0], [25.0, 16.0]])

    lam_c = mock._observed_central_wavelengths(gen, positions)

    intercept = gen.transmission.lambda_intercept
    at_y16 = intercept + gen.transmission.lambda_slope * 16.0 * gen.pixel_scale
    assert lam_c[0] == pytest.approx(intercept)
    assert lam_c[1] == pytest.approx(at_y16)
    assert lam_c[2] == pytest.approx(at_y16)


def test_select_plot_sources_takes_the_top_and_a_random_sample():
    """Selection: highest-amplitude bright sources plus random bright ones."""
    n = 30
    rng = np.random.default_rng(0)
    true_params = np.column_stack([
        np.log(rng.uniform(1e-18, 1e-16, n)), np.zeros(n),
    ])
    bright = np.zeros(n, dtype=bool)
    bright[[1, 4, 5, 9, 11, 12, 17, 20, 23, 28, 29]] = True

    indices, groups = mock._select_plot_sources(
        true_params, bright, n_top=3, n_random=4, seed=5)

    assert len(indices) == 7
    assert groups[:3] == ["top"] * 3 and groups[3:] == ["random"] * 4
    assert len(set(indices.tolist())) == 7          # the two draws are disjoint
    assert bright[indices].all()                    # every pick is bright

    bright_idx = np.where(bright)[0]
    by_amplitude = bright_idx[np.argsort(true_params[bright_idx, 0])[::-1]]
    assert indices[:3].tolist() == by_amplitude[:3].tolist()

    again, _ = mock._select_plot_sources(true_params, bright, n_top=3,
                                         n_random=4, seed=5)
    np.testing.assert_array_equal(indices, again)


def test_select_plot_sources_handles_degenerate_cases():
    n = 5
    true_params = np.column_stack([np.log(np.full(n, 1e-17)), np.zeros(n)])

    # Fewer bright sources than requested: take what exists, no random draw.
    indices, groups = mock._select_plot_sources(
        true_params, np.array([True, False, True, False, False]),
        n_top=8, n_random=8)
    assert indices.size == 2 and groups == ["top", "top"]

    # No bright sources at all.
    indices, groups = mock._select_plot_sources(
        true_params, np.zeros(n, dtype=bool))
    assert indices.size == 0 and groups == []


def test_source_spectrum_fname_pads_the_index():
    """``{:02d}`` is a minimum width, so 2- and 3-digit indices both work."""
    assert mock._source_spectrum_fname(7, "d") == os.path.join(
        "d", "source_spectrum_07.svg")
    assert mock._source_spectrum_fname(127, "d") == os.path.join(
        "d", "source_spectrum_127.svg")


def test_source_observation_points_scale_the_error_by_one_over_snr():
    """The bar is the TRUE flux over the matched-filter S/N of the exposure."""
    model = mock._build_spectrum_model(kind="blackbody", verbose=False)
    mock._set_spectrum_model(model)
    per_exposure_data, inf_exposures = _tiny_exposure(model=model, n_sources=2)

    true_params = np.asarray(per_exposure_data[0][1])
    recovered = true_params.copy()
    recovered[:, 0] += 0.2

    snr_per_exposure = mock._per_exposure_source_snr(per_exposure_data,
                                                     inf_exposures)
    obs = mock._source_observation_points(
        0, per_exposure_data, inf_exposures, snr_per_exposure,
        true_params, recovered, model=model)

    assert obs["wavelength"].size == 1
    assert obs["band"].tolist() == [3]
    assert obs["snr"][0] > 0
    np.testing.assert_allclose(obs["f_error"], obs["f_true"] / obs["snr"])
    assert np.all(obs["f_recovered"] > 0)

    # The point is the model at the RECOVERED parameters, at that wavelength.
    lam_c = jnp.asarray(obs["wavelength"], dtype=jnp.float32)
    expected = np.exp(recovered[0, 0]) * np.asarray(normalized_shape(
        model, lam_c, jnp.asarray(recovered[0, 1:], dtype=jnp.float32)))
    np.testing.assert_allclose(obs["f_recovered"], expected, rtol=1e-6)


def test_plot_one_source_spectrum_truncates_an_unconstraining_bar(tmp_path):
    """A bar far taller than the axes must be truncated, not drawn in full.

    An observation whose S/N is far below 1 has a bar of ~``f / S/N``, which
    would otherwise be drawn as a full-height thin line - indistinguishable
    from a band boundary.
    """
    model = mock._build_spectrum_model(kind="blackbody", verbose=False)
    mock._set_spectrum_model(model)
    per_exposure_data, _ = _tiny_exposure(model=model, n_sources=1)

    params = np.asarray(per_exposure_data[0][1])
    f_true = np.exp(params[0, 0]) * np.ones(2)
    obs = {
        "wavelength": np.array([1.5, 2.5]),
        "f_true": f_true,
        "f_recovered": f_true,
        "f_error": np.array([0.01 * f_true[0], 1e6 * f_true[1]]),
        "band": np.array([3, 3]),
        "snr": np.array([100.0, 1e-6]),
        "detected": np.array([True, False]),
        "band_width": np.array([0.1, 0.1]),
    }
    fname = tmp_path / "truncated.svg"
    mock._plot_one_source_spectrum(0, "top", obs, params, params, model=model,
                                   fname=str(fname))

    assert fname.exists()
    assert fname.stat().st_size > 0
    # The function closes its own figure: nothing is left open to display.
    import matplotlib.pyplot as plt
    assert plt.get_fignums() == []


def test_source_observation_points_skip_off_detector_stamps():
    """A stamp that falls off the detector is not an observation.

    The catalog keeps sources within ``half_stamp`` of the detector edge, so a
    source can appear in an exposure while contributing essentially no flux.
    Its matched-filter S/N is then a numerical zero, and ``f_true / S/N`` would
    draw a bar ~1e10 times the flux - so the point must be dropped instead.
    """
    model = mock._build_spectrum_model(kind="blackbody", verbose=False)
    mock._set_spectrum_model(model)
    per_exposure_data, inf_exposures = _tiny_exposure(model=model, n_sources=2)

    positions, params, band, wcs, half_stamp, src_idx = per_exposure_data[0]
    # Source 1 is moved far outside the image: its postage stamp is empty.
    # Both lists carry the positions (the diagnostics read them from
    # ``per_exposure_data``, the S/N from ``inf_exposures``).
    moved = np.array(np.asarray(positions), dtype=np.float32, copy=True)
    moved[1] = [-200.0, 16.0]
    moved = jnp.asarray(moved)
    per_exposure_data = [(moved, params, band, wcs, half_stamp, src_idx)]
    noisy, sigma, _pos, gen, half, idx = inf_exposures[0]
    inf_exposures = [(noisy, sigma, moved, gen, half, idx)]

    true_params = np.asarray(params)
    snr_per_exposure = mock._per_exposure_source_snr(per_exposure_data,
                                                     inf_exposures)

    kept = mock._source_observation_points(
        0, per_exposure_data, inf_exposures, snr_per_exposure,
        true_params, true_params, model=model)
    dropped = mock._source_observation_points(
        1, per_exposure_data, inf_exposures, snr_per_exposure,
        true_params, true_params, model=model)

    assert kept["wavelength"].size == 1          # source 0 still observed
    assert dropped["wavelength"].size == 0       # source 1 was never observed
    assert dropped["f_error"].size == 0
    assert np.all(np.isfinite(kept["f_error"]))


def test_plot_source_spectra_writes_one_file_per_source(tmp_path):
    """The driver writes exactly one figure per selected source."""
    model = mock._build_spectrum_model(kind="blackbody", verbose=False)
    mock._set_spectrum_model(model)
    per_exposure_data, inf_exposures = _tiny_exposure(model=model, n_sources=3)

    true_params = np.asarray(per_exposure_data[0][1])
    recovered = true_params + 0.01
    snr_per_exposure = mock._per_exposure_source_snr(per_exposure_data,
                                                     inf_exposures)
    bright = np.array([True, True, False])

    written = mock.plot_source_spectra(
        true_params, recovered, per_exposure_data, inf_exposures, bright,
        snr_per_exposure, model=model, n_top=1, n_random=1,
        out_dir=str(tmp_path),
    )

    assert len(written) == 2
    for fname in written:
        assert fname.endswith(".svg")
        assert os.path.getsize(fname) > 0
    assert sorted(os.listdir(tmp_path)) == [
        "source_spectrum_00.svg", "source_spectrum_01.svg",
    ]

