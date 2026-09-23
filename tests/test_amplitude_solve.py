"""Tests for the direct amplitude least-squares solve.

The forward model is exactly linear in the source amplitude (the spectrum
model supplies only the *shape*), so ``solve_log_amplitudes`` should recover
the true amplitudes exactly on a noiseless problem, with the spectral shapes
held fixed.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
))

from spherex import SpherexImageGenerator3, BlackbodySpectrum  # noqa: E402
from inference import (  # noqa: E402
    EXPOSURE_TIME,
    compute_loss,
    infer_parameters,
    solve_log_amplitudes,
)


N_SOURCES = 6
HALF_STAMP = 10
N_LAMBDA = 3
OVERSAMPLING = 2
BACKGROUND_RATE = 1e-3


def _make_problem(seed=0, noise_rate=0.0):
    """Build a single-exposure synthetic problem with known amplitudes.

    Returns ``(exposures, log_backgrounds, true_log_params)`` in the format
    expected by the inference helpers.
    """
    gen = SpherexImageGenerator3(
        band=3, image_width=32, image_height=32, pixel_scale=6.2,
    )

    rng = np.random.default_rng(seed)
    positions = jnp.asarray(
        rng.uniform(8.0, 24.0, size=(N_SOURCES, 2)), dtype=jnp.float32
    )
    true_log_amp = np.log(rng.uniform(2e-21, 1e-16, size=N_SOURCES))
    true_log_T = np.log(rng.uniform(3.5, 7.0, size=N_SOURCES))
    true_log_params = jnp.asarray(
        np.column_stack([true_log_amp, true_log_T]), dtype=jnp.float32
    )

    rate = gen(
        positions, true_log_params,
        postage_stamp_half_size=HALF_STAMP,
        n_wavelength_samples=N_LAMBDA, oversampling=OVERSAMPLING,
    )
    counts = (np.asarray(rate) + BACKGROUND_RATE) * EXPOSURE_TIME

    # Poisson-like sigma derived from the (noiseless) expected counts, so the
    # normalised residuals are O(1) and float32 remains well conditioned.
    sigma = np.sqrt(np.maximum(counts, 0.0) + 1.0)
    if noise_rate > 0.0:
        counts = counts + rng.normal(size=counts.shape) * sigma

    exposures = [(
        jnp.asarray(counts, dtype=jnp.float32),
        jnp.asarray(sigma, dtype=jnp.float32),
        positions,
        gen,
        HALF_STAMP,
        jnp.arange(N_SOURCES, dtype=jnp.int32),
    )]
    log_backgrounds = jnp.array([np.log(BACKGROUND_RATE)], dtype=jnp.float32)

    return exposures, log_backgrounds, true_log_params


def test_solve_recovers_amplitudes_noiseless():
    """On a noiseless problem the exact amplitudes must be recovered."""
    exposures, log_backgrounds, true_log_params = _make_problem(seed=0)

    # Deliberately wrong initial amplitude guess (shapes are exact, since they
    # are held fixed by the solve).
    init = true_log_params.at[:, 0].add(0.7)

    log_amp, stats = solve_log_amplitudes(
        exposures, init, log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
    )

    assert "num_steps" in stats
    np.testing.assert_allclose(
        np.asarray(log_amp), np.asarray(true_log_params[:, 0]),
        rtol=1e-3, atol=1e-4,
    )


def test_solve_reduces_loss():
    """The solve must lower chi^2 / pixel versus the initial guess."""
    exposures, log_backgrounds, true_log_params = _make_problem(
        seed=1, noise_rate=0.0
    )
    init = true_log_params.at[:, 0].add(0.5)

    before = compute_loss(
        exposures, init, log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
    )

    log_amp, _stats = solve_log_amplitudes(
        exposures, init, log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
    )
    after_params = init.at[:, 0].set(log_amp)
    after = compute_loss(
        exposures, after_params, log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
    )

    assert after < before
    assert after < 1e-4   # noiseless: should fit essentially exactly


def test_solve_leaves_shapes_untouched():
    """Only the amplitude column may change."""
    exposures, log_backgrounds, true_log_params = _make_problem(seed=2)
    init = true_log_params.at[:, 0].add(-0.3)

    log_amp, _stats = solve_log_amplitudes(
        exposures, init, log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
    )
    updated = init.at[:, 0].set(log_amp)

    np.testing.assert_array_equal(
        np.asarray(updated[:, 1:]), np.asarray(init[:, 1:])
    )


def test_spectrum_model_has_no_amplitude():
    """Guard: the spectrum model must remain shape-only."""
    assert BlackbodySpectrum().n_params == 1


# ---------------------------------------------------------------------------
# The solve runs BEFORE the first SGD step
# ---------------------------------------------------------------------------

def test_infer_parameters_solves_amplitudes_before_the_first_step():
    """``infer_parameters`` must solve for the amplitudes up front.

    The drawn initial amplitudes are wrong by an arbitrary factor, so the early
    gradients - and the statistics the RMS preconditioner accumulates from them
    - would be dominated by an amplitude error the exact solve removes.  Two
    things are checked, and both would fail if the solve only happened at the
    ``amp_solve_every`` cadence or at the end of the loop:

    1. the *first* recorded loss (taken after step 0) is already the post-solve
       loss, not the raw initialisation's;
    2. the returned amplitudes are the least-squares solution for the INITIAL
       shapes and backgrounds.

    ``learning_rate=0`` makes the SGD steps no-ops, so the only thing that can
    change the parameters is the solve, and ``amp_solve_every`` is set beyond
    ``n_steps`` so the interlace never triggers within the loop.
    """
    exposures, log_backgrounds, true_log_params = _make_problem(seed=3)

    # True shapes (so the amplitude solution is exact), amplitudes wrong by
    # a factor exp(2) ~ 7.4.
    init = true_log_params.at[:, 0].add(2.0)
    raw_loss = float(compute_loss(
        exposures, init, log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
    ))

    rec_params, rec_backgrounds, losses, _lrs = infer_parameters(
        exposures, init,
        n_steps=2, learning_rate=0.0, momentum=0.0, warmup_steps=0,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
        amp_solve_every=10_000,          # never fires: isolate the initial solve
    )

    # (1) The solve already happened before the first recorded step.
    assert losses[0] < raw_loss / 10.0

    # (2) It is the exact solution for the initial shapes and the backgrounds
    #     the routine itself initialised (median of each image).
    solved, _stats = solve_log_amplitudes(
        exposures, init, rec_backgrounds,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
    )
    np.testing.assert_allclose(
        np.asarray(rec_params[:, 0]), np.asarray(solved),
        rtol=1e-5, atol=1e-6,
    )
    # With the true shapes held fixed (lr = 0), that solution is the true one.
    np.testing.assert_allclose(
        np.asarray(rec_params[:, 0]), np.asarray(true_log_params[:, 0]),
        rtol=1e-3, atol=1e-4,
    )
    # ...and no SGD step moved anything.
    np.testing.assert_array_equal(
        np.asarray(rec_params[:, 1:]), np.asarray(init[:, 1:]),
    )


# ---------------------------------------------------------------------------
# Multi-band / multi-exposure regression
# ---------------------------------------------------------------------------

def _make_multi_exposure_problem(seed=0):
    """Three exposures: band 1, then TWO band-5 exposures.

    Having two exposures share a band (and therefore a generator) exercises
    the band-grouped code path, while giving those two exposures *different*
    backgrounds exercises the per-exposure background handling inside the
    amplitude problem builder (a bug there silently mis-weights one group).
    """
    gen_by_band = {
        band: SpherexImageGenerator3(
            band=band, image_width=32, image_height=32, pixel_scale=6.2,
        )
        for band in (1, 5)
    }

    rng = np.random.default_rng(seed)
    positions = jnp.asarray(
        rng.uniform(10.0, 22.0, size=(N_SOURCES, 2)), dtype=jnp.float32
    )
    true_log_amp = np.log(rng.uniform(1e-15, 5e-11, size=N_SOURCES))
    true_log_T = np.log(rng.uniform(3.5, 7.0, size=N_SOURCES))
    true_log_params = jnp.asarray(
        np.column_stack([true_log_amp, true_log_T]), dtype=jnp.float32
    )

    # Distinct per-exposure background rates (rates, s^-1).
    bands = [1, 5, 5]
    backgrounds = [5.0, 0.03, 0.12]
    half_stamp = 8

    exposures = []
    for band, bg in zip(bands, backgrounds):
        gen = gen_by_band[band]
        rate = gen(
            positions, true_log_params,
            postage_stamp_half_size=half_stamp,
            n_wavelength_samples=N_LAMBDA, oversampling=OVERSAMPLING,
        )
        counts = (np.asarray(rate) + bg) * EXPOSURE_TIME
        sigma = np.sqrt(np.maximum(counts, 0.0) + 1.0)
        exposures.append((
            jnp.asarray(counts, dtype=jnp.float32),
            jnp.asarray(sigma, dtype=jnp.float32),
            positions,
            gen,
            half_stamp,
            jnp.arange(N_SOURCES, dtype=jnp.int32),
        ))

    log_backgrounds = jnp.asarray(np.log(backgrounds), dtype=jnp.float32)
    return exposures, log_backgrounds, true_log_params


def test_solve_multi_exposure_multiband():
    """Amplitudes must be recovered across band groups and exposures."""
    exposures, log_backgrounds, true_log_params = _make_multi_exposure_problem(
        seed=3
    )
    init = true_log_params.at[:, 0].add(0.6)

    log_amp, stats = solve_log_amplitudes(
        exposures, init, log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
    )

    assert stats["success"]
    np.testing.assert_allclose(
        np.asarray(log_amp), np.asarray(true_log_params[:, 0]),
        rtol=1e-2, atol=1e-3,
    )

    # And the fit must reach the true chi^2.
    after = compute_loss(
        exposures, init.at[:, 0].set(log_amp), log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=OVERSAMPLING,
    )
    assert after < 0.05
