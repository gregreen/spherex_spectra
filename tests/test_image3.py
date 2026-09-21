"""Correctness parity tests for the chunked-batch ``ImageGenerator3``.

Verifies that :class:`~spherex.image3.ImageGenerator3` produces the same
image as the reference :class:`~spherex.image.ImageGenerator`, across a
few different ``source_batch_size`` values (including full-batch and
single-source chunking).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from spherex import (
    BlackbodySpectrum,
    GaussianPSF,
    GaussianFilterTransmission,
    ImageGenerator,
    ImageGenerator3,
)


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
def reference_generator(psf, transmission, spectrum_model):
    return ImageGenerator(
        psf=psf,
        transmission=transmission,
        spectrum_model=spectrum_model,
        aperture=1.0,
    )


@pytest.fixture
def fast_generator(psf, transmission, spectrum_model):
    return ImageGenerator3(
        psf=psf,
        transmission=transmission,
        spectrum_model=spectrum_model,
        aperture=1.0,
    )


@pytest.fixture
def sources():
    rng = np.random.default_rng(0)
    n_sources = 5
    positions = jnp.asarray(
        rng.uniform(5, 25, size=(n_sources, 2)), dtype=jnp.float32
    )
    log_amplitudes = jnp.asarray(
        np.log(rng.uniform(1e-3, 1.0, size=n_sources)), dtype=jnp.float32
    )
    log_temperatures = jnp.asarray(
        np.log(rng.uniform(3.0, 8.0, size=n_sources)), dtype=jnp.float32
    )
    # source_params = [log_amplitude, log_temperature]  (amplitude FIRST)
    params = jnp.stack([log_amplitudes, log_temperatures], axis=-1)
    return positions, params


COMMON_KWARGS = dict(
    image_width=32,
    image_height=32,
    pixel_scale=1.0,
    postage_stamp_half_size=6,
    n_wavelength_samples=5,
    oversampling=2,
)


@pytest.mark.parametrize("source_batch_size", [None, 1, 2, 5])
def test_image3_matches_reference(
    reference_generator, fast_generator, sources, source_batch_size
):
    positions, params = sources

    ref_image = reference_generator(positions, params, **COMMON_KWARGS)
    fast_image = fast_generator(
        positions, params, source_batch_size=source_batch_size, **COMMON_KWARGS
    )

    np.testing.assert_allclose(
        np.asarray(fast_image), np.asarray(ref_image), rtol=1e-5, atol=1e-8
    )


def test_image3_grad(reference_generator, fast_generator, sources):
    """Gradients w.r.t. source params should also match (used for inference)."""
    import jax

    positions, params = sources

    def ref_loss(p):
        return jnp.sum(reference_generator(positions, p, **COMMON_KWARGS) ** 2)

    def fast_loss(p):
        return jnp.sum(fast_generator(positions, p, **COMMON_KWARGS) ** 2)

    ref_grad = jax.grad(ref_loss)(params)
    fast_grad = jax.grad(fast_loss)(params)

    np.testing.assert_allclose(
        np.asarray(fast_grad), np.asarray(ref_grad), rtol=1e-4, atol=1e-6
    )


def test_source_stamps_scatter_add_matches_image(
    reference_generator, fast_generator, sources
):
    """Scatter-adding per-source stamps must reproduce ``__call__``.

    ``source_stamps`` is the public hook used to build the diagonal
    preconditioner for the direct amplitude solve, so it must be exactly
    consistent with the image the generator produces.
    """
    positions, params = sources

    ref_image = reference_generator(positions, params, **COMMON_KWARGS)

    stamps, i_all, j_all = fast_generator.source_stamps(
        positions, params,
        image_width=COMMON_KWARGS["image_width"],
        image_height=COMMON_KWARGS["image_height"],
        pixel_scale=COMMON_KWARGS["pixel_scale"],
        postage_stamp_half_size=COMMON_KWARGS["postage_stamp_half_size"],
        n_wavelength_samples=COMMON_KWARGS["n_wavelength_samples"],
        oversampling=COMMON_KWARGS["oversampling"],
    )

    image = jnp.zeros(
        (COMMON_KWARGS["image_height"], COMMON_KWARGS["image_width"])
    )
    image = image.at[i_all, j_all].add(stamps)

    np.testing.assert_allclose(
        np.asarray(image), np.asarray(ref_image), rtol=1e-5, atol=1e-8
    )
