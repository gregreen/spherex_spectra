#!/usr/bin/env python3
"""Mock SPHEREx exposure simulation.

Generates a catalog of blackbody point sources, simulates multiple SPHEREx
exposures with random bands and pointing offsets, renders images via
``SpherexImageGenerator``, and saves them as percentile-clipped PNGs.

References
----------
* SPHEREx instrument: https://spherex.caltech.edu/page/instrument
"""

import os
import time
import hashlib
import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from PIL import Image

from spherex import (SpherexImageGenerator, SpherexImageGenerator3,
                     BlackbodySpectrum, NeuralNetSpectrum)
from spherex.config import _BANDS
from spherex.constants import HC_JAX, TEMPERATURE_UNIT
from spherex.spectrum import (
    LAMBDA_0, split_source_params, normalized_shape, normalized_log_shape,
)
from spherex.plotting_utils import HistEqNormalize
from jax.scipy.integrate import trapezoid
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import infer_parameters, infer_parameters_lm, compute_loss, compute_lm_loss, plot_loss_history, plot_comparison
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Reduce the pixel width and FOV of the images by this scale (vs. the default)
DOWNSAMPLE = 16

# PSF scaling (for making wavelength-dependent effects more visible) vs. the default
PSF_SCALE = 1.0

# Minimum postage-stamp half-size in pixels (floor).  Without this, a
# combination of small PSF, small wavelength range, or coarse pixel scale
# can produce half_stamp < 2, leaving too little context around each
# source for the model to fit the local background reliably.
HALF_STAMP_FLOOR = 3

N_SOURCES = 1024 // 8
N_EXPOSURES = 32
EXPOSURE_TIME = 15.0           # seconds  (approximate SPHEREx frame time)
PIXEL_SCALE = 6.2              # arcsec
DETECTOR_PIXELS = 2048 // DOWNSAMPLE

# Catalog covers a ~5° × 5° patch on the celestial equator
CATALOG_CENTER = SkyCoord(ra=180.0, dec=75.0, unit="deg", frame="icrs")
CATALOG_RADIUS = 3.5 / DOWNSAMPLE          # degrees  (spherical cap radius)

# Power-law index for source amplitudes
AMPLITUDE_ALPHA = 1.5          # P(A) ∝ A^{-alpha}

# Source amplitude range, specified DIRECTLY on the physical flux-density
# scale f_lambda(LAMBDA_0) in W m^-2 um^-1 (LAMBDA_0 = 1 um; see
# ``spherex.spectrum``).  There is NO photon-count targeting: amplitudes are
# drawn from this range (power-law P(A) ∝ A^-alpha) and the resulting photon
# counts are only *reported* as a diagnostic - see ``_report_photon_counts``.
# The values below correspond to roughly 1e2 .. 5e6 detected photons per
# exposure for a T = 3000 K source in a mid SPHEREx band.
AMPLITUDE_MIN = 1.0e-15
AMPLITUDE_MAX = 5.0e-11

# PNG percentile clipping
PNG_PERCENTILE_LO = 0.2
PNG_PERCENTILE_HI = 99.8

# "Bright" source selection for the diagnostic comparison plot: a source is
# labelled bright if its detection S/N exceeds SNR_THRESHOLD in at least
# MIN_BANDS distinct SPHEREx bands (not merely exposures).
SNR_THRESHOLD = 5.0
MIN_BANDS = 3

# Output directory
PLOTS_DIR = "plots"

# Background: peak pixel rate of A_min star × this factor
BACKGROUND_FACTOR = 2.0  # >1 makes faintest stars below background

# Number of wavelength samples used for the local filter-bandpass
# integral (see ``spherex.image._one_subpixel_rate``, quantile-based
# integration).  Used throughout: image generation, the true-parameter
# loss sanity check, inference, and the benchmark.  Thanks to the
# quantile-based integration, even small values are highly accurate -
# see ``compare_n_lambda()``.
N_LAMBDA = 1

# ---------------------------------------------------------------------------
# Spectrum model (the source's spectral SHAPE template)
# ---------------------------------------------------------------------------
# The mock can run with either analytic blackbodies or a frozen random neural
# network.  In both cases the model returns the source's UNNORMALISED
# LOG-FLUX (see ``spherex.spectrum``) and describes only the SHAPE: the
# generators anchor it at LAMBDA_0, so the amplitude stays in the first column
# of ``source_params`` (as log f_lambda(LAMBDA_0)) and is drawn / inferred by
# the usual machinery.  Nothing downstream of the shape (image generation,
# PSF, transmissions, the amplitude solve, the optimisers) changes.
#
# "blackbody" : analytic BlackbodySpectrum, 1 shape parameter (log T).
# "nn"        : NeuralNetSpectrum with FROZEN random weights; the number of
#               shape parameters (theta) is free.
SPECTRUM_KIND = "nn"

# Neural-network hyperparameters (only used if SPECTRUM_KIND = "nn")
NN_N_PARAMS = 1
NN_N_HIDDEN_LAYERS = 1
NN_HIDDEN_SIZE = 32
NN_SEED = 314159

# Fourier (positional) embedding of the wavelength: the input vector carries
# ln(wavelength) plus sin/cos of NN_N_EMBEDDINGS geometrically spaced
# frequencies covering NN_DELTA_LN_WAVELENGTH e-foldings of wavelength.  Log
# space keeps the embedding scale-invariant, matching the instrument: a
# bandpass has a constant *fractional* width, so one setting gives every band
# the same relative resolution.  More embeddings let the shape bend on finer
# scales - but note that the image generators integrate each pixel's bandpass
# with as few as N_LAMBDA samples, which is only valid while the shape varies
# slowly across the filter (see ``NeuralNetSpectrum``'s quadrature caution).
NN_N_EMBEDDINGS = 8
NN_DELTA_LN_WAVELENGTH = np.log(5.0 / 0.75)   # full 0.75 - 5.0 um range

# The randomly initialised network is rescaled so that its in-band
# LAMBDA_0-normalised log-shape has roughly this standard deviation over the
# prior, i.e. |log shape| ~ 1-2 out to 2-4 sigma.  Without this the frozen
# random net could produce spectra spanning many decades, which would move the
# amplitude range / background / detection S/N regime away from the blackbody
# run that the amplitudes were chosen for.
NN_TARGET_LOG_SHAPE_STD = 0.5
NN_RESCALE_SAMPLES = 64

# Standard deviation of the INITIAL theta guess.  Kept separate from the theta
# PRIOR (which is N(0, 1)) because the initialisation is deliberately
# different for each model - as it also is for the blackbody, whose prior
# (log of a uniform draw) is not the same as its initialisation (uniform in
# log T).
NN_INIT_STD = 1.0

# Reference shape parameters used by the analytic (blackbody-style)
# diagnostics: a plain 3000 K blackbody for the blackbody model, and theta = 0
# for the neural network.  These only set the *reported* photon counts and the
# background level - they do not target any photon count.
REFERENCE_TEMPERATURE_K = 3.0

# Blackbody temperature prior / initialisation bounds, log(kK).
# Prior:        log T = log(U(3, 8) kK)
# Init guess:   log T ~ U(LOG_T_BOUNDS)   (deliberately not the same draw)
LOG_T_BOUNDS = np.log(
    ([3000.0, 8000.0] * u.K).to(u.Unit(TEMPERATURE_UNIT)).value
)

# ---------------------------------------------------------------------------
# Step 0: Spectrum model (shape template) + shape-parameter bookkeeping
# ---------------------------------------------------------------------------
# Everything model-specific about the mock lives in this block.  The rest of
# the pipeline is generic in the number of shape parameters P: source
# parameters are always ``[log_amplitude, theta_0, ..., theta_{P-1}]``.

#: The single spectrum-model instance used by the WHOLE simulation.  Rebuilt
#: (and replaced) at the start of each top-level entry point via
#: :func:`_set_spectrum_model`.  It is never handed to an optimiser, so its
#: weights are frozen by construction: the inference helpers differentiate only
#: ``(log_params, log_backgrounds)`` and see the generator purely as a
#: closed-over constant.
SPECTRUM_MODEL = None


def _is_blackbody(model):
    """True if ``model`` is the analytic blackbody template."""
    return isinstance(model, BlackbodySpectrum)


def _wavelength_grid(n_points=512):
    """Wavelength grid (um) spanning every SPHEREx band, LOG-spaced.

    The neural network embeds the wavelength in log space, so its finest
    Fourier features have a constant width in ``ln(lambda)`` - a linear grid
    would under-sample them at the short-wavelength end (and over-sample the
    long-wavelength end).  A log-spaced grid gives every feature the same
    number of samples, and is what the shape diagnostics and the output
    rescaling should measure.
    """
    lam_min = min(lo for lo, _hi, _r, _name in _BANDS.values())
    lam_max = max(hi for _lo, hi, _r, _name in _BANDS.values())
    return jnp.logspace(jnp.log10(lam_min), jnp.log10(lam_max), n_points)


def _rescale_nn_output(model, key, target_std=NN_TARGET_LOG_SHAPE_STD,
                       n_samples=NN_RESCALE_SAMPLES):
    """Rescale a :class:`NeuralNetSpectrum` output layer to an O(1) log-shape.

    The dimensionless shape is
    ``exp(model(lambda, theta) - model(LAMBDA_0, theta))`` (see
    ``spherex.spectrum``), which is *exactly linear* in the final layer's
    weight block: the final bias cancels in the difference, so scaling that
    block by ``factor`` scales the log-shape by exactly ``factor``.  One
    measurement therefore suffices - draw theta from the prior, measure the
    log-shape's spread over the full wavelength range, and scale the weight
    block to hit ``target_std``.

    This keeps a *randomly initialised* network in the same dynamic range as
    the blackbody it replaces, so the directly-specified amplitude range, the
    background level and the resulting detection S/N all stay meaningful.

    Returns
    -------
    (model, factor) : the rescaled model and the factor applied.
    """
    lambdas = _wavelength_grid()
    thetas = jax.random.normal(key, (n_samples, model.n_params))

    # The normalised log-shape is exactly the log-flux minus its value at
    # LAMBDA_0 - the quantity the generators compute per source.
    log_shape = jax.vmap(
        lambda theta: normalized_log_shape(model, lambdas, theta)
    )(thetas)                                         # (n_samples, n_lambda)
    current = float(jnp.std(log_shape))
    if not np.isfinite(current) or current <= 0.0:
        return model, 1.0

    factor = float(target_std) / current
    last = len(model.layers) - 1
    rescaled = eqx.tree_at(
        lambda m: m.layers[last].weight,
        model,
        model.layers[last].weight * factor,
    )
    return rescaled, factor


def _build_spectrum_model(
    kind=SPECTRUM_KIND,
    n_params=NN_N_PARAMS,
    n_hidden_layers=NN_N_HIDDEN_LAYERS,
    hidden_size=NN_HIDDEN_SIZE,
    seed=NN_SEED,
    n_embeddings=NN_N_EMBEDDINGS,
    delta_ln_wavelength=NN_DELTA_LN_WAVELENGTH,
    *,
    target_log_shape_std=NN_TARGET_LOG_SHAPE_STD,
    verbose=True,
):
    """Build the (frozen) spectrum-shape template used by the mock.

    Parameters
    ----------
    kind : {"blackbody", "nn"}
    n_params : int
        Number of shape parameters ``theta`` (neural-network model only).
    n_hidden_layers, hidden_size : int
    seed : int
        Seed for the neural network's weight initialisation, and (with a
        derived key) for the output rescaling.
    n_embeddings, delta_ln_wavelength : int, float
        Wavelength Fourier-embedding hyperparameters in log-wavelength space
        (neural-network model only; see ``spherex.spectrum.NeuralNetSpectrum``).
    target_log_shape_std : float
        Target standard deviation of the ``LAMBDA_0``-normalised log-shape.

    Returns
    -------
    equinox.Module
        One instance, to be shared by every band.  Sharing matters for the
        neural network, whose weights are state: a per-band copy would give
        each band a different spectrum.
    """
    if kind == "blackbody":
        model = BlackbodySpectrum()
        if verbose:
            print("  Spectrum model: analytic blackbody "
                  f"(P = {model.n_params} shape parameter(s))")
        return model

    if kind != "nn":
        raise ValueError(
            f"Unknown spectrum kind {kind!r}; expected 'blackbody' or 'nn'"
        )

    model = NeuralNetSpectrum(
        n_params, n_hidden_layers, hidden_size,
        n_embeddings=n_embeddings, delta_ln_wavelength=delta_ln_wavelength,
        key=jax.random.PRNGKey(seed),
    )
    model, factor = _rescale_nn_output(
        model, jax.random.PRNGKey(seed + 1), target_log_shape_std
    )
    if verbose:
        print(f"  Spectrum model: NeuralNetSpectrum(P={n_params}, "
              f"layers={n_hidden_layers}, hidden={hidden_size}, "
              f"embeddings={n_embeddings}/{delta_ln_wavelength:.3f}ln-lambda, "
              f"seed={seed}), random weights FROZEN")
        print(f"    output layer rescaled by {factor:.4g} to give an in-band "
              f"log-shape std of ~{target_log_shape_std:g}")
    return model


def _set_spectrum_model(model):
    """Install ``model`` as the module-level active spectrum model."""
    global SPECTRUM_MODEL
    SPECTRUM_MODEL = model
    return model


# Default active model: the analytic blackbody, built once at import so that
# the helper functions below always have something to fall back on.  Every
# top-level entry point replaces it via ``_set_spectrum_model``.
SPECTRUM_MODEL = _build_spectrum_model(verbose=False)


def _model_fingerprint(model):
    """Order-stable fingerprint of every leaf of ``model``.

    Used to prove the spectrum model is *frozen*: the weights must be
    bit-identical before and after inference.
    """
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(model):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _reference_shape_params(model=None):
    """Reference shape parameters for the analytic diagnostics.

    A 3000 K blackbody for the blackbody model, ``theta = 0`` for the neural
    network.  Only the *reported* photon counts and the background level use
    this, so it does not impose any photon-count target.
    """
    model = SPECTRUM_MODEL if model is None else model
    if _is_blackbody(model):
        return jnp.array([np.log(REFERENCE_TEMPERATURE_K)])
    return jnp.zeros(model.n_params)


def _draw_shape_params(rng, n_sources, model=None):
    """Draw ``(n_sources, P)`` shape parameters from the model's PRIOR.

    (For the neural network the prior is ``theta ~ N(0, I)`` per column; for
    the blackbody it is ``log T = log(U(3, 8) kK)``.)
    """
    model = SPECTRUM_MODEL if model is None else model
    if _is_blackbody(model):
        temperatures = rng.uniform(
            np.exp(LOG_T_BOUNDS[0]), np.exp(LOG_T_BOUNDS[1]), n_sources
        )
        return np.log(temperatures)[:, None]
    return rng.standard_normal((n_sources, model.n_params))


def _draw_init_shape_params(rng, n_sources, model=None):
    """Draw ``(n_sources, P)`` INITIAL-GUESS shape parameters.

    Deliberately *different* from :func:`_draw_shape_params` (and different
    between models), so that the inference starts away from the prior's own
    sampling distribution: the blackbody is initialised uniformly in log T,
    the neural network from ``N(0, NN_INIT_STD^2)``.
    """
    model = SPECTRUM_MODEL if model is None else model
    if _is_blackbody(model):
        return rng.uniform(
            LOG_T_BOUNDS[0], LOG_T_BOUNDS[1], (n_sources, 1)
        )
    return NN_INIT_STD * rng.standard_normal((n_sources, model.n_params))


def _shape_param_labels(model=None):
    """Human-readable label per shape parameter (length ``P``)."""
    model = SPECTRUM_MODEL if model is None else model
    if _is_blackbody(model):
        return ["log(T / kK)"]
    return [f"theta_{k}" for k in range(model.n_params)]


def _describe_shape_params(shape_params, model=None):
    """One-line summary of a ``(N, P)`` shape-parameter array."""
    model = SPECTRUM_MODEL if model is None else model
    shape_params = np.asarray(shape_params)
    if _is_blackbody(model):
        t = np.exp(shape_params[:, 0])
        return f"T range: [{t.min():.1f}, {t.max():.1f}] kK"
    return (f"theta range: [{shape_params.min():+.3f}, "
            f"{shape_params.max():+.3f}] (P = {shape_params.shape[1]})")


def _make_params(log_amplitudes, shape_params):
    """Assemble ``(N, 1 + P)`` source parameters, amplitude FIRST.

    The only place the parameter layout is constructed, so the amplitude-first
    convention is enforced in exactly one spot.
    """
    return jnp.asarray(
        np.column_stack([np.asarray(log_amplitudes), np.asarray(shape_params)]),
        dtype=jnp.float32,
    )


# ---------------------------------------------------------------------------
# Step 1: Amplitude range + photon-count diagnostic
# ---------------------------------------------------------------------------
# Amplitudes are drawn directly in [AMPLITUDE_MIN, AMPLITUDE_MAX] (defined in
# the configuration block above).  There is no photon-count *targeting*;
# instead we report the detected counts implied by the chosen range.

def _total_photons_per_exposure(band, amplitude, model=None):
    """Analytic detected photons per exposure for the active spectrum model.

    Integrates the telescope-collected photon rate over the band, including
    the Gaussian filter transmission, and multiplies by ``EXPOSURE_TIME``.
    ``amplitude`` is f_lambda at LAMBDA_0 (W m^-2 um^-1).  The model's shape is
    evaluated at the *reference* shape parameters
    (:func:`_reference_shape_params`), so the count is a faithful
    representative of the model's typical brightness.  Used only as a
    diagnostic and to set the background level (no count targeting).
    """
    model = SPECTRUM_MODEL if model is None else model
    lam_min, lam_max, R, _name = _BANDS[band]
    lambdas = jnp.linspace(lam_min, lam_max, 500)
    lam_mid = 0.5 * (lam_min + lam_max)
    sigma_t = lam_mid / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)) * R)
    T_weight = jnp.exp(-(lambdas - lam_mid) ** 2 / (2.0 * sigma_t ** 2))

    shape = normalized_shape(model, lambdas, _reference_shape_params(model))
    f_lam = amplitude * shape

    aperture = jnp.pi * (0.10) ** 2
    rate = aperture * trapezoid((lambdas / HC_JAX) * f_lam * T_weight, lambdas)
    return float(rate) * EXPOSURE_TIME


def _report_photon_counts(model=None):
    """Print the photon-count range implied by AMPLITUDE_MIN/AMPLITUDE_MAX."""
    model = SPECTRUM_MODEL if model is None else model
    if _is_blackbody(model):
        reference = (f"log(T / kK) = {np.log(REFERENCE_TEMPERATURE_K):.3f} "
                     f"(T = {REFERENCE_TEMPERATURE_K * 1000.0:.0f} K)")
    else:
        reference = "theta = 0"
    print("\n--- Amplitude range (specified directly, no count targeting) ---")
    print(f"  f_lambda(LAMBDA_0={LAMBDA_0} um) in "
          f"[{AMPLITUDE_MIN:.3e}, {AMPLITUDE_MAX:.3e}] W m^-2 um^-1")
    print(f"  evaluated at the reference shape parameters: {reference}")
    for band in sorted(_BANDS):
        lo = _total_photons_per_exposure(band, AMPLITUDE_MIN, model)
        hi = _total_photons_per_exposure(band, AMPLITUDE_MAX, model)
        print(f"  Band {band}: {lo:10.1f} - {hi:12.1f} photons / exposure")


# ---------------------------------------------------------------------------
# Step 2: Generate source catalog
# ---------------------------------------------------------------------------

def _generate_catalog(rng, ra_center=None, dec_center=None,
                       radius_deg=None, model=None):
    """Return (skycoords, shape_params, log_amplitudes).

    Sources are drawn uniformly from the surface of a sphere within a
    spherical cap of radius ``radius_deg`` centred on (``ra_center``,
    ``dec_center``).  Shape parameters come from the *active model's prior*
    (``log T = log(U(3, 8) kK)`` for the blackbody, ``theta ~ N(0, I)`` for the
    neural network) and amplitudes are power-law distributed
    (P(A) ∝ A^-alpha) over the directly-specified range
    [AMPLITUDE_MIN, AMPLITUDE_MAX].

    Parameters
    ----------
    rng : np.random.Generator
    ra_center, dec_center, radius_deg : float, optional
    model : equinox.Module, optional
        Spectrum model whose prior to draw from; defaults to the module-level
        :data:`SPECTRUM_MODEL`.

    Returns
    -------
    skycoords : SkyCoord, shape (N_SOURCES,)
    shape_params : np.ndarray, shape (N_SOURCES, P)
    log_amplitudes : np.ndarray, shape (N_SOURCES,)
    """
    model = SPECTRUM_MODEL if model is None else model
    if ra_center is None:
        ra_center = CATALOG_CENTER.ra.deg
    if dec_center is None:
        dec_center = CATALOG_CENTER.dec.deg
    if radius_deg is None:
        radius_deg = CATALOG_RADIUS

    center = SkyCoord(ra=ra_center, dec=dec_center, unit="deg", frame="icrs")
    theta_max = np.deg2rad(radius_deg)

    # Uniform on the sphere within a spherical cap:
    #   cos(theta) ~ U[cos(theta_max), 1]
    #   phi        ~ U[0, 2*pi)
    cos_theta = rng.uniform(np.cos(theta_max), 1.0, N_SOURCES)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2.0 * np.pi, N_SOURCES)

    skycoords = center.directional_offset_by(
        phi * u.rad, theta * u.rad
    )

    # Spectral shape parameters, drawn from the model's prior.
    shape_params = _draw_shape_params(rng, N_SOURCES, model)

    # Amplitudes: truncated power-law  P(A) ∝ A^{-alpha}
    alpha = AMPLITUDE_ALPHA
    u_vals = rng.uniform(0.0, 1.0, N_SOURCES)
    # Inverse CDF for A^{-alpha} on [AMPLITUDE_MIN, AMPLITUDE_MAX]:
    #   CDF(A) = (A^{1-alpha} - A_min^{1-alpha}) / (A_max^{1-alpha} - A_min^{1-alpha})
    #   => A = [A_min^{1-alpha} + u * (A_max^{1-alpha} - A_min^{1-alpha})]^{1/(1-alpha)}
    exp = 1.0 - alpha
    A_min_exp = AMPLITUDE_MIN ** exp
    A_max_exp = AMPLITUDE_MAX ** exp
    amplitudes = (A_min_exp + u_vals * (A_max_exp - A_min_exp)) ** (1.0 / exp)
    log_amplitudes = np.log(amplitudes)

    return skycoords, shape_params, log_amplitudes


# ---------------------------------------------------------------------------
# Step 3: Generate exposure WCSs
# ---------------------------------------------------------------------------

def _make_wcs(ra_center, dec_center):
    """Build an astropy WCS for a 2048×2048 detector at 6.2 arcsec/pixel."""
    w = WCS(naxis=2)
    w.wcs.crpix = [DETECTOR_PIXELS / 2.0, DETECTOR_PIXELS / 2.0]
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.cd = [[-PIXEL_SCALE / 3600.0, 0.0], [0.0, PIXEL_SCALE / 3600.0]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (DETECTOR_PIXELS, DETECTOR_PIXELS)
    return w


def _generate_exposures(rng):
    """Return list of (band, wcs, half_stamp) tuples."""
    exposures = []
    # Maximum offset from catalog centre so detector stays within catalog
    max_offset = CATALOG_RADIUS - 0.5 * (DETECTOR_PIXELS * PIXEL_SCALE / 3600.0) * np.sqrt(2)

    for _ in range(N_EXPOSURES):
        band = int(rng.integers(1, 7))       # 1..6
        # Random offset via exponential map (tangent-plane projection).
        sep = max_offset * np.sqrt(rng.uniform(0.0, 1.0))  # deg
        pa = rng.uniform(0.0, 360.0)                        # deg
        center = CATALOG_CENTER.directional_offset_by(
            pa * u.deg, sep * u.deg
        )

        wcs = _make_wcs(center.ra.deg, center.dec.deg)

        # Postage stamp half-size: capture ~3× FWHM of PSF at band centre
        lam_min, lam_max, _R, _name = _BANDS[band]
        lam_mid = 0.5 * (lam_min + lam_max)
        fwhm = PSF_SCALE * 6.0 * (lam_mid / 1.0)        # PSF FWHM proportional to lambda
        half_stamp = max(int(np.ceil(5.0 * fwhm / PIXEL_SCALE)), HALF_STAMP_FLOOR)

        exposures.append((band, wcs, half_stamp))
        print(f"  Exposure {len(exposures)-1}: Band {band}, "
              f"center=({center.ra.deg:.3f}°, {center.dec.deg:.3f}°), "
              f"half_stamp={half_stamp}")

    print(f'Included bands: {np.unique([d[0] for d in exposures])}')

    return exposures


# ---------------------------------------------------------------------------
# Step 4: Filter sources per exposure
# ---------------------------------------------------------------------------

def _filter_sources(skycoords, shape_params, log_amplitudes, exposures):
    """Filter to sources observed in at least one exposure.

    Returns
    -------
    skycoords, shape_params, log_amplitudes  (filtered)
    per_exposure_data : list of (positions_pix, params_array, band, wcs, half_stamp, src_idx)
        ``params_array`` has shape ``(n_in, 1 + P)``: the log-amplitude first,
        then the ``P`` shape parameters (see ``spherex.spectrum``).  This works
        for any ``P`` - the blackbody simply has ``P = 1``.
    """
    n_total = len(skycoords)
    in_any = np.zeros(n_total, dtype=bool)

    per_exposure_data = []

    for band, wcs, half_stamp in exposures:
        # Pixel coordinates of all sources in this exposure
        x_pix, y_pix = wcs.world_to_pixel(skycoords)

        # Include sources within half_stamp of detector edges
        in_this = (
            (x_pix >= -half_stamp)
            & (x_pix < DETECTOR_PIXELS + half_stamp)
            & (y_pix >= -half_stamp)
            & (y_pix < DETECTOR_PIXELS + half_stamp)
        )
        in_any |= in_this

        n_in = np.sum(in_this)
        print(f"  Band {band}: {n_in} sources in/near FoV")

        # Convert pixel coords to pixel-based positions for the generator
        x_px = np.asarray(x_pix[in_this], dtype=np.float32)
        y_px = np.asarray(y_pix[in_this], dtype=np.float32)
        positions_pix = jnp.stack(
            [jnp.asarray(x_px), jnp.asarray(y_px)], axis=-1
        )
        # source_params = [log_amplitude, theta_0, ..., theta_{P-1}]
        # (amplitude ALWAYS first; shape columns follow, whatever P is).
        params = _make_params(log_amplitudes[in_this], shape_params[in_this])

        per_exposure_data.append(
            (positions_pix, params, band, wcs, half_stamp,
             np.where(in_this)[0].astype(np.int32))
        )

    # Filter catalog to sources in at least one exposure
    n_kept = np.sum(in_any)
    print(f"\nKeeping {n_kept} / {n_total} sources (in ≥1 exposure)")

    skycoords_filt = skycoords[in_any]
    shape_params_filt = shape_params[in_any]
    log_amplitudes_filt = log_amplitudes[in_any]

    return (skycoords_filt, shape_params_filt, log_amplitudes_filt,
            per_exposure_data)


# ---------------------------------------------------------------------------
# Step 5 & 6: Generate images and save
# ---------------------------------------------------------------------------


def _compute_background(band, model=None):
    """Estimate the background level so the faintest stars sit below it.

    Computes the approximate peak pixel rate for an ``AMPLITUDE_MIN`` source
    with the *active* spectrum model at its reference shape parameters (a
    3000 K blackbody for the blackbody model, ``theta = 0`` for the neural
    network), then scales by ``BACKGROUND_FACTOR``.
    """
    model = SPECTRUM_MODEL if model is None else model
    lam_min, lam_max, R, _name = _BANDS[band]
    lam_mid = 0.5 * (lam_min + lam_max)

    # PSF at band centre
    fwhm = PSF_SCALE * 6.0 * (lam_mid / 1.0)               # arcsec
    sigma_psf = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # arcsec
    psf_peak = 1.0 / (2.0 * np.pi * sigma_psf**2)  # arcsec^-2

    # Flux density at band centre: amplitude (f_lambda at LAMBDA_0) times the
    # dimensionless spectrum *shape*.
    lam_arr = jnp.array([lam_mid])
    shape = normalized_shape(model, lam_arr, _reference_shape_params(model))
    f_lam = float(AMPLITUDE_MIN * shape[0])       # W / (m^2 um)

    # Effective bandwidth of the Gaussian transmission
    # sigma = lambda_mid / (2.355 * R)
    sigma_t = lam_mid / (2.0 * np.sqrt(2.0 * np.log(2.0)) * R)
    bandwidth_eff = np.sqrt(2.0 * np.pi) * sigma_t  # um

    # Photon rate in peak pixel
    aperture = np.pi * (0.10) ** 2
    photon_energy_factor = float(lam_mid / HC_JAX)   # photons / J
    peak_rate = (
        aperture
        * PIXEL_SCALE**2
        * photon_energy_factor
        * f_lam
        * psf_peak
        * bandwidth_eff
    )                                                # s^-1

    background = BACKGROUND_FACTOR * peak_rate
    print(f"  Band {band}: A_min peak = {peak_rate:.4e} s^-1, "
          f"background = {background:.4e} s^-1 per pixel")
    return background


def _generate_and_save(per_exposure_data, model=None):
    """For each exposure, generate an image (timed) and save as PNG.

    Every band's generator is built with the SAME ``model`` instance, so a
    stateful spectrum model (a neural network) produces one consistent
    spectrum per source across all bands.

    Returns
    -------
    inference_exposures : list of (noisy_img, sigma_img, wcs, gen_module)
    noisy_vmins, noisy_vmaxs : per-exposure stretch bounds
    """
    model = SPECTRUM_MODEL if model is None else model
    os.makedirs(PLOTS_DIR, exist_ok=True)
    band_generators = {}  # reuse per band
    inference_exposures = []
    noisy_vmins = []
    noisy_vmaxs = []

    for i, (positions_pix, params, band, wcs, half_stamp, src_idx) in enumerate(
        per_exposure_data
    ):
        print(f"\nGenerating image {i} (Band {band}, {params.shape[0]} sources)...")

        if band not in band_generators:
            band_generators[band] = SpherexImageGenerator3(
                band=band, psf_scale=PSF_SCALE,
                image_width=DETECTOR_PIXELS,
                image_height=DETECTOR_PIXELS,
                lambda_slope_scale=DOWNSAMPLE,
                spectrum_model=model,
            )
        gen = band_generators[band]

        t0 = time.perf_counter()
        img = gen(
            positions_pix,
            params,
            postage_stamp_half_size=half_stamp,
            n_wavelength_samples=N_LAMBDA,
            oversampling=2,
        )
        img.block_until_ready()
        dt = time.perf_counter() - t0

        img_np = np.asarray(img)
        print(f"  Generation time: {dt:.2f} s")
        print(f"  Image range: [{img_np.min():.4e}, {img_np.max():.4e}]")
        print(f"  Total flux:   {img_np.sum():.4e} s^-1")

        # --- noisy image ----------------------------------------------------
        # ``img_np``/``bg`` are photon RATES (s^-1); multiply by
        # EXPOSURE_TIME to get expected photon COUNTS before drawing
        # Poisson noise (both source and background must be scaled the
        # same way - previously only the background was, which
        # systematically under-counted the source flux relative to a
        # correctly-scaled background).
        bg = _compute_background(band, model)
        img_with_bg = (img_np + bg) * EXPOSURE_TIME
        # img_noisy = np.random.default_rng(i).poisson(img_with_bg)
        # sigma should reflect the TRUE Poisson variance of the simulated
        # process (variance = mean), not the noisy realisation with an
        # arbitrary +1 floor - that floor is far larger than the true
        # variance in the low-count regime this simulation lives in
        # (background ~0.01-0.5 counts/pixel), which was silently
        # suppressing chi^2 well below 1.
        # sigma_img = np.sqrt(np.maximum(img_with_bg, 1e-6))
        noise_floor = 1.0 # Minimum uncertainty on counts per pixel
        sigma_img = np.sqrt(img_with_bg + noise_floor**2)
        img_noisy = img_with_bg + np.random.default_rng(i).normal(size=img_with_bg.shape) * sigma_img

        print('true percentiles:', np.percentile(img_with_bg, [0.2, 1., 10, 50., 90., 99., 99.8]))
        print('noisy percentiles:', np.percentile(img_noisy, [0.2, 1., 10, 50., 90., 99., 99.8]))

        vmin_n, vmax_n = np.percentile(
            img_noisy, [PNG_PERCENTILE_LO, PNG_PERCENTILE_HI]
        )
        if vmax_n <= vmin_n:
            vmax_n = vmin_n + 1e-30
        noisy_vmins.append(vmin_n)
        noisy_vmaxs.append(vmax_n)

        # --- save true image (noisy vmin/vmax) -------------------------------
        # Use ``img_with_bg`` (noiseless expected COUNTS, source+background,
        # scaled by EXPOSURE_TIME) - NOT the raw source-only rate ``img_np`` -
        # so "true" is on the same scale as "noisy"/"pred" and their shared
        # vmin/vmax (from the noisy image's percentiles) is meaningful.
        img_t_scaled = np.clip(
            (img_with_bg - vmin_n) / (vmax_n - vmin_n) * 255.0, 0, 255
        ).astype(np.uint8)
        fname_true = os.path.join(
            PLOTS_DIR, f"exposure_{i:03d}_band_{band}_true.png"
        )
        Image.fromarray(img_t_scaled.T, mode="L").save(fname_true)
        print(f"  Saved {fname_true}")

        # --- save noisy image ------------------------------------------------
        img_n_scaled = np.clip(
            (img_noisy - vmin_n) / (vmax_n - vmin_n) * 255.0, 0, 255
        ).astype(np.uint8)
        fname_noisy = os.path.join(
            PLOTS_DIR, f"exposure_{i:03d}_band_{band}_noisy.png"
        )
        Image.fromarray(img_n_scaled.T, mode="L").save(fname_noisy)
        print(f"  Saved {fname_noisy}")

        # --- collect inference data ------------------------------------------
        inference_exposures.append(
            (jnp.asarray(img_noisy, dtype=jnp.float32),
             jnp.asarray(sigma_img, dtype=jnp.float32),
             positions_pix,
             gen, half_stamp,
             jnp.asarray(src_idx))
        )

    return inference_exposures, noisy_vmins, noisy_vmaxs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Band colors (tab10)
_BAND_COLORS = {
    1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c",
    4: "#d62728", 5: "#9467bd", 6: "#8c564b",
}

def _plot_sky_locations(skycoords, exposures, fname):
    """Plot catalog sources and exposure footprints on the sky."""
    # Build a WCS centred on the catalog
    plot_wcs = WCS(naxis=2)
    plot_wcs.wcs.crpix = [512, 512]
    plot_wcs.wcs.crval = [CATALOG_CENTER.ra.deg, CATALOG_CENTER.dec.deg]
    margin = 1.15  # show a bit beyond the catalog edges
    fov_deg = CATALOG_RADIUS * 2.0 * margin
    cdelt = fov_deg / 1024.0
    plot_wcs.wcs.cd = [[-cdelt, 0], [0, cdelt]]
    plot_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    plot_wcs.pixel_shape = (1024, 1024)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=plot_wcs)

    # Catalog sources
    ax.scatter(
        skycoords.ra.deg, skycoords.dec.deg,
        transform=ax.get_transform("world"),
        s=0.5, color="gray", alpha=0.5, rasterized=True,
    )

    # Exposure footprints
    px_corners = np.array([
        [0, 0],
        [DETECTOR_PIXELS - 1, 0],
        [DETECTOR_PIXELS - 1, DETECTOR_PIXELS - 1],
        [0, DETECTOR_PIXELS - 1],
        [0, 0],   # close the loop
    ])
    for band, wcs, _half_stamp in exposures:
        ra_c, dec_c = wcs.all_pix2world(px_corners[:, 0], px_corners[:, 1], 0)
        ax.plot(ra_c, dec_c, transform=ax.get_transform("world"),
                color=_BAND_COLORS.get(band, "black"), linewidth=1.5,
                label=f"Band {band}")

    # Deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(f"Catalog ({len(skycoords)} sources) + {len(exposures)} exposures")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def _source_snr(gen, positions, params, sigma_img, half_stamp,
                n_lambda=N_LAMBDA, oversampling=2):
    """Estimated detection S/N of each source in a single exposure.

    Uses the expected matched-filter significance of the source's own
    (noiseless) model counts against the per-pixel noise::

        S/N = sqrt( sum_i (m_i * t_exp)^2 / sigma_i^2 )

    where ``m_i`` is the source-only model rate in pixel ``i`` (taken from its
    unit-amplitude postage stamp, so it includes the PSF and the source's
    amplitude) and ``sigma_i`` the per-pixel noise.  This is the expectation
    of the optimal (matched-filter) flux estimator, i.e. how many sigma the
    source's flux stands above the noise.  It uses the model rather than the
    noisy data, so it is neither noise-biased nor circular (the *true*
    parameters are used, never the recovered ones), and pixels with no source
    flux drop out automatically.

    Returns
    -------
    np.ndarray, shape (S,)
        S/N per source, in the exposure's local source ordering.
    """
    stamps, i_all, j_all = gen.source_stamps(
        positions, params,
        image_width=gen.image_width,
        image_height=gen.image_height,
        pixel_scale=gen.pixel_scale,
        postage_stamp_half_size=half_stamp,
        n_wavelength_samples=n_lambda,
        oversampling=oversampling,
    )
    counts = np.asarray(stamps) * EXPOSURE_TIME          # (S, ss, ss)
    sig = np.asarray(sigma_img)[np.asarray(i_all), np.asarray(j_all)]
    return np.sqrt(((counts / sig) ** 2).sum(axis=(1, 2)))


def _compute_bright_mask(per_exposure_data, inf_exposures, n_sources,
                         snr_threshold=SNR_THRESHOLD, min_bands=MIN_BANDS,
                         n_lambda=N_LAMBDA, oversampling=2):
    """Flag sources detected in enough distinct bands to be well measured.

    For every exposure the detection S/N of each source is estimated with
    :func:`_source_snr`, and the exposure's band is recorded whenever the
    threshold is exceeded.  A source is "bright" if it clears the threshold
    in at least ``min_bands`` **unique** bands.

    Counting *bands* rather than exposures matters: a source repeatedly
    observed in one band still only constrains a single point of its
    spectrum, so it is not a useful case for judging recovered parameters.
    Only sources actually present in an exposure are considered (via
    ``src_idx``), so a source's S/N is never counted in a band it was not
    observed in.

    Returns
    -------
    np.ndarray of bool, shape (n_sources,)
        True where the source was detected in >= ``min_bands`` bands.
    """
    detected = np.zeros((n_sources, len(_BANDS)), dtype=bool)

    for k, (_, params, band, _, half_stamp, src_idx) in enumerate(
        per_exposure_data
    ):
        _, sigma_img, positions, gen, _, _ = inf_exposures[k]
        sigma_img = np.asarray(sigma_img)
        snr = _source_snr(
            gen, positions, params, sigma_img, half_stamp,
            n_lambda=n_lambda, oversampling=oversampling,
        )
        above = snr > snr_threshold
        if np.any(above):
            detected[np.asarray(src_idx)[above], band - 1] = True

    return detected.sum(axis=1) >= min_bands


def plot_spectra_comparison(true_params, recovered_params, bright_mask,
                            model=None,
                            fname=os.path.join(PLOTS_DIR,
                                               "spectra_comparison.svg"),
                            max_sources=32, n_points=400):
    """Overlay the true and recovered spectrum SHAPES of the brightest sources.

    A scatter plot of true vs recovered shape parameters (``plot_comparison``)
    is only directly interpretable when those parameters have a physical
    meaning - ``log T`` for a blackbody.  For a flexible model, and especially
    for the randomly initialised neural network, they do not: what matters is
    whether the recovered *spectrum* matches the true one.  This plots
    ``shape(lambda)`` for the ``max_sources`` brightest sources (ranked by true
    amplitude, i.e. by ``f_lambda(LAMBDA_0)``) at their true (solid) and
    recovered (dashed) shape parameters, over the full SPHEREx wavelength
    range, and prints the median absolute log-shape error.

    Parameters
    ----------
    true_params, recovered_params : np.ndarray, shape (N, 1 + P)
    bright_mask : np.ndarray of bool, shape (N,)
    model : equinox.Module, optional
        Model used to turn shape parameters into spectra; defaults to the
        module-level :data:`SPECTRUM_MODEL`.
    fname : str
    max_sources : int
        Maximum number of sources to draw (keeps the figure readable).
    n_points : int
        Number of wavelengths in the plotted grid.
    """
    model = SPECTRUM_MODEL if model is None else model
    true_params = np.asarray(true_params)
    recovered_params = np.asarray(recovered_params)
    bright_mask = np.asarray(bright_mask)

    _, true_shape = split_source_params(true_params)
    _, rec_shape = split_source_params(recovered_params)

    bright_idx = np.where(bright_mask)[0]
    if bright_idx.size == 0:
        print("  No bright sources; skipping the spectra comparison plot.")
        return

    # "Brightest" proxy: the largest true amplitude (which IS f_lambda at
    # LAMBDA_0, thanks to the amplitude-first convention).
    order = bright_idx[np.argsort(true_params[bright_idx, 0])[::-1]]
    selected = order[:max_sources]

    lambdas = _wavelength_grid(n_points)
    lam_np = np.asarray(lambdas)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    errors = []
    for i in selected:
        # Dimensionless shapes (1 at LAMBDA_0) - the model itself returns an
        # unnormalised log-flux, so ask the helper for the physical shape.
        true_s = np.asarray(normalized_shape(
            model, lambdas, jnp.asarray(true_shape[i], jnp.float32)
        ))
        rec_s = np.asarray(normalized_shape(
            model, lambdas, jnp.asarray(rec_shape[i], jnp.float32)
        ))
        ax.plot(lam_np, true_s, color="C0", alpha=0.5, lw=1.0)
        ax.plot(lam_np, rec_s, color="C3", alpha=0.5, lw=1.0, ls="--")
        good = (np.isfinite(true_s) & np.isfinite(rec_s)
                & (true_s > 0) & (rec_s > 0))
        if np.any(good):
            errors.append(
                np.abs(np.log(rec_s[good]) - np.log(true_s[good]))
            )

    # Band boundaries, so it is obvious which parts of the spectrum each
    # exposure actually samples.
    for band in sorted(_BANDS):
        lo, hi, _r, _name = _BANDS[band]
        ax.axvline(lo, color="gray", lw=0.5, alpha=0.4)
        ax.text(0.5 * (lo + hi), 0.99, f"{band}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="gray")
    ax.axvline(_BANDS[max(_BANDS)][1], color="gray", lw=0.5, alpha=0.4)
    ax.axvline(LAMBDA_0, color="k", lw=0.7, alpha=0.4, ls=":")
    ax.text(LAMBDA_0, 0.02, f"LAMBDA_0", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=7, color="k")

    ax.set_yscale("log")
    ax.set_xlim(lam_np.min(), lam_np.max())
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel(f"Shape  (f_lambda / f_lambda(LAMBDA_0={LAMBDA_0} um))")
    ax.plot([], [], color="C0", lw=1.5, label="True")
    ax.plot([], [], color="C3", lw=1.5, ls="--", label="Recovered")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Spectra of the {len(selected)} brightest sources "
                 f"({_spectrum_model_name(model)})")
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

    if errors:
        err = np.concatenate(errors)
        print(f"  Median |Delta log shape| over bright sources: "
              f"{np.median(err):.4f} dex "
              f"(90th pct {np.percentile(err, 90):.4f})")


def _spectrum_model_name(model=None):
    """Short human-readable name of the active spectrum model."""
    model = SPECTRUM_MODEL if model is None else model
    if _is_blackbody(model):
        return "blackbody"
    return (f"neural net, P={model.n_params}, "
            f"{model.n_hidden_layers}x{model.hidden_size}")


def end_to_end_mock(use_lm=False, spectrum_model=None):
    """Run the full mock simulation, inference and diagnostics.

    Parameters
    ----------
    use_lm : bool
        Use Levenberg-Marquardt instead of SGD for the inference step.
    spectrum_model : equinox.Module, optional
        Spectrum-shape template to simulate with.  ``None`` builds the default
        (see :data:`SPECTRUM_KIND`).  The SAME instance is injected into every
        band's generator and is never handed to an optimiser, so its weights
        stay frozen.
    """
    if spectrum_model is None:
        spectrum_model = _build_spectrum_model()
    _set_spectrum_model(spectrum_model)

    print("=" * 60)
    print("SPHEREx Mock Exposure Simulation")
    print("=" * 60)

    # ---- Step 1: amplitude range + photon-count diagnostic ----------------
    print("\n--- Step 1: Amplitude range ---")
    _report_photon_counts(spectrum_model)

    # ---- Step 2: catalog --------------------------------------------------
    print("\n--- Step 2: Generating source catalog ---")
    rng = np.random.default_rng(42)
    skycoords, shape_params, log_amplitudes = _generate_catalog(rng)
    print(f"  Generated {N_SOURCES} sources")
    print(f"  {_describe_shape_params(shape_params, spectrum_model)}")
    print(f"  A range: [{np.exp(log_amplitudes).min():.4e}, "
          f"{np.exp(log_amplitudes).max():.4e}]")

    # ---- Step 3: exposures -------------------------------------------------
    print("\n--- Step 3: Generating exposure WCSs ---")
    exposures = _generate_exposures(rng)

    # ---- pre-filter: which sources are in any exposure? -----------------
    print("\n--- Pre-filtering catalog ---")
    in_any = np.zeros(N_SOURCES, dtype=bool)
    for _band, wcs, half_stamp in exposures:
        x_pix, y_pix = wcs.world_to_pixel(skycoords)
        in_any |= (
            (x_pix >= -half_stamp) & (x_pix < DETECTOR_PIXELS + half_stamp)
            & (y_pix >= -half_stamp) & (y_pix < DETECTOR_PIXELS + half_stamp)
        )
    skycoords_obs = skycoords[in_any]
    print(f"  {in_any.sum()} / {N_SOURCES} sources in >=1 exposure")

    # ---- sky-location plot -----------------------------------------------
    print("--- Sky-location plot ---")
    _plot_sky_locations(skycoords_obs, exposures,
                        os.path.join(PLOTS_DIR, 'sky_locations.svg'))

    # ---- Step 4: filter sources -------------------------------------------
    print("\n--- Step 4: Filtering sources per exposure ---")
    _, _, _, per_exposure_data = _filter_sources(
        skycoords, shape_params, log_amplitudes, exposures
    )

    # ---- Step 5 + 6: generate & save ---------------------------------------
    print("\n--- Steps 5 & 6: Generating and saving images ---")
    inf_exposures, noisy_vmins, noisy_vmaxs = _generate_and_save(
        per_exposure_data, spectrum_model
    )

    # ---- Step 7: inference -------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7: Parameter inference via SGD")
    print("=" * 60)

    # True log-params for ALL sources in the full catalog (NOT filtered by
    # in_any). This must stay in the SAME index space as ``src_idx``
    # (built in ``_filter_sources`` from indices into the full,
    # unfiltered catalog) - indexing with a filtered/compacted array here
    # would silently select the wrong source's parameters for most
    # sources.
    # Parameter layout is [log_amplitude, theta_0, ..., theta_{P-1}]: the
    # amplitude is ALWAYS the first column (see ``spherex.spectrum``).
    true_log_params = np.column_stack([
        log_amplitudes, shape_params
    ]).astype(np.float32)

    # Initial guess: amplitudes uniform within their prior range, shape
    # parameters from the model-specific initialisation (which is deliberately
    # NOT the same draw as the prior - see _draw_init_shape_params).
    rng_inf = np.random.default_rng(99)
    n_sources = true_log_params.shape[0]
    init_log_params = np.column_stack([
        rng_inf.uniform(
            np.log(AMPLITUDE_MIN), np.log(AMPLITUDE_MAX), n_sources
        ),
        _draw_init_shape_params(rng_inf, n_sources, spectrum_model),
    ]).astype(np.float32)

    # The spectrum model must stay FROZEN: only (log_params, log_backgrounds)
    # are ever handed to an optimiser, and the generator is a closed-over
    # constant, so no code path should be able to touch these weights.
    # Fingerprint them here and re-check after inference.
    model_fingerprint_before = _model_fingerprint(spectrum_model)

    # # Initial guess: perturb true values
    # rng_inf = np.random.default_rng(99)
    # init_log_params = true_log_params + 0.1 * rng_inf.standard_normal(
    #     true_log_params.shape
    # ).astype(np.float32)

    # ---- sanity check: loss at the TRUE parameters should be ~1 -----------
    # (chi^2 / pixel).  A value far from 1 indicates a bug in the forward
    # model or in how the noise / sigma was generated, rather than an
    # optimisation failure.
    true_log_backgrounds = np.log([
        _compute_background(band, spectrum_model)
        for _, _, band, _, _, _ in per_exposure_data
    ]).astype(np.float32)
    # NOTE: n_lambda must match the ``n_wavelength_samples`` used when
    # generating the images (N_LAMBDA, see ``_generate_and_save``) - using
    # a coarser wavelength grid here than in the forward simulation used
    # to introduce a systematic wavelength-integration mismatch that
    # inflated chi^2 even at the true parameters (was 8.8 with n_lambda=3
    # vs the correct ~1.0 with n_lambda=5, before switching to the
    # quantile-based integration - now much less sensitive to this).
    true_loss = compute_loss(
        inf_exposures, true_log_params, true_log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=2,
    )
    print(f"\nLoss (chi^2 / pixel) at TRUE parameters: {true_loss:.4f} "
          f"(should be ~1)")

    # Cross-check: what does the LM fitter see at the true parameters?
    # This uses the EXACT same residual function that optimistix internally
    # builds (band-grouped, padded, flattened), so any discrepancy vs
    # compute_loss above points to a mismatch between the two code paths.
    lm_true_loss = compute_lm_loss(
        inf_exposures, true_log_params, true_log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=2,
    )
    total_pixels = sum(np.asarray(m).size for m, *_ in inf_exposures)
    raw_optx = 0.5 * total_pixels * lm_true_loss
    print(f"LM loss (chi^2 / pixel) at TRUE parameters: {lm_true_loss:.4f} "
          f"(raw optimistix: {raw_optx:.1f})")
    if abs(lm_true_loss - true_loss) > 0.01:
        print(f"  ⚠ MISMATCH vs compute_loss ({true_loss:.4f}) — "
              f"difference {lm_true_loss - true_loss:.4f}")

    # ---- loss at the INITIAL GUESS, before optimisation ---------------------
    # (independent of whether --lm is set) - uses the same background
    # initialisation convention as infer_parameters/infer_parameters_lm
    # (median counts converted back to a rate via EXPOSURE_TIME).
    init_log_backgrounds = np.log([
        max(np.median(np.asarray(mean_img)) / EXPOSURE_TIME, 1e-6)
        for mean_img, _, _, _, _, _ in inf_exposures
    ]).astype(np.float32)
    init_loss = compute_loss(
        inf_exposures, init_log_params, init_log_backgrounds,
        n_lambda=N_LAMBDA, oversampling=2,
    )
    print(f"Loss (chi^2 / pixel) at INITIAL GUESS: {init_loss:.4f}")

    if use_lm:
        # ---- Phase 1: short SGD warmup (~128 steps) to get into the right
        # ballpark before LM takes over.  LM converges quadratically near
        # the optimum, but from a random initialisation the very first
        # Gauss-Newton step can wildly overshoot (exponentiated
        # parameterisation).  A few cheap SGD steps fix that.
        print("\n--- Phase 1: SGD warmup (128 steps) ---")
        sgd_params, sgd_backgrounds, losses, lrs = infer_parameters(
            inf_exposures,
            init_log_params,
            n_steps=128,
            learning_rate=2e-2,
            momentum=0.5,
            warmup_steps=8,
            n_lambda=N_LAMBDA,
            oversampling=2,
        )

        # ---- Phase 2: Levenberg-Marquardt, warm-started from SGD ------------
        print("\n--- Phase 2: Levenberg-Marquardt (optimistix) ---")
        rec_log_params, log_backgrounds, result, stats = infer_parameters_lm(
            inf_exposures,
            sgd_params,
            init_log_backgrounds=sgd_backgrounds,
            n_lambda=N_LAMBDA,
            oversampling=2,
            max_steps=64,
            initial_step_size=1e-6,
            cg_max_steps=100,
            max_step_size=1.0,
            verbose=True,
        )
        print(f"LM result: {result}")
        print(f"LM stats: {stats}")
        # losses/lrs already captured from SGD warmup (kept for plotting)
    else:
        rec_log_params, log_backgrounds, losses, lrs = infer_parameters(
            inf_exposures,
            init_log_params,
            n_steps=1024,
            learning_rate=1e-2,
            momentum=0.3,
            warmup_steps=16,
            n_lambda=N_LAMBDA,
            oversampling=2,
            precondition_rms=True,
            # Interleave direct amplitude least-squares solves with SGD
            amp_solve_every=16,
            amp_verbose=True,
        )

    # ---- loss at the RECOVERED parameters -----------------------------------
    final_loss = compute_loss(
        inf_exposures, np.asarray(rec_log_params),
        np.asarray(log_backgrounds),
        n_lambda=N_LAMBDA, oversampling=2,
    )
    print(f"\nLoss (chi^2 / pixel) at RECOVERED parameters: {final_loss:.4f}")

    # ---- spectrum model must be unchanged (frozen weights) ------------------
    # Freezing is automatic: only (log_params, log_backgrounds) are handed to
    # the optimiser and the generator is a closed-over constant.  Re-check the
    # fingerprint anyway, so a future refactor that accidentally makes the
    # generator a dynamic JAX argument (the only way the weights could turn
    # into traced values) is caught here rather than silently invalidating
    # the run.
    if _model_fingerprint(spectrum_model) == model_fingerprint_before:
        print("  Spectrum model weights unchanged (frozen) ✔")
    else:
        print("  ⚠ WARNING: spectrum model weights CHANGED during inference - "
              "they are supposed to be frozen!")

    # ---- Step 8: predicted + residual images --------------------------------
    print("\n--- Step 8: Generating predicted and residual images ---")
    for k, (noisy_img, sigma_img, pos, gen, half_stamp, src_idx) in enumerate(inf_exposures):
        # Find which per_exposure_data entry this corresponds to
        _, _, band, _, half_stamp, _ = per_exposure_data[k]

        pred = np.asarray(gen(
            pos,
            rec_log_params[np.asarray(src_idx)],
            postage_stamp_half_size=half_stamp,
            n_wavelength_samples=N_LAMBDA,
            oversampling=2,
        ))
        # pred/background are RATES (s^-1); scale both by EXPOSURE_TIME to
        # get expected counts, matching the data-generation convention.
        pred_bg = (pred + np.exp(float(log_backgrounds[k]))) * EXPOSURE_TIME

        vmin_n, vmax_n = noisy_vmins[k], noisy_vmaxs[k]

        # predicted image (same stretch as noisy)
        if np.any(~np.isfinite(pred_bg)):
            print(f"  WARNING: pred_bg has non-finite values, skipping PNG")
            continue
        pred_scaled = np.clip(
            (pred_bg - vmin_n) / (vmax_n - vmin_n) * 255.0, 0, 255
        ).astype(np.uint8)
        fname_pred = os.path.join(
            PLOTS_DIR, f"exposure_{k:03d}_band_{band}_pred.png"
        )
        Image.fromarray(pred_scaled.T, mode="L").save(fname_pred)

        # residual image (independent percentile stretch)
        resid = np.asarray(noisy_img) - pred_bg
        mask = noisy_img > np.median(noisy_img) + 3 * np.median(sigma_img)

        if np.any(mask):
            vr_min, vr_max = np.percentile(resid[mask], [PNG_PERCENTILE_LO, PNG_PERCENTILE_HI])
        else:
            vr_min, vr_max = np.percentile(resid, [PNG_PERCENTILE_LO, PNG_PERCENTILE_HI])

        if vr_max <= vr_min:
            vr_max = vr_min + 1e-30
        resid_scaled = np.clip(
            (resid - vr_min) / (vr_max - vr_min) * 255.0, 0, 255
        ).astype(np.uint8)
        fname_resid = os.path.join(
            PLOTS_DIR, f"exposure_{k:03d}_band_{band}_resid.png"
        )
        Image.fromarray(resid_scaled.T, mode="L").save(fname_resid)
        print(f"  Saved {fname_pred}, {fname_resid}")

        # residual score image
        resid_score = resid / sigma_img
        vmin_s, vmax_s = np.percentile(resid_score, [PNG_PERCENTILE_LO, PNG_PERCENTILE_HI])
        if vmax_s <= vmin_s:
            vmax_s = vmin_s + 1e-30
        resid_score_scaled = np.clip(
            (resid_score - vmin_s) / (vmax_s - vmin_s) * 255.0, 0, 255
        ).astype(np.uint8)
        fname_resid_score = os.path.join(
            PLOTS_DIR, f"exposure_{k:03d}_band_{band}_resid_score.png"
        )
        Image.fromarray(resid_score_scaled.T, mode="L").save(fname_resid_score)
        print(f"  Saved {fname_resid_score}")

    # ---- Step 9: diagnostic plots ------------------------------------------
    print("\n--- Step 9: Diagnostic plots ---")

    if losses:
        plot_loss_history(
            losses, lrs,
            os.path.join(PLOTS_DIR, "loss_history.svg")
        )
    else:
        print("  Skipping loss-history plot (not tracked by the LM solver).")

    # Bright sources: detected at S/N > SNR_THRESHOLD in >= MIN_BANDS distinct
    # bands.  This uses the TRUE parameters throughout: using the recovered
    # ones would make the label circular (a source whose amplitude estimate
    # diverges upward would be "detected" precisely because of the
    # convergence problem being diagnosed).  Counting unique bands rather
    # than exposures avoids promoting a source that was simply observed many
    # times in a single band.  See _source_snr / _compute_bright_mask.
    bright = _compute_bright_mask(
        per_exposure_data, inf_exposures, true_log_params.shape[0],
        n_lambda=N_LAMBDA, oversampling=2,
    )
    print(f"  {int(bright.sum())} / {bright.size} sources are bright "
          f"(S/N > {SNR_THRESHOLD} in >= {MIN_BANDS} bands)")

    plot_comparison(true_log_params, np.asarray(rec_log_params), bright,
                    os.path.join(PLOTS_DIR, "comparison.svg"),
                    shape_labels=_shape_param_labels(spectrum_model))

    # Shape-parameter recovery is only directly interpretable when the
    # parameters have a physical meaning (log T for a blackbody).  For a
    # flexible model - especially a random neural network - the meaningful
    # check is whether the recovered *spectrum* matches the true one, so the
    # two are overlaid for the brightest sources.
    plot_spectra_comparison(true_log_params, np.asarray(rec_log_params), bright,
                            spectrum_model,
                            os.path.join(PLOTS_DIR, "spectra_comparison.svg"))

    print("\nDone.")



# ---------------------------------------------------------------------------
# Benchmarking (not a formal test - for timing/perf investigation)
# ---------------------------------------------------------------------------

def _build_generators(bands, cls, model=None):
    """Build one generator of class ``cls`` per band in ``bands``.

    All bands share the SAME spectrum-model instance (required for a stateful
    model such as the neural network).
    """
    model = SPECTRUM_MODEL if model is None else model
    return {
        band: cls(band=band, psf_scale=PSF_SCALE,
                  image_width=DETECTOR_PIXELS, image_height=DETECTOR_PIXELS,
                  spectrum_model=model)
        for band in bands
    }


def _perturb_params(shape_params, log_amplitudes, rng, scale=0.05):
    """Return a fresh perturbation of the parameters (new draw each call),
    used to force re-computation (not just re-tracing) across benchmark
    repeats.

    ``shape_params`` is the ``(N, P)`` shape block (columns 1 onward of
    ``source_params``); it is perturbed independently for any ``P``.
    """
    shape_params = np.asarray(shape_params)
    log_amplitudes = np.asarray(log_amplitudes)
    d_shape = rng.normal(0.0, scale, size=shape_params.shape)
    d_A = rng.normal(0.0, scale, size=log_amplitudes.shape)
    return shape_params + d_shape, log_amplitudes + d_A


# ---------------------------------------------------------------------------
# n_lambda resolution comparison (not a formal test)
# ---------------------------------------------------------------------------

def compare_n_lambda(n_lambda_values=(1, 3, 15, 63), oversampling=2,
                     exposure_index=0,
                     spectrum_model=None,
                     fname=os.path.join(PLOTS_DIR, "n_lambda_comparison.png")):
    """Generate one exposure's image at several ``n_lambda`` (wavelength
    sample count) values and compare them.

    Reuses the catalog / exposure-generation helpers from ``end_to_end_mock()`` to
    build a realistic exposure, renders it with each value in
    ``n_lambda_values`` (an arbitrary number of values, at least 2), and
    plots:

    - All images (on the *same* colour scale, from the percentiles of
      the highest-resolution image, i.e. the last entry of
      ``n_lambda_values``).
    - The residual of every *other* setting against the highest
      resolution one (``n_lambda[i] - n_lambda[-1]`` for all ``i`` but
      the last), all on the same, zero-centred colour scale.

    This demonstrates that ``n_lambda`` (the number of wavelength samples
    used to numerically integrate the photon rate over each pixel's local
    filter bandpass) must be high enough that the forward model is a
    faithful representation of the "true" continuous integral - too few
    samples introduces a systematic bias, not just added noise, which
    inflates chi^2 even at the true source parameters.
    """
    if len(n_lambda_values) < 2:
        raise ValueError("n_lambda_values must have at least 2 entries")

    spectrum_model = _set_spectrum_model(
        _build_spectrum_model() if spectrum_model is None else spectrum_model
    )

    print("=" * 60)
    print("n_lambda resolution comparison")
    print("=" * 60)

    rng = np.random.default_rng(42)
    skycoords, shape_params, log_amplitudes = _generate_catalog(rng)
    exposures = _generate_exposures(rng)
    _, _, _, per_exposure_data = _filter_sources(
        skycoords, shape_params, log_amplitudes, exposures
    )

    positions_pix, params, band, wcs, half_stamp, src_idx = (
        per_exposure_data[exposure_index]
    )
    print(f"\nExposure {exposure_index}: Band {band}, "
          f"{params.shape[0]} sources, half_stamp={half_stamp}")

    gen = SpherexImageGenerator3(
        band=band, psf_scale=PSF_SCALE,
        image_width=DETECTOR_PIXELS, image_height=DETECTOR_PIXELS,
        spectrum_model=spectrum_model,
    )

    images = []
    for n_lambda in n_lambda_values:
        t0 = time.perf_counter()
        img = gen(
            positions_pix, params,
            postage_stamp_half_size=half_stamp,
            n_wavelength_samples=n_lambda,
            oversampling=oversampling,
        )
        img.block_until_ready()
        dt = time.perf_counter() - t0
        img_np = np.asarray(img)
        images.append(img_np)
        print(f"  n_lambda={n_lambda:3d}: generated in {dt:.2f}s, "
              f"total flux={img_np.sum():.4e} s^-1")

    # ---- residuals of every setting except the last against the last -------
    # (the last entry, i.e. the highest n_lambda, is treated as the
    # reference "ground truth").
    img_ref = images[-1]
    n_ref = n_lambda_values[-1]
    residuals = [img - img_ref for img in images[:-1]]
    rms_values = [float(np.sqrt(np.mean(r ** 2))) for r in residuals]
    for n_lambda, rms in zip(n_lambda_values[:-1], rms_values):
        print(f"Residual RMS (n_lambda={n_lambda} - n_lambda={n_ref}): "
              f"{rms:.4e}")

    # ---- shared colour scale for all images ---------------------------------
    vmin_img, vmax_img = np.percentile(
        img_ref, [PNG_PERCENTILE_LO, PNG_PERCENTILE_HI]
    )
    if vmax_img <= vmin_img:
        vmax_img = vmin_img + 1e-30

    # ---- shared, zero-centred colour scale for all residual images ---------
    resid_scale = max(
        max(np.percentile(np.abs(r), PNG_PERCENTILE_HI) for r in residuals),
        1e-30,
    )

    fig, axes = plt.subplots(
        2, len(n_lambda_values),
        figsize=(4 * len(n_lambda_values), 8)
    )

    for ax, img, n_lambda in zip(axes[0], images, n_lambda_values):
        im = ax.imshow(img.T, origin="lower", vmin=vmin_img, vmax=vmax_img,
                       cmap="viridis")
        ax.set_title(f"n_lambda = {n_lambda}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # One residual per non-reference n_lambda value, left-aligned in the
    # second row; any left-over column (there are len(n_lambda_values)
    # columns but only len(n_lambda_values) - 1 residuals) is left empty.
    axes[1, 0].axis("off")
    for i, (n_lambda, r, rms) in enumerate(
        zip(n_lambda_values[:-1], residuals, rms_values)
    ):
        ax = axes[1, i + 1]
        im = ax.imshow(r.T, origin="lower",
                       vmin=-resid_scale, vmax=resid_scale, cmap="RdBu_r")
        ax.set_title(f"n_lambda={n_lambda} - n_lambda={n_ref}  "
                     f"(RMS={rms:.2e})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[1, len(residuals) + 1:]:
        ax.axis("off")

    fig.suptitle(f"Band {band} exposure - effect of n_lambda "
                f"(wavelength samples) on the forward model")
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nSaved {fname}")


def run_benchmark(n_repeats=5, source_batch_sizes=(None, 8, 32),
                  spectrum_model=None):
    """Benchmark image generation (image.py vs image3.py) and inference
    (legacy per-exposure JIT vs per-band-grouped batched JIT).

    Reuses the catalog/exposure-generation helpers from ``end_to_end_mock()`` so the
    benchmark exercises realistic source counts / postage-stamp sizes.
    Prints a timing summary; not a pytest test.
    """
    print("=" * 60)
    print("SPHEREx Benchmark")
    print("=" * 60)

    spectrum_model = _set_spectrum_model(
        _build_spectrum_model() if spectrum_model is None else spectrum_model
    )

    rng = np.random.default_rng(42)
    skycoords, shape_params, log_amplitudes = _generate_catalog(rng)
    exposures = _generate_exposures(rng)
    _, _, _, per_exposure_data = _filter_sources(
        skycoords, shape_params, log_amplitudes, exposures
    )

    bands_present = sorted({band for _, _, band, _, _, _ in per_exposure_data})
    print(f"\nBands present: {bands_present}, "
          f"{len(per_exposure_data)} exposures, "
          f"{sum(p.shape[0] for p, *_ in per_exposure_data)} total source-obs")

    gens1 = _build_generators(bands_present, SpherexImageGenerator,
                              spectrum_model)
    gens3 = _build_generators(bands_present, SpherexImageGenerator3,
                              spectrum_model)

    # ---- Part 1: image-generation timing --------------------------------
    print("\n--- Image generation: image.py vs image3.py ---")
    perturb_rng = np.random.default_rng(7)

    results = {"image.py": [], "image3.py (batch=None)": []}
    for bs in source_batch_sizes:
        if bs is not None:
            results[f"image3.py (batch={bs})"] = []

    max_chi2 = 0.0

    for i, (positions_pix, params, band, wcs, half_stamp, src_idx) in enumerate(
        per_exposure_data
    ):
        gen1 = gens1[band]
        gen3 = gens3[band]

        # Fresh perturbation per repeat forces real recomputation.
        # source_params layout is [log_amplitude, theta_0, ..., theta_{P-1}].
        n_src = params.shape[0]
        base_A = np.asarray(params[:, 0])
        base_shape = np.asarray(params[:, 1:])

        ref_img = None
        for rep in range(n_repeats):
            p_shape, pA = _perturb_params(base_shape, base_A, perturb_rng)
            p = jnp.asarray(
                np.column_stack([pA, p_shape]), dtype=jnp.float32
            )

            t0 = time.perf_counter()
            img1 = gen1(positions_pix, p, postage_stamp_half_size=half_stamp,
                       n_wavelength_samples=N_LAMBDA, oversampling=2)
            img1.block_until_ready()
            dt1 = time.perf_counter() - t0
            results["image.py"].append(dt1)

            for bs in source_batch_sizes:
                key = ("image3.py (batch=None)" if bs is None
                       else f"image3.py (batch={bs})")
                t0 = time.perf_counter()
                img3 = gen3(positions_pix, p, postage_stamp_half_size=half_stamp,
                           n_wavelength_samples=N_LAMBDA, oversampling=2,
                           source_batch_size=bs)
                img3.block_until_ready()
                dt3 = time.perf_counter() - t0
                results[key].append(dt3)

            if ref_img is None:
                ref_img = np.asarray(img1)
                ref_img3 = np.asarray(img3)
                chi2 = np.mean((ref_img - ref_img3) ** 2)
                max_chi2 = max(max_chi2, float(chi2))

        print(f"  Exposure {i} (Band {band}, {n_src} sources, "
              f"half_stamp={half_stamp}): image.py first={results['image.py'][0]:.3f}s")

    print(f"\nMax MSE between image.py and image3.py outputs "
          f"(same params): {max_chi2:.3e}  (should be ~0)")

    print("\nTiming summary (mean +/- std over all exposures x repeats, "
          "excluding first call per exposure = compile):")
    for key, vals in results.items():
        vals = np.asarray(vals)
        # crude compile-vs-steady-state split: first value per exposure group
        n_per_exposure = n_repeats
        steady = vals.reshape(-1, n_per_exposure)[:, 1:].ravel() \
            if n_per_exposure > 1 else vals
        compiles = vals.reshape(-1, n_per_exposure)[:, 0]
        print(f"  {key:28s}: compile(first)={compiles.mean():.3f}s, "
              f"steady={steady.mean() * 1e3:.2f}+/-{steady.std() * 1e3:.2f} ms")

    # ---- Part 2: inference timing ----------------------------------------
    print("\n--- Inference: batched=False vs batched=True ---")
    inf_exposures, _, _ = _generate_and_save(per_exposure_data, spectrum_model)

    init_log_params = np.column_stack(
        [log_amplitudes, shape_params]
    ).astype(np.float32)
    init_log_params += 0.05 * np.random.default_rng(1).standard_normal(
        init_log_params.shape
    ).astype(np.float32)

    n_bench_steps = 5
    for batched in (False, True):
        t0 = time.perf_counter()
        infer_parameters(
            inf_exposures, init_log_params, n_steps=n_bench_steps,
            learning_rate=1e-3, warmup_steps=1, momentum=0.0,
            n_lambda=N_LAMBDA, oversampling=2, batched=batched,
        )
        dt = time.perf_counter() - t0
        print(f"  batched={batched!s:5s}: {n_bench_steps} steps in {dt:.2f}s "
              f"({dt / n_bench_steps:.3f} s/step, incl. compile)")

    print("\nBenchmark done.")


# ---------------------------------------------------------------------------
# Spectrum-model pre-flight check (not a formal test)
# ---------------------------------------------------------------------------

def nn_shape_check(model=None, n_samples=64, n_points=400, seed=NN_SEED + 2,
                   fname=os.path.join(PLOTS_DIR, "shape_check.svg")):
    """Pre-flight diagnostic for the active spectrum model.

    Answers the questions that decide whether an inference run with this model
    can be trusted at all, *before* waiting for that run:

    1. **Does the shape parameter actually change the in-band shape?**  If it
       does not, the parameters are unidentifiable in principle and any
       recovered values are meaningless.  Measured directly from the
       sensitivity matrix ``d log shape / d theta`` restricted to the sampled
       wavelengths: a near-zero singular value is a direction that leaves the
       spectrum unchanged.
    2. **Is the in-band shape O(1)?**  A random network whose spectra span many
       decades moves the amplitude range / background / detection S/N regime
       away from the blackbody run the amplitudes were chosen for.  Reported as
       the per-band log-shape dynamic range over ``theta`` drawn from the
       prior.

    Also plots ``shape(lambda; theta)`` over the full SPHEREx wavelength range
    and prints the implied photon counts for the directly specified amplitude
    range.
    """
    model = _set_spectrum_model(
        _build_spectrum_model() if model is None else model
    )

    print("=" * 60)
    print("Spectrum-model shape check")
    print("=" * 60)
    print(f"  Model: {_spectrum_model_name(model)}")

    lambdas = _wavelength_grid(n_points)
    lam_np = np.asarray(lambdas)
    rng = np.random.default_rng(seed)
    theta = _draw_shape_params(rng, n_samples, model)
    theta_j = jnp.asarray(theta, dtype=jnp.float32)

    # Dimensionless log-shape for every draw: the quantity the generators
    # exponentiate (see spherex.spectrum.normalized_log_shape).
    log_shapes = np.asarray(
        jax.vmap(lambda th: normalized_log_shape(model, lambdas, th))(theta_j)
    )
    shapes = np.exp(log_shapes)

    # ---- 1. per-band log-shape dynamic range ------------------------------
    print(f"\n--- In-band log-shape range over {n_samples} draws from the "
          f"prior ---")
    for band in sorted(_BANDS):
        lo, hi, _r, _name = _BANDS[band]
        inside = (lam_np >= lo) & (lam_np <= hi)
        if not np.any(inside):
            continue
        log_shape = log_shapes[:, inside]
        lo_med, hi_med = np.median(log_shape.min(axis=1)), \
            np.median(log_shape.max(axis=1))
        lo_w, hi_w = np.percentile(log_shape.min(axis=1), 5), \
            np.percentile(log_shape.max(axis=1), 95)
        print(f"  Band {band} [{lo:.2f}-{hi:.2f} um]: "
              f"[{lo_med:+.2f}, {hi_med:+.2f}] (median), "
              f"[{lo_w:+.2f}, {hi_w:+.2f}] (5-95%)")
    print("  (band 1 contains LAMBDA_0, so its in-band dynamic range is "
          "intrinsically the smallest)")

    # ---- 2. identifiability (sensitivity singular values) ------------------
    n_params = model.n_params
    if n_params > 0:
        print(f"\n--- Identifiability: singular values of d log shape / d "
              f"theta ({n_points} x {n_params}) ---")
        jac = jax.vmap(
            jax.jacfwd(lambda th: normalized_log_shape(model, lambdas, th))
        )(theta_j)
        jac = np.asarray(jac).reshape(-1, n_params)
        sv = np.linalg.svd(jac, compute_uv=False)
        rank = int(np.sum(sv > sv[0] * 1e-6))
        print("  " + np.array2string(sv, precision=4, suppress_small=False))
        print(f"  numerical rank (1e-6 threshold): {rank} / {n_params}"
              + (f",  condition number {sv[0]/sv[-1]:.3g}" if sv[-1] > 0 else ""))
        if rank < n_params:
            print("  ⚠ DEGENERATE: some shape-parameter directions do not "
                  "change the spectrum at all, so they cannot be recovered "
                  "from these bands no matter how long the fit runs.")

    # ---- 3. plot + implied counts -----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for th, curve in zip(theta, shapes):
        ax.plot(lam_np, curve, color="C0", alpha=0.35, lw=1.0)
    for band in sorted(_BANDS):
        lo, hi, _r, _name = _BANDS[band]
        ax.axvline(lo, color="gray", lw=0.5, alpha=0.4)
        ax.text(0.5 * (lo + hi), 0.99, f"{band}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="gray")
    ax.axvline(_BANDS[max(_BANDS)][1], color="gray", lw=0.5, alpha=0.4)
    ax.axvline(LAMBDA_0, color="k", lw=0.7, alpha=0.4, ls=":")
    ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlim(lam_np.min(), lam_np.max())
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Shape  (normalised to 1 at LAMBDA_0)")
    ax.set_title(f"{_spectrum_model_name(model)}: shape(lambda; theta) for "
                 f"{n_samples} draws from the prior")
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nSaved {fname}")

    _report_photon_counts(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true",
                       help="Run the image-gen / inference benchmark "
                            "instead of the full simulation.")
    parser.add_argument("--compare-n-lambda", action="store_true",
                       help="Compare images generated with n_lambda = "
                            "3, 5, 15 instead of the full simulation.")
    parser.add_argument("--lm", action="store_true",
                       help="Use Levenberg-Marquardt (optimistix) instead "
                            "of SGD for the inference step.")
    parser.add_argument("--spectrum", choices=("blackbody", "nn"),
                       default=SPECTRUM_KIND,
                       help="Spectral shape template to simulate with "
                            "(default: %(default)s).  The model's weights are "
                            "frozen; only the source parameters are inferred.")
    parser.add_argument("--nn-seed", type=int, default=NN_SEED,
                       help="Seed for the neural network weights "
                            "(default: %(default)s).")
    parser.add_argument("--nn-hidden", type=int, default=NN_HIDDEN_SIZE,
                       help="Hidden-layer width of the neural network "
                            "(default: %(default)s).")
    parser.add_argument("--nn-layers", type=int, default=NN_N_HIDDEN_LAYERS,
                       help="Number of hidden layers of the neural network "
                            "(default: %(default)s).")
    parser.add_argument("--nn-params", type=int, default=NN_N_PARAMS,
                       help="Number of shape parameters theta (neural "
                            "network only; default: %(default)s).")
    parser.add_argument("--nn-embeddings", type=int, default=NN_N_EMBEDDINGS,
                       help="Number of wavelength Fourier features (neural "
                            "network only; 0 feeds the raw wavelength; "
                            "default: %(default)s).")
    parser.add_argument("--nn-delta-ln-wavelength", type=float,
                       default=NN_DELTA_LN_WAVELENGTH,
                       help="Span of the log-wavelength Fourier embedding, "
                            "i.e. the range of ln(lambda) covered "
                            "(default: %(default)s).")
    parser.add_argument("--nn-shape-check", action="store_true",
                       help="Print/plot the spectrum-model shape and "
                            "identifiability diagnostic, then exit.")
    args = parser.parse_args()

    spectrum_model = _build_spectrum_model(
        kind=args.spectrum,
        n_params=args.nn_params,
        n_hidden_layers=args.nn_layers,
        hidden_size=args.nn_hidden,
        seed=args.nn_seed,
        n_embeddings=args.nn_embeddings,
        delta_ln_wavelength=args.nn_delta_ln_wavelength,
    )
    _set_spectrum_model(spectrum_model)

    if args.nn_shape_check:
        nn_shape_check(spectrum_model)
    elif args.benchmark:
        run_benchmark(spectrum_model=spectrum_model)
    elif args.compare_n_lambda:
        compare_n_lambda(spectrum_model=spectrum_model)
    else:
        end_to_end_mock(use_lm=args.lm, spectrum_model=spectrum_model)
