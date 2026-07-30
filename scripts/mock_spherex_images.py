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
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from PIL import Image

from spherex import SpherexImageGenerator, SpherexImageGenerator3, BlackbodySpectrum
from spherex.config import _BANDS
from spherex.constants import HC_JAX, TEMPERATURE_UNIT
from spherex.plotting_utils import HistEqNormalize
from jax.scipy.integrate import trapezoid
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import infer_parameters, compute_loss, plot_loss_history, plot_comparison
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SOURCES = 256
N_EXPOSURES = 8
EXPOSURE_TIME = 15.0           # seconds  (approximate SPHEREx frame time)
PIXEL_SCALE = 6.2              # arcsec
DETECTOR_PIXELS = 2048 // 8

# Catalog covers a ~5° × 5° patch on the celestial equator
CATALOG_CENTER = SkyCoord(ra=180.0, dec=75.0, unit="deg", frame="icrs")
CATALOG_RADIUS = 3.5 / 8          # degrees  (spherical cap radius)

# Power-law index for source amplitudes
AMPLITUDE_ALPHA = 1.5          # P(A) ∝ A^{-alpha}

# PNG percentile clipping
PNG_PERCENTILE_LO = 0.2
PNG_PERCENTILE_HI = 99.8

# Output directory
PLOTS_DIR = "plots"

# PSF scaling (for making wavelength-dependent effects more visible)
PSF_SCALE = 4.0

# Background: peak pixel rate of A_min star × this factor
BACKGROUND_FACTOR = 2.0  # >1 makes faintest stars below background

# ---------------------------------------------------------------------------
# Step 1: Estimate amplitude limits
# ---------------------------------------------------------------------------

def _estimate_amplitude_limits():
    """Compute A_min, A_max so that a T=3000 K blackbody produces
    ~100 to ~5e6 detected photons per exposure over a representative
    SPHEREx band (Band 3, 1.63–2.41 um)."""
    lam_min, lam_max, _R, _name = _BANDS[3]

    # Dense wavelength grid for integration
    n_lam = 500
    lambdas = jnp.linspace(lam_min, lam_max, n_lam)
    dlam = lambdas[1] - lambdas[0]

    # Blackbody at T = 3 kK, amplitude = 1
    bb = BlackbodySpectrum()
    T_ref = (3000 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value   # 3.0
    params = jnp.array([np.log(T_ref), 0.0])  # log-space
    f_lam = bb(lambdas, params)                        # W / (m^2 um)

    # Photon rate per unit amplitude  [s^{-1}]
    #   aperture * ∫ (lambda / hc) * f_lambda(lambda) d_lambda
    aperture = jnp.pi * (0.10) ** 2                    # m^2
    integrand = (lambdas / HC_JAX) * f_lam             # s^{-1} m^{-2} um^{-1}  ???  actually photons/s per (m^2 um)
    rate_per_amp = float(aperture * trapezoid(integrand, lambdas))  # s^{-1}

    photons_per_sec_per_amp = rate_per_amp
    photons_per_exp_per_amp = photons_per_sec_per_amp * EXPOSURE_TIME

    target_bright = 5.0e6
    target_faint = 100.0

    A_max = target_bright / photons_per_exp_per_amp
    A_min = target_faint / photons_per_exp_per_amp

    print(f"Photons / s per unit amplitude (T=3000K): {photons_per_sec_per_amp:.4e}")
    print(f"Photons / exposure per unit amplitude:     {photons_per_exp_per_amp:.4e}")
    print(f"A_min = {A_min:.4e}  (~{target_faint:.0f} photons)")
    print(f"A_max = {A_max:.4e}  (~{target_bright:.0f} photons)")

    return A_min, A_max


# ---------------------------------------------------------------------------
# Step 2: Generate source catalog
# ---------------------------------------------------------------------------

def _generate_catalog(A_min, A_max, rng, ra_center=None, dec_center=None,
                       radius_deg=None):
    """Return (skycoords, temperatures, amplitudes).

    Sources are drawn uniformly from the surface of a sphere within a
    spherical cap of radius ``radius_deg`` centred on (``ra_center``,
    ``dec_center``).
    """
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

    # log-Temperatures  U(log(3), log(8))  [log(kK)]
    log_temperatures = np.log(rng.uniform(
        (3000 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value,
        (8000 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value,
        N_SOURCES,
    ))

    # Amplitudes: truncated power-law  P(A) ∝ A^{-alpha}
    alpha = AMPLITUDE_ALPHA
    u_vals = rng.uniform(0.0, 1.0, N_SOURCES)
    # Inverse CDF for A^{-alpha} on [A_min, A_max]
    #   CDF(A) = (A^{1-alpha} - A_min^{1-alpha}) / (A_max^{1-alpha} - A_min^{1-alpha})
    #   => A = [A_min^{1-alpha} + u * (A_max^{1-alpha} - A_min^{1-alpha})]^{1/(1-alpha)}
    exp = 1.0 - alpha
    A_min_exp = A_min ** exp
    A_max_exp = A_max ** exp
    amplitudes = (A_min_exp + u_vals * (A_max_exp - A_min_exp)) ** (1.0 / exp)
    log_amplitudes = np.log(amplitudes)

    return skycoords, log_temperatures, log_amplitudes


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
        half_stamp = int(np.ceil(5.0 * fwhm / PIXEL_SCALE))

        exposures.append((band, wcs, half_stamp))
        print(f"  Exposure {len(exposures)-1}: Band {band}, "
              f"center=({center.ra.deg:.3f}°, {center.dec.deg:.3f}°), "
              f"half_stamp={half_stamp}")

    print(f'Included bands: {np.unique([d[0] for d in exposures])}')

    return exposures


# ---------------------------------------------------------------------------
# Step 4: Filter sources per exposure
# ---------------------------------------------------------------------------

def _filter_sources(skycoords, log_temperatures, log_amplitudes, exposures):
    """Filter to sources observed in at least one exposure.

    Returns
    -------
    skycoords, log_temperatures, log_amplitudes  (filtered, log-space)
    per_exposure_data : list of (positions_pix, params_array, band, wcs, half_stamp, src_idx)
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

        params = jnp.stack(
            [
                jnp.asarray(log_temperatures[in_this], dtype=jnp.float32),
                jnp.asarray(log_amplitudes[in_this], dtype=jnp.float32),
            ],
            axis=-1,
        )

        per_exposure_data.append(
            (positions_pix, params, band, wcs, half_stamp,
             np.where(in_this)[0].astype(np.int32))
        )

    # Filter catalog to sources in at least one exposure
    n_kept = np.sum(in_any)
    print(f"\nKeeping {n_kept} / {n_total} sources (in ≥1 exposure)")

    skycoords_filt = skycoords[in_any]
    log_temperatures_filt = log_temperatures[in_any]
    log_amplitudes_filt = log_amplitudes[in_any]

    return skycoords_filt, log_temperatures_filt, log_amplitudes_filt, per_exposure_data


# ---------------------------------------------------------------------------
# Step 5 & 6: Generate images and save
# ---------------------------------------------------------------------------


def _compute_background(band, A_min):
    """Estimate background level so A_min stars are slightly below it.

    Computes the approximate peak pixel rate for a T=3000 K blackbody
    with amplitude ``A_min`` at the band-centre wavelength, then scales
    by ``BACKGROUND_FACTOR``.
    """
    lam_min, lam_max, R, _name = _BANDS[band]
    lam_mid = 0.5 * (lam_min + lam_max)

    # PSF at band centre
    fwhm = PSF_SCALE * 6.0 * (lam_mid / 1.0)               # arcsec
    sigma_psf = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # arcsec
    psf_peak = 1.0 / (2.0 * np.pi * sigma_psf**2)  # arcsec^-2

    # Blackbody flux at band centre
    bb = BlackbodySpectrum()
    lam_arr = jnp.array([lam_mid])
    params = jnp.array([np.log((3000 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value), np.log(A_min)])
    f_lam = float(bb(lam_arr, params)[0])       # W / (m^2 um)

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


def _generate_and_save(per_exposure_data, A_min):
    """For each exposure, generate an image (timed) and save as PNG.

    Returns
    -------
    inference_exposures : list of (noisy_img, sigma_img, wcs, gen_module)
    noisy_vmins, noisy_vmaxs : per-exposure stretch bounds
    true_params_all : (N, 2) log-params of all sources used anywhere
    """
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
            )
        gen = band_generators[band]

        t0 = time.perf_counter()
        img = gen(
            positions_pix,
            params,
            postage_stamp_half_size=half_stamp,
            n_wavelength_samples=5,
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
        bg = _compute_background(band, A_min)
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


def main():
    print("=" * 60)
    print("SPHEREx Mock Exposure Simulation")
    print("=" * 60)

    # ---- Step 1: amplitude limits -----------------------------------------
    print("\n--- Step 1: Estimating amplitude limits ---")
    A_min, A_max = _estimate_amplitude_limits()

    # ---- Step 2: catalog --------------------------------------------------
    print("\n--- Step 2: Generating source catalog ---")
    rng = np.random.default_rng(42)
    skycoords, log_temperatures, log_amplitudes = _generate_catalog(A_min, A_max, rng)
    print(f"  Generated {N_SOURCES} sources")
    print(f"  T range: [{np.exp(log_temperatures).min():.1f}, "
          f"{np.exp(log_temperatures).max():.1f}] kK")
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
        skycoords, log_temperatures, log_amplitudes, exposures
    )

    # ---- Step 5 + 6: generate & save ---------------------------------------
    print("\n--- Steps 5 & 6: Generating and saving images ---")
    inf_exposures, noisy_vmins, noisy_vmaxs = _generate_and_save(
        per_exposure_data, A_min
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
    true_log_params = np.column_stack([
        log_temperatures, log_amplitudes
    ]).astype(np.float32)

    # Initial guess: random values within the prior bounds
    rng_inf = np.random.default_rng(99)
    ln_teff_bounds = np.log(([3000.,8000.] * u.K).to(u.Unit(TEMPERATURE_UNIT)).value)
    init_log_params = np.column_stack([
        rng_inf.uniform(
            ln_teff_bounds[0],
            ln_teff_bounds[1],
            true_log_params.shape[0],
        ),
        rng_inf.uniform(
            np.log(A_min), np.log(A_max), true_log_params.shape[0]
        ),
    ]).astype(np.float32)

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
        _compute_background(band, A_min) for _, _, band, _, _, _ in per_exposure_data
    ]).astype(np.float32)
    # NOTE: n_lambda must match the ``n_wavelength_samples`` used when
    # generating the images (5, see ``_generate_and_save``) - using a
    # coarser wavelength grid here than in the forward simulation
    # introduces a systematic wavelength-integration mismatch that
    # inflates chi^2 even at the true parameters (was 8.8 with
    # n_lambda=3 vs the correct ~1.0 with n_lambda=5).
    true_loss = compute_loss(
        inf_exposures, true_log_params, true_log_backgrounds,
        n_lambda=5, oversampling=2,
    )
    print(f"\nLoss (chi^2 / pixel) at TRUE parameters: {true_loss:.4f} "
          f"(should be ~1)")

    return 0

    rec_log_params, log_backgrounds, losses, lrs = infer_parameters(
        inf_exposures,
        init_log_params,
        n_steps=128,
        learning_rate=1e-2,
        momentum=0.5,
        warmup_steps=16,
        n_lambda=5,
        oversampling=2,
    )

    # ---- Step 8: predicted + residual images --------------------------------
    print("\n--- Step 8: Generating predicted and residual images ---")
    for k, (noisy_img, sigma_img, pos, gen, half_stamp, src_idx) in enumerate(inf_exposures):
        # Find which per_exposure_data entry this corresponds to
        _, _, band, _, half_stamp, _ = per_exposure_data[k]

        pred = np.asarray(gen(
            pos,
            rec_log_params[np.asarray(src_idx)],
            postage_stamp_half_size=half_stamp,
            n_wavelength_samples=5,
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

    # ---- Step 9: diagnostic plots ------------------------------------------
    print("\n--- Step 9: Diagnostic plots ---")

    plot_loss_history(losses, lrs,
                      os.path.join(PLOTS_DIR, "loss_history.svg"))

    # Bright sources: peak > 5× background in ≥3 bands (simplified)
    # NOTE: indexed in the FULL catalog space (same as true_log_params /
    # rec_log_params), not the in_any-filtered skycoords_obs subset.
    n_above = np.zeros(true_log_params.shape[0], dtype=int)
    for _noisy, _sigma, _pos, gen, _hs, _src_idx in inf_exposures:
        rec_A = np.exp(np.asarray(rec_log_params)[:, 1])
        bg_level = np.exp(np.asarray(log_backgrounds))[0]  # avg
        n_above += (rec_A > 5.0 * bg_level).astype(int)
    bright = n_above >= 3

    plot_comparison(true_log_params, np.asarray(rec_log_params), bright,
                    os.path.join(PLOTS_DIR, "comparison.svg"))

    print("\nDone.")



# ---------------------------------------------------------------------------
# Benchmarking (not a formal test - for timing/perf investigation)
# ---------------------------------------------------------------------------

def _build_generators(bands, cls):
    """Build one generator of class ``cls`` per band in ``bands``."""
    return {
        band: cls(band=band, psf_scale=PSF_SCALE,
                  image_width=DETECTOR_PIXELS, image_height=DETECTOR_PIXELS)
        for band in bands
    }


def _perturb_params(log_temperatures, log_amplitudes, rng, scale=0.05):
    """Return a fresh perturbation of the log-parameters (new draw each
    call), used to force re-computation (not just re-tracing) across
    benchmark repeats."""
    dT = rng.normal(0.0, scale, size=log_temperatures.shape)
    dA = rng.normal(0.0, scale, size=log_amplitudes.shape)
    return log_temperatures + dT, log_amplitudes + dA


# ---------------------------------------------------------------------------
# n_lambda resolution comparison (not a formal test)
# ---------------------------------------------------------------------------

def compare_n_lambda(n_lambda_values=(1, 3, 5, 15), oversampling=2,
                     exposure_index=0,
                     fname=os.path.join(PLOTS_DIR, "n_lambda_comparison.png")):
    """Generate one exposure's image at several ``n_lambda`` (wavelength
    sample count) values and compare them.

    Reuses the catalog / exposure-generation helpers from ``main()`` to
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

    print("=" * 60)
    print("n_lambda resolution comparison")
    print("=" * 60)

    rng = np.random.default_rng(42)
    A_min, A_max = _estimate_amplitude_limits()
    skycoords, log_temperatures, log_amplitudes = _generate_catalog(
        A_min, A_max, rng
    )
    exposures = _generate_exposures(rng)
    _, _, _, per_exposure_data = _filter_sources(
        skycoords, log_temperatures, log_amplitudes, exposures
    )

    positions_pix, params, band, wcs, half_stamp, src_idx = (
        per_exposure_data[exposure_index]
    )
    print(f"\nExposure {exposure_index}: Band {band}, "
          f"{params.shape[0]} sources, half_stamp={half_stamp}")

    gen = SpherexImageGenerator3(
        band=band, psf_scale=PSF_SCALE,
        image_width=DETECTOR_PIXELS, image_height=DETECTOR_PIXELS,
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


def run_benchmark(n_repeats=5, source_batch_sizes=(None, 8, 32)):
    """Benchmark image generation (image.py vs image3.py) and inference
    (legacy per-exposure JIT vs per-band-grouped batched JIT).

    Reuses the catalog/exposure-generation helpers from ``main()`` so the
    benchmark exercises realistic source counts / postage-stamp sizes.
    Prints a timing summary; not a pytest test.
    """
    print("=" * 60)
    print("SPHEREx Benchmark")
    print("=" * 60)

    rng = np.random.default_rng(42)
    A_min, A_max = _estimate_amplitude_limits()
    skycoords, log_temperatures, log_amplitudes = _generate_catalog(
        A_min, A_max, rng
    )
    exposures = _generate_exposures(rng)
    _, _, _, per_exposure_data = _filter_sources(
        skycoords, log_temperatures, log_amplitudes, exposures
    )

    bands_present = sorted({band for _, _, band, _, _, _ in per_exposure_data})
    print(f"\nBands present: {bands_present}, "
          f"{len(per_exposure_data)} exposures, "
          f"{sum(p.shape[0] for p, *_ in per_exposure_data)} total source-obs")

    gens1 = _build_generators(bands_present, SpherexImageGenerator)
    gens3 = _build_generators(bands_present, SpherexImageGenerator3)

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
        n_src = params.shape[0]
        base_T = np.asarray(params[:, 0])
        base_A = np.asarray(params[:, 1])

        ref_img = None
        for rep in range(n_repeats):
            dT, dA = _perturb_params(base_T, base_A, perturb_rng)
            p = jnp.stack(
                [jnp.asarray(dT, jnp.float32), jnp.asarray(dA, jnp.float32)],
                axis=-1,
            )

            t0 = time.perf_counter()
            img1 = gen1(positions_pix, p, postage_stamp_half_size=half_stamp,
                       n_wavelength_samples=3, oversampling=2)
            img1.block_until_ready()
            dt1 = time.perf_counter() - t0
            results["image.py"].append(dt1)

            for bs in source_batch_sizes:
                key = ("image3.py (batch=None)" if bs is None
                       else f"image3.py (batch={bs})")
                t0 = time.perf_counter()
                img3 = gen3(positions_pix, p, postage_stamp_half_size=half_stamp,
                           n_wavelength_samples=3, oversampling=2,
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
    inf_exposures, _, _ = _generate_and_save(per_exposure_data, A_min)

    init_log_params = np.column_stack(
        [log_temperatures, log_amplitudes]
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
            n_lambda=3, oversampling=2, batched=batched,
        )
        dt = time.perf_counter() - t0
        print(f"  batched={batched!s:5s}: {n_bench_steps} steps in {dt:.2f}s "
              f"({dt / n_bench_steps:.3f} s/step, incl. compile)")

    print("\nBenchmark done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true",
                       help="Run the image-gen / inference benchmark "
                            "instead of the full simulation.")
    parser.add_argument("--compare-n-lambda", action="store_true",
                       help="Compare images generated with n_lambda = "
                            "3, 5, 15 instead of the full simulation.")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.compare_n_lambda:
        compare_n_lambda()
    else:
        main()
