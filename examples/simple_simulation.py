#!/usr/bin/env python3
"""Simple end-to-end simulation of a SPHEREx exposure.

Generates a small detector image with a few point sources and displays it.
Also demonstrates taking gradients of the total flux with respect to source
parameters via ``jax.grad``.
"""

import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from spherex import (
    BlackbodySpectrum,
    GaussianPSF,
    GaussianFilterTransmission,
    ImageGenerator,
)
from spherex.constants import TEMPERATURE_UNIT


def main():
    # ------------------------------------------------------------------
    # 1. Instantiate the physical models
    # ------------------------------------------------------------------
    psf = GaussianPSF(
        fwhm_ref=6.0,          # arcsec  (SPHEREx PSF ~6 arcsec in the optical)
        wavelength_ref=1.0,     # um
    )

    transmission = GaussianFilterTransmission(
        lambda_intercept=0.75,  # um  (blue end of the LVF at y=0)
        lambda_slope=0.05,       # um / arcsec
        width=0.02,              # um  (narrow bandpass)
    )

    spectrum_model = BlackbodySpectrum()

    # SPHEREx aperture is ~20 cm diameter -> ~0.031 m²
    aperture = jnp.pi * (0.10) ** 2   # m²

    generator = ImageGenerator(
        psf=psf,
        transmission=transmission,
        spectrum_model=spectrum_model,
        aperture=aperture,
    )

    # ------------------------------------------------------------------
    # 2. Define sources
    # ------------------------------------------------------------------
    source_positions = jnp.array([
        [20.0, 25.0],     # arcsec
        [40.0, 25.0],
        [30.0, 40.0],
    ])

    # Each row: [log-amplitude, log-temperature (kK)].
    # The spectrum model is a *shape* template normalised to f_lambda = 1 at
    # LAMBDA_0 (1 um); the amplitude (f_lambda at LAMBDA_0, W m^-2 um^-1) is
    # carried separately as the FIRST parameter (see ``spherex.spectrum``).
    source_params = jnp.log(jnp.array([
        [1e-13, (5778 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value],   # Sun-like
        [2e-13, (3500 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value],   # Cool star
        [5e-14, (10000 * u.K).to(u.Unit(TEMPERATURE_UNIT)).value],  # Hot star
    ]))

    # ------------------------------------------------------------------
    # 3. Generate the image
    # ------------------------------------------------------------------
    W, H = 60, 60              # pixels
    pixel_scale = 1.0          # arcsec / pixel
    stamp_half = 15            # enough to capture PSF wings

    print("Generating image ...")
    image = generator(
        source_positions,
        source_params,
        image_width=W,
        image_height=H,
        pixel_scale=pixel_scale,
        postage_stamp_half_size=stamp_half,
        n_wavelength_samples=61,
        oversampling=3,
    )
    image_np = np.asarray(image)   # bring back to CPU / NumPy

    print(f"Image shape : {image_np.shape}")
    print(f"Total flux  : {image_np.sum():.4e} s^-1")

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    extent = [0, W * pixel_scale, 0, H * pixel_scale]
    im = ax.imshow(
        image_np,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="inferno",
    )
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("SPHEREx simulated exposure")

    # Mark source positions
    ax.scatter(
        np.asarray(source_positions[:, 0]),
        np.asarray(source_positions[:, 1]),
        marker="o", facecolors="none", edgecolors="cyan",
        s=80, linewidths=1.5, label="Sources",
    )

    ax.legend()
    plt.colorbar(im, ax=ax, label="Photon rate [s$^{-1}$]")
    fig.tight_layout()
    plt.savefig("simple_simulation.png", dpi=150)
    print("Saved simple_simulation.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Demonstrate gradient computation
    # ------------------------------------------------------------------
    print("\n--- Gradient demo ---")

    def total_flux(T1):
        """Total image flux as a function of the first source's temperature."""
        p = source_params.at[0, 1].set(T1)   # column 1 is log-temperature
        return jnp.sum(
            generator(
                source_positions, p,
                W, H, pixel_scale, stamp_half,
                n_wavelength_samples=41,
                oversampling=2,
            )
        )

    T0 = float(source_params[0, 1])
    grad = jax.grad(total_flux)(T0)
    print(f"Temperature of source 0    : {T0:.0f} K")
    print(f"d(total flux) / d(T)       : {grad:.4e} s^-1 K^-1")


if __name__ == "__main__":
    main()
