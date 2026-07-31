# SPHEREx PSF Photometry Pipeline — Explanatory Supplement

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Mathematical Setup](#2-mathematical-setup)
3. [Physical Model Objects](#3-physical-model-objects)
4. [Forward-Modeling an Image](#4-forward-modeling-an-image)
5. [Parameter Inference](#5-parameter-inference)
6. [Mock Data Pipeline](#6-mock-data-pipeline)
7. [Implementation Gotchas & Lessons Learned](#7-implementation-gotchas--lessons-learned)
8. [File Map](#8-file-map)

---

## 1. Design Philosophy

The pipeline is built around three principles:

**Modularity.** Every physical ingredient — the source spectrum, the PSF, the filter transmission, the telescope aperture — is a separate `equinox.Module` with a well-defined `__call__` signature. The image generator composes them by calling each one, never assuming a specific functional form (Gaussian PSF, blackbody spectrum, etc.). Swapping in a new model (e.g., a pixelized PSF or a spline transmission profile) requires changing only that one module. This is enforced by an **interface contract**: every transmission model must expose `quantile(q, omega_p)` and `total_transmission(omega_p)` in addition to `__call__`. See `spherex/transmission.py` for the contract documentation.

**Mathematical correctness for precision results.** The pipeline is designed for scientific use where systematic errors at the sub-percent level matter:

- **Quantile-based wavelength integration** (`_one_subpixel_rate` in `spherex/image.py`): instead of a fixed `linspace` + trapezoid rule (which mostly samples the near-zero tails of a narrow bandpass), the code draws wavelength samples at evenly spaced quantiles of the transmission profile's normalised CDF via `transmission.quantile(q, omega_p)`. This concentrates samples where the transmission actually has support. At `N_LAMBDA = 1` the RMS residual against a 63-sample reference is only `2.3e-5` — far more accurate than naive quadrature at the same sample count.
- **Float32-safe gradients** (`BlackbodySpectrum.__call__` in `spherex/spectrum.py`): the Planck function gradient `d/dT` is reformulated to avoid `(K_B·T)²` underflow at `T ~ 5 kK` in float32. An `inv_T_scale` pre-factor is computed from `HC/(λ·K_B)` so the gradient becomes `-inv_T_scale / T²`, neither factor underflowing.
- **Loss normalization** (`scripts/inference.py`): the loss is `chi² / n_pixels`, summed over *all* exposures simultaneously divided by the *global* pixel count — not a mean of per-exposure means (which would up-weight small exposures). This makes the loss directly interpretable: ~1 at convergence, ≫1 indicates poor fit, ≪1 indicates overfitting or sigma miscalibration.

**Consistent physical units throughout.** The codebase uses a fixed set of base units:

| Quantity | Unit | Variable suffix |
|---|---|---|
| Temperature | kiloKelvin (kK) | `_T`, `temperature` |
| Wavelength | micron (µm) | `_lam`, `wavelength`, `lambda` |
| Angular position | arcsec | `omega_p`, `omega_s` |
| Flux density | W m⁻² µm⁻¹ | `f_lam` |
| Photon rate | s⁻¹ | `rate`, `Ndot` |
| PSF value | arcsec⁻² | — |

All constants (`HC`, `KB`, etc.) are derived from `astropy.constants` and converted to these units in `spherex/constants.py`. JAX-traceable copies (`HC_JAX`, `KB_JAX`, etc.) are provided alongside plain-Python floats. The transmission wavelength gradient is in µm/arcsec (not µm/pixel), decoupling the physical model from the detector pixel scale.

---

## 2. Mathematical Setup

### 2.1 Source spectrum

A source at position $(x_s, y_s)$ on the sky (denoted $\Omega_s = (x_s, y_s)$ in arcsec) has a spectral flux density

$$f_\lambda(\lambda; T, A) = \frac{A}{\text{sr}} \cdot B_\lambda(\lambda, T)$$

where $B_\lambda$ is the Planck function and $A$ absorbs the solid-angle factor. The free parameters are $\log_{10}(T/\text{kK})$ and $\log_{10}(A / \text{W m}^{-2} \text{µm}^{-1})$, stored in log-space to enforce positivity and improve optimizer conditioning.

### 2.2 PSF

The telescope PSF is a circular Gaussian whose FWHM scales linearly with wavelength:

$$\text{FWHM}(\lambda) = \text{fwhm}_\text{ref} \cdot \frac{\lambda}{\lambda_\text{ref}}$$

with $\text{fwhm}_\text{ref} = 6.0''$ at $\lambda_\text{ref} = 1.0$ µm (configurable via `PSF_SCALE`). The PSF is normalized to integrate to 1 over $\mathbb{R}^2$ in arcsec²:

$$\text{PSF}(\Omega_p | \Omega_s, \lambda) = \frac{1}{2\pi\sigma(\lambda)^2} \exp\!\left(-\frac{|\Omega_p - \Omega_s|^2}{2\sigma(\lambda)^2}\right)$$

where $\sigma(\lambda) = \text{FWHM}(\lambda) / (2\sqrt{2\ln 2})$.

### 2.3 Filter transmission (Linear Variable Filter)

SPHEREx uses a linear-variable filter (LVF): the central wavelength varies linearly with detector y-position:

$$\lambda_c(y_p) = \lambda_\text{intercept} + \lambda_\text{slope} \cdot y_p \quad [\text{µm}]$$

The transmission profile at detector position $\Omega_p = (x_p, y_p)$ is a Gaussian in wavelength:

$$T(\lambda | \Omega_p) = \exp\!\left(-\frac{(\lambda - \lambda_c(y_p))^2}{2\,\text{width}^2}\right)$$

where $\text{width} = \lambda_\text{mid} / (2\sqrt{2\ln 2} \cdot R)$ and $R = \lambda / \Delta\lambda$ is the spectral resolution.

The transmission module also exposes:
- `quantile(q, omega_p)`: inverse CDF — `λ_c + width·√2·erfinv(2q-1)` — used for importance sampling
- `total_transmission(omega_p)`: `∫ T(λ|Ω_p) dλ = √(2π)·width` — the normalization constant

### 2.4 Photon detection rate (the integral)

The photon detection rate in one sub-pixel at position $\Omega_p$ from a source at $\Omega_s$ is:

$$\dot{N}(\Omega_p | \Omega_s) = A_\text{tel} \cdot (\Delta p)^2 \cdot \int_0^\infty \frac{\lambda}{hc}\, f_\lambda(\lambda; T, A)\, \text{PSF}(\Omega_p | \Omega_s, \lambda)\, T(\lambda | \Omega_p)\, d\lambda$$

where $A_\text{tel} = \pi(0.10\text{ m})^2$ and $\Delta p = 6.2''$ is the pixel scale.

**Wavelength integration method** (quantile importance sampling):

Define $p(\lambda | \Omega_p) = T(\lambda | \Omega_p) / N(\Omega_p)$ where $N(\Omega_p) = \int T \, d\lambda$. Then:

$$\dot{N} = A_\text{tel} (\Delta p)^2 \cdot N(\Omega_p) \cdot \mathbb{E}_{\lambda \sim p}\!\left[\frac{\lambda}{hc} f_\lambda(\lambda)\, \text{PSF}(\Omega_p | \Omega_s, \lambda)\right]$$

Draw $K$ samples at evenly spaced quantiles: $\lambda_k = \text{quantile}((k+0.5)/K, \Omega_p)$ for $k = 0,\ldots,K-1$. Then:

$$\dot{N} \approx A_\text{tel} (\Delta p)^2 \cdot N(\Omega_p) \cdot \frac{1}{K} \sum_{k=0}^{K-1} \frac{\lambda_k}{hc} f_\lambda(\lambda_k)\, \text{PSF}(\Omega_p | \Omega_s, \lambda_k)$$

Crucially, the integrand does **not** include $T(\lambda)$ — the transmission profile is absorbed into the sampling distribution $p(\lambda)$. This is the key to the method's accuracy at low $K$: all $K$ samples land where the transmission has support, rather than wasting most of them in the tails.

### 2.5 Postage stamps and image compositing

For each source, only pixels within a `half_stamp` radius are computed (typically 5× the PSF FWHM). The stamp is computed at $K^2$ sub-pixels per pixel (oversampling), the sub-pixel rates are averaged per pixel, and the result is scatter-added into the full image. ImageGenerator3 parallelizes this across sources via `jax.lax.map` with configurable batch size.

### 2.6 Chi-squared objective

For inference, the observed image (in counts) at pixel $i$ is $d_i$. The model prediction is $m_i(\theta, b) = (r_i(\theta) + b) \cdot t_\text{exp}$, where $r_i$ is the predicted rate (s⁻¹), $b$ is the background rate (s⁻¹), and $t_\text{exp} = 15$ s. The noise is $\sigma_i = \sqrt{m_i^\text{true} + 1^2}$ (Poisson variance of the true counts plus a 1-count floor). The loss is:

$$\chi^2_\nu = \frac{1}{N_\text{pix}} \sum_i \frac{(d_i - m_i)^2}{\sigma_i^2}$$

When fit with SGD, this is the objective directly. For Levenberg-Marquardt, the residual vector is $r_i = (d_i - m_i)/\sigma_i$ and `optimistix` minimizes $\frac{1}{2}\sum_i r_i^2 = \frac{1}{2}\chi^2$. The reported per-step loss is $\frac{1}{2}\chi^2$ (not $\chi^2_\nu$).

---

## 3. Physical Model Objects

All models are `equinox.Module` subclasses in `spherex/`. They store hyperparameters as attributes and accept per-source/call parameters explicitly, enabling JAX autodiff through them.

### 3.1 `BlackbodySpectrum` (`spherex/spectrum.py`)

- **Attributes**: `n_params = 2` (log-T, log-A)
- **`__call__(wavelength, params)`**: returns $f_\lambda$ in W m⁻² µm⁻¹
- **Implementation detail**: uses `inv_T_scale = HC/(λ·KB)` to avoid float32 underflow in the Planck gradient

### 3.2 `GaussianPSF` (`spherex/psf.py`)

- **Attributes**: `fwhm_ref` (arcsec), `wavelength_ref` (µm)
- **`__call__(omega_p, omega_s, wavelength)`**: returns PSF value in arcsec⁻²
- **Normalization**: integrates to 1 over ℝ² in arcsec² (2-D Gaussian normalization)

### 3.3 `GaussianFilterTransmission` (`spherex/transmission.py`)

- **Attributes**: `lambda_intercept` (µm), `lambda_slope` (µm/arcsec), `width` (µm)
- **`__call__(wavelength, omega_p)`**: returns transmission ∈ [0,1]
- **`central_wavelength(omega_p)`**: returns λ_c(y_p)
- **`quantile(q, omega_p)`**: inverse CDF via `erfinv`
- **`total_transmission(omega_p)`**: returns √(2π)·width

**Interface contract**: any replacement transmission model must implement `quantile` and `total_transmission`. These are used by the wavelength integrator and NOT optional.

### 3.4 Image generators (`spherex/image.py`, `spherex/image3.py`)

- `ImageGenerator`: iterates over sources via `jax.lax.scan`, computing one postage stamp at a time
- `ImageGenerator3`: uses `jax.lax.map(..., batch_size=...)` for chunked parallel processing, then a single vectorized scatter-add — faster but functionally identical

Both accept the same `__call__` signature: `(source_positions, source_params, image_width, image_height, pixel_scale, postage_stamp_half_size, n_wavelength_samples, oversampling)`.

### 3.5 Pre-configured generators (`spherex/config.py`)

- `SpherexImageGenerator` and `SpherexImageGenerator3` pre-configure an `ImageGenerator`/`ImageGenerator3` for a specific SPHEREx band (1–6)
- Constructor accepts `band`, `psf_scale`, `lambda_slope_scale`, and detector geometry
- `_build_band(band, psf_scale, lambda_slope_scale)` creates the PSF, transmission, and spectrum model from the band table in `_BANDS`
- Bands: 1 (0.75–1.11 µm), 2 (1.11–1.63), 3 (1.63–2.41), 4 (2.41–3.55), 5 (3.55–4.88), 6 (4.88–5.19)
- `lambda_slope_scale` compensates the wavelength gradient when the detector is downsampled (set to `DOWNSAMPLE` in mock scripts)

---

## 4. Forward-Modeling an Image

The call chain for generating one detector image:

```
SpherexImageGenerator3.__call__()
  └─ ImageGenerator3.__call__()
       └─ jax.lax.map(_one_stamp, sources, batch_size=...)
            └─ _one_stamp() per source
                 └─ jax.vmap(_one_subpixel_rate) over K² sub-pixels
                      └─ _one_subpixel_rate() per sub-pixel
                           ├─ transmission.quantile(q, omega_p) → K_λ wavelengths
                           ├─ spectrum_model(wavelengths, params) → f_λ
                           ├─ psf(omega_p, omega_s, wavelengths) → PSF vals
                           └─ jnp.mean((λ/HC) * f_λ * PSF_val) * norm
       └─ scatter-add stamps into full image
```

### Key details

- **Per-pixel oversampling**: each pixel is split into $K \times K$ sub-pixels. The detector-plane positions $\Omega_p$ are in arcsec (pixel index × `pixel_scale` + sub-pixel offset). The source position $\Omega_s$ is also in arcsec (pixel position × `pixel_scale`).
- **Valid mask**: sub-pixels outside the detector are clipped and masked to zero, preventing out-of-bounds scatter-add.
- **Aperture and pixel-scale factors**: the factor `aperture * pixel_scale²` is applied after averaging sub-pixels, at the `_one_stamp` level.

---

## 5. Parameter Inference

Inference is in `scripts/inference.py`. Two optimizers are available.

### 5.1 Shared infrastructure

**Exposure grouping**: exposures sharing the same band (and thus the same `gen_module` and PSF wavelength) are grouped together. Within a group, source arrays are padded to a common size and processed by a single `jax.lax.scan`. This reduces the traced program size compared to an unrolled loop over all exposures.

**Loss and residual functions**:
- `_per_exposure_sumsq`: returns Σ(residual²) for one exposure
- `_per_exposure_diff`: returns the **flat residual vector** `(data - model)/sigma` for one exposure — used by LM
- `_group_sumsq` / `_group_diffs`: scan over exposures in one band group
- `_make_loss_fn`: builds a `jax.jit`-compiled `(log_params, log_backgrounds) → chi²/n_pixels` function
- `_make_residual_fn`: builds `params → flat residual vector` for LM (not jit-compiled; `optimistix` handles this internally)

**Background initialization**: `log(median(image) / EXPOSURE_TIME)` per exposure. This is an estimate of the background rate (s⁻¹) assuming background dominates the median pixel.

### 5.2 SGD (`infer_parameters`)

- Optimizer: `clip(1.0) → scale_by_rms → sgd(momentum, warmup_cosine_decay_schedule)`
- Learning rate: linear warmup from 0 to peak, then cosine decay to 0
- The entire training step (loss + grad + update) is compiled into a single `jax.jit` via `_make_train_step`
- Returns `(log_params, log_backgrounds, losses, lrs)`

### 5.3 Levenberg-Marquardt (`infer_parameters_lm`)

- Uses `optimistix.least_squares` with a custom solver subclassing `optx.AbstractGaussNewton`
- **Inner linear solve**: `lineax.Normal(lineax.CG(rtol, atol, max_steps))` — matrix-free conjugate gradient on the damped normal equations $(J^T J + \lambda I)\Delta = -J^T r$. The Jacobian is never materialized; CG uses `jax.linearize` for Jacobian-vector products.
- **Custom trust region**: `_InitStepTrustRegion(optx.ClassicalTrustRegion)` with:
  - `initial_step_size` (upstream hardcodes to 1.0, too aggressive for log-parameterized models)
  - `max_step_size` (prevents the trust region from growing unboundedly; without it, the damping λ ~ 1/r² vanishes after ~5 accepted steps and CG hangs on a near-singular $J^T J$)
  - `step()` override that clamps `new_step_size` via `jnp.minimum`
- **CG tolerances**: `cg_rtol=1e-2`, `cg_atol=1e-4` (looser than defaults; LM only needs a direction, not an exact linear solve)
- **`cg_max_steps`**: hard cap on CG iterations; prevents hangs
- **`init_log_backgrounds`**: optional parameter for warm-starting from SGD

### 5.4 Hybrid SGD+LM (`mock_spherex_images.py --lm`)

Phase 1: short SGD warmup (50–128 steps) to get into the right ballpark.
Phase 2: LM warm-started from SGD-refined params and backgrounds, with `initial_step_size=1e-6` (very conservative).

This prevents the first Gauss-Newton step from overshooting when starting from a random initialization, which is a fundamental problem with exponentiated parameterizations.

### 5.5 Diagnostic: `compute_lm_loss`

Evaluates the **exact same** residual function that LM uses internally (`_make_residual_fn`), returning `sum(r²)/n_pixels`. This can be checked against `compute_loss` (which uses `_per_exposure_sumsq`) to verify consistency. Both should agree to machine precision.

---

## 6. Mock Data Pipeline

`scripts/mock_spherex_images.py` implements a full end-to-end simulation:

### 6.1 Configuration

Key globals at the top of the file:

| Parameter | Default | Meaning |
|---|---|---|
| `DOWNSAMPLE` | 4 | Detector binning factor (2048 → 512 pixels) |
| `PSF_SCALE` | 1.0 | Multiplier on PSF FWHM |
| `N_SOURCES` | 1024 | Number of catalog sources |
| `N_EXPOSURES` | 32 | Number of random detector pointings |
| `N_LAMBDA` | 1 | Wavelength integration samples (quantile-based) |
| `HALF_STAMP_FLOOR` | 3 | Minimum postage stamp radius (pixels) |
| `BACKGROUND_FACTOR` | 2.0 | Background = FACTOR × peak rate of faintest star |

### 6.2 Pipeline steps

1. **Estimate amplitude limits** (`_estimate_amplitude_limits`): computes the amplitude range such that a T=3000K blackbody produces ~100 to ~5×10⁶ detected photons per exposure over Band 3. Uses `trapezoid` integration over a dense 500-point wavelength grid.

2. **Generate catalog** (`_generate_catalog`): sources uniformly distributed on the sphere within a spherical cap, with log-uniform temperatures in [3000, 8000] K and power-law amplitudes P(A) ∝ A^{-1.5}.

3. **Generate exposures** (`_generate_exposures`): random detector pointings with random bands (1–6). Computes `half_stamp = max(ceil(5×FWHM/pixel_scale), HALF_STAMP_FLOOR)`.

4. **Filter sources** (`_filter_sources`): for each exposure, select sources within `half_stamp` of the detector edges. Builds per-exposure `(positions_pix, params, band, wcs, half_stamp, src_idx)` tuples where `src_idx` maps into the **full** (unfiltered) catalog.

5. **Generate images** (`_generate_and_save`):
   - Creates band-generator instances with `lambda_slope_scale=DOWNSAMPLE`
   - Generates noiseless rate images via `gen(positions, params, ...)`
   - Adds background (constant per band) and multiplies by `EXPOSURE_TIME` to get counts
   - Adds Gaussian noise: `σ = sqrt(true_counts + noise_floor²)` with `noise_floor = 1.0`
   - Saves true/noisy/predicted/residual PNGs

6. **Inference** (see §5)

7. **Diagnostic plots**: loss history (asinh y-scale), true-vs-recovered scatter, predicted/residual images

### 6.3 Wavelength gradient scaling with downsampling

When `DOWNSAMPLE > 1`, the detector has fewer pixels covering the same physical area. The WCS pixel scale remains 6.2"/pixel, so the physical detector extent shrinks to `(2048/DOWNSAMPLE) × 6.2"`. Without correction, the wavelength range across the detector is compressed by `DOWNSAMPLE`. The fix: `lambda_slope_scale = DOWNSAMPLE` in the generator constructor, which multiplies the µm/arcsec slope so the wavelength range across the (smaller) detector matches the full band.

---

## 7. Implementation Gotchas & Lessons Learned

### 7.1 Levenberg-Marquardt with log-parameterized models

`optimistix.LevenbergMarquardt` hardcodes `initial_step_size=1.0`. For a model where the actual quantity is `exp(log_param)`, a step of 1.0 in log-space corresponds to a factor of `e ≈ 2.7` in the physical quantity — enormous. The fix is our custom `_InitStepTrustRegion` subclass. Start with `1e-6` to `1e-4` for safety.

### 7.2 Trust-region blowup and CG hangs

Without a `max_step_size` cap, the trust region grows as `step_size × 3.5^n` on accepted steps. After ~5 accepted steps, the damping λ ~ 1/r² effectively vanishes, and CG tries to solve an unregularized $J^T J$ which may be near-singular — it spins forever. Fix: `max_step_size` clamp in `_InitStepTrustRegion.step()`.

### 7.3 CG tolerances

Tight CG tolerances (`rtol=1e-4`) can require hundreds of iterations per LM step, each costing one JVP (~0.5s). Looser tolerances (`rtol=1e-2`) are sufficient because LM only needs a direction, not an exact linear solve. Combine with `cg_max_steps` as a safety net.

### 7.4 Do NOT `jax.jit` the full residual function

The residual function involves `lax.scan`, `lax.map`, `vmap`, and Python for-loops over band groups. Wrapping the entire `_fn` in `jax.jit` causes JAX to trace a massive computation graph that takes many minutes to compile (or hangs entirely). `optimistix` internally handles JIT compilation of the step function; the residual function itself should remain a plain Python callable.

### 7.5 `jax.linearize` is the real cost per LM step

Each LM step calls `jax.linearize(fn, y)` once to produce the JVP function used by CG. This traces the entire forward model (~2s for 128 sources × 8 exposures). The resulting JVP function is then called ~N_CG times at ~0.5s each. The outer `jax.linearize` trace is amortized over the CG iterations within one step.

### 7.6 Index-space correctness for source parameters

`per_exposure_data[i][5]` (`src_idx`) contains indices into the **full** (unfiltered) catalog. When building `true_log_params` or `init_log_params`, you must use the full catalog, not the filtered one. Using the filtered catalog causes a silent index-space mismatch where the optimizer fits the wrong source's parameters.

### 7.7 EXPOSURE_TIME scaling must be consistent

Both the source prediction AND the background must be multiplied by `EXPOSURE_TIME` when converting from rates (s⁻¹) to expected counts. An earlier bug only scaled the background, which systematically under-counted the source flux relative to the background.

### 7.8 Background initialization

The LM fitter initializes `log_backgrounds` from `log(median(image) / EXPOSURE_TIME)` when `init_log_backgrounds=None`. When warm-starting from SGD, pass the SGD-optimized backgrounds explicitly via `init_log_backgrounds`.

### 7.9 Sigma floor and chi² calibration

The noise model uses `σ = sqrt(true_counts + noise_floor²)` with `noise_floor = 1.0`. An earlier version used `σ = sqrt(noisy_counts + 1)` which inflated sigma for faint pixels (adding 1 to a background of ~0.01 counts/pixel → σ ≈ 1 instead of σ ≈ 0.1), suppressing chi² below 1.0 at the true parameters.

### 7.10 Float32 safety in the Planck gradient

The naive gradient `d/dT [exp(hc/(λ·k_B·T)) - 1]⁻¹` involves `(k_B·T)²` in float32, which underflows to zero at astrophysical temperatures (T ~ 5 kK → (k_B·T)² ~ 4.8×10⁻³⁹). The fix pre-computes `inv_T_scale = HC/(λ·K_B)` so the gradient becomes `-inv_T_scale / T²`, neither factor underflowing.

### 7.11 Quantile integration and the `T(λ)` factor

When using quantile-based importance sampling, the integrand is `(λ/hc)·f_λ·PSF_val` — NOT including `T(λ)`. The transmission profile is absorbed into the sampling distribution. Including `T(λ)` in the integrand would double-count it.

### 7.12 Memory safety on CPU

Full-scale computations (hundreds of sources × many exposures) can exhaust memory on CPU-only machines. Always test correctness at tiny scale (~10 sources, 2 exposures, 32×32 detector) before scaling up.

---

## 8. File Map

| File | Purpose |
|---|---|
| `spherex/constants.py` | Physical constants in codebase-native units (kJ, µm, arcsec) |
| `spherex/spectrum.py` | `BlackbodySpectrum` — source spectrum template |
| `spherex/psf.py` | `GaussianPSF` — wavelength-dependent Gaussian PSF |
| `spherex/transmission.py` | `GaussianFilterTransmission` — LVF transmission with quantile interface |
| `spherex/image.py` | `_one_subpixel_rate` (quantile integration), `ImageGenerator` (scan-based) |
| `spherex/image3.py` | `ImageGenerator3` (chunked-batch, uses `_one_subpixel_rate` from `image.py`) |
| `spherex/config.py` | `SpherexImageGenerator`, `SpherexImageGenerator3` — pre-configured per-band generators, `_build_band`, band table |
| `spherex/__init__.py` | Public API exports |
| `scripts/inference.py` | `infer_parameters` (SGD), `infer_parameters_lm` (LM), `compute_loss`, `compute_lm_loss`, `plot_loss_history`, `plot_comparison`, residual functions, custom LM solver |
| `scripts/mock_spherex_images.py` | End-to-end mock: catalog, exposures, image generation, hybrid SGD+LM inference, diagnostics, benchmarks |
| `tests/test_image3.py` | Numerical tests verifying ImageGenerator3 ≡ ImageGenerator |
| `pyproject.toml` | Dependencies: `jax`, `equinox`, `optimistix`, `lineax`, `numpy` |
| `/memories/jax_optimization_lessons.md` | Persistent notes on LM trust-region tuning and memory safety |
