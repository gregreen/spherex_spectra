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

$$f_\lambda(\lambda; T, A) = A \cdot \hat{B}_\lambda(\lambda, T), \qquad
\hat{B}_\lambda(\lambda, T) = \frac{B_\lambda(\lambda, T)}{B_\lambda(\lambda_0, T)}$$

where $B_\lambda$ is the Planck function and $\lambda_0$ is a single **global** reference wavelength (`LAMBDA_0 = 1.0 µm` in `spherex/spectrum.py`) shared by every band — so the modelled spectrum stays smooth in amplitude across the whole wavelength range. The spectrum model returns only the dimensionless *shape* $\hat{B}_\lambda$, which equals exactly 1 at $\lambda_0$.

**Amplitude is not part of the spectrum model.** It is carried as a separate per-source parameter, always the **first** column of `source_params` (`LOG_AMPLITUDE_INDEX = 0`). The image generator applies it as $\exp(\texttt{source\_params}[:, 0])$ and passes only `source_params[:, 1:]` to the spectrum model. Because $A = f_\lambda(\lambda_0)$ is a physical flux density, $\log A$ has a direct interpretation. All parameters are stored in log-space to enforce positivity and improve optimizer conditioning.

`BlackbodySpectrum` therefore has a single free parameter, $\log(T/\text{kK})$. Crucially, the forward model is **exactly linear in the amplitude** $a = \exp(\log A)$ — a source's contribution is just $a$ times its unit-amplitude postage stamp — which is what makes the direct amplitude least-squares solve of §5.5 possible.

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

- **Attributes**: `lambda_0` (reference wavelength, defaulting to the global `LAMBDA_0 = 1.0 µm`), `n_params = 1` (log-T only — the amplitude lives *outside* the model)
- **`__call__(wavelength, shape_params)`**: returns the dimensionless shape $\hat{B}_\lambda = B_\lambda(\lambda,T)/B_\lambda(\lambda_0,T)$, which is exactly 1 at $\lambda_0$. Physical flux density is $\exp(\log A)\cdot\hat{B}_\lambda$.
- **Layout helpers**: `LOG_AMPLITUDE_INDEX`, `split_source_params`, `join_source_params` define the `[log_amplitude, shape…]` convention in one place
- **Implementation detail**: uses `inv_T_scale = HC/(λ·KB)` to avoid float32 underflow in the Planck gradient

### 3.2 `NeuralNetSpectrum` (`spherex/spectrum.py`)

A flexible alternative to the blackbody. A small MLP maps `(wavelength, shape_params)` to a log-flux and the shape is the exponential of the difference between that log-flux and its value at `LAMBDA_0`:

$$\hat{B}_\lambda(\lambda) = \exp\big(\mathrm{NN}(\lambda, \theta) - \mathrm{NN}(\lambda_0, \theta)\big)$$

which is *exactly* 1 at $\lambda_0$ for **any** $\theta$. Keeping the blackbody's normalisation convention is what makes the amplitude (first column of `source_params`) remain $f_\lambda(\lambda_0)$, and it is why **the entire amplitude pipeline works unchanged**: the least-squares solve only requires the model to be linear in amplitude and the shape to be independent of it, both of which hold (verified numerically — the solve recovers amplitudes to ~2×10⁻² in log on a synthetic problem, driving $\chi^2/\text{pixel}$ from ~2×10⁴ to ~1.1).

- **Attributes**: `layers` (list of `eqx.nn.Linear`), `n_params` (number of *shape* parameters), `n_hidden_layers`, `hidden_size`
- **`__init__(n_params, n_hidden_layers, hidden_size, *, key)`**: `key` is **required** (`eqx.nn.Linear` needs a PRNG key); one key per layer is derived with `jax.random.split`. All fields are declared as annotations, which is mandatory for `equinox.Module` (pytree) behaviour.
- **Batching contract** (identical to `BlackbodySpectrum`): `__call__(wavelength (N_λ,), shape_params (P,)) -> (N_λ,)`, i.e. **one source, many wavelengths**. The image generators already supply the other axes — `jax.vmap` over sub-pixels and `lax.scan`/`lax.map` over sources — and call `spectrum_model(lambdas, shape_params)` once per source per sub-pixel, so the model must not try to batch them itself.
- **Internals**: the MLP is evaluated one wavelength at a time and vectorised with `jax.vmap`. An MLP layer expects the feature axis last, so `concatenate([λ, θ])` is only well defined for a *scalar* λ (a wavelength *vector* would be read as extra features). To batch over sources as well, compose a second vmap at the call site: `jax.vmap(model, in_axes=(None, 0))(wavelengths, shape_params_batch)`.
- **Cost caution**: the shape is evaluated at every wavelength sample of every sub-pixel of every source, and an MLP costs orders of magnitude more per point than the closed-form blackbody — this slows the forward model *and* each interleaved amplitude solve (which rebuilds the postage stamps). Keep `hidden_size` modest. Each call also evaluates the network once at `LAMBDA_0` to normalise, so with `n_wavelength_samples = 1` the MLP runs twice per point.
- The raw network output is unbounded, so `exp` of a large log-ratio can overflow; clip the exponent if the shape parameters are free to wander far.

### 3.3 `GaussianPSF` (`spherex/psf.py`)

- **Attributes**: `fwhm_ref` (arcsec), `wavelength_ref` (µm)
- **`__call__(omega_p, omega_s, wavelength)`**: returns PSF value in arcsec⁻²
- **Normalization**: integrates to 1 over ℝ² in arcsec² (2-D Gaussian normalization)

### 3.4 `GaussianFilterTransmission` (`spherex/transmission.py`)

- **Attributes**: `lambda_intercept` (µm), `lambda_slope` (µm/arcsec), `width` (µm)
- **`__call__(wavelength, omega_p)`**: returns transmission ∈ [0,1]
- **`central_wavelength(omega_p)`**: returns λ_c(y_p)
- **`quantile(q, omega_p)`**: inverse CDF via `erfinv`
- **`total_transmission(omega_p)`**: returns √(2π)·width

**Interface contract**: any replacement transmission model must implement `quantile` and `total_transmission`. These are used by the wavelength integrator and NOT optional.

### 3.5 Image generators (`spherex/image.py`, `spherex/image3.py`)

- `ImageGenerator`: iterates over sources via `jax.lax.scan`, computing one postage stamp at a time
- `ImageGenerator3`: uses `jax.lax.map(..., batch_size=...)` for chunked parallel processing, then a single vectorized scatter-add — faster but functionally identical
 `source_params` has shape `(S, 1 + P)`: column 0 is the log-amplitude, and the remaining columns are passed to the spectrum model as its shape parameters. `ImageGenerator3` additionally exposes `source_stamps(...)`, returning the per-source postage stamps (before scatter-add) so callers can build per-source quadratic forms such as the amplitude-solve preconditioner diagonal.
Both accept the same `__call__` signature: `(source_positions, source_params, image_width, image_height, pixel_scale, postage_stamp_half_size, n_wavelength_samples, oversampling)`.

### 3.6 Pre-configured generators (`spherex/config.py`)

- `SpherexImageGenerator` and `SpherexImageGenerator3` pre-configure an `ImageGenerator`/`ImageGenerator3` for a specific SPHEREx band (1–6)
- Constructor accepts `band`, `psf_scale`, `lambda_slope_scale`, `spectrum_model`, and detector geometry
- `_build_band(band, psf_scale, lambda_slope_scale, spectrum_model=None)` creates the PSF, transmission, and spectrum model from the band table in `_BANDS`
- **`spectrum_model=None`** (the default) builds a fresh `BlackbodySpectrum`, so every pre-existing caller behaves exactly as before. Pass an explicit instance to substitute a different shape template (e.g. `NeuralNetSpectrum`). This is the *only* library change needed to swap the spectrum model — the generators, PSF, transmission and the whole amplitude-solve / inference stack are model-agnostic.
- **Pass the SAME instance to every band.** A stateful model (a neural network) would otherwise give each band a *different* spectrum, since each band builds its own generator. Stateless models are immune, but sharing one instance is correct in both cases.
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

Inference is in `scripts/inference.py`. Two optimizers are available (SGD and Levenberg–Marquardt), plus a **direct amplitude least-squares solve** (§5.5) that exploits the exact linearity of the model in the source amplitude and is interleaved with SGD by default.

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
- Learning rate: linear warmup from 0 to peak, then cosine decay to 0 (default peak `learning_rate=1e-2`, `warmup_steps=50`, `momentum=0.5`)
- The entire training step (loss + grad + update) is compiled into a single `jax.jit` via `_make_train_step`
- The amplitude least-squares solve is interleaved every `amp_solve_every` SGD steps (default **16**; cf. §5.6)
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

### 5.5 Direct amplitude least-squares solve + SGD interlace (`solve_log_amplitudes`)

Because the forward model is exactly **linear** in the per-source amplitude $a = \exp(\log A)$ (see §2.1) — a source contributes $a$ times its fixed unit-amplitude postage stamp — finding the optimal amplitude of every source, *holding the spectral shapes and backgrounds fixed*, is a linear weighted least-squares problem:

$$\min_a \; \Big\| \frac{d - t_\text{exp}(G a + b)}{\sigma} \Big\|^2$$

where $G$ is the (sparse, per-source postage-stamp) design operator and $y = (d - t_\text{exp}b)/\sigma$. Writing the residual as $f(a) = y - Ma$ (affine in $a$), `solve_log_amplitudes` forms the normal equations

$$(M^\top M)\, a = M^\top y .$$

**`M` is built directly from the unit-amplitude postage stamps**, not by autodiff: `gen_module.source_stamps` is called once per band group (vectorized over exposures with `jax.vmap`) to produce each source's stamp plus its pixel indices, after which every CG iteration is a cheap scatter/gather (`_amplitude_forward` / `_amplitude_adjoint`) rather than a forward-model evaluation. Note the adjoint indexes the *spatial* axes, so the scatter/gather must be `vmap`-ed over the exposure axis (indexing a stacked `(E, H, W)` array directly would silently index the wrong axes).

**Everything above is compiled into a single `jax.jit`** by `_make_amplitude_solver`: the stamps, the diagonal, the right-hand side, and a hand-rolled `jax.lax.while_loop` conjugate gradient (`_cg_solve`) — no Python-level iteration and no `lineax` re-dispatch. This matters enormously. Profiling at 128 sources × 32 exposures showed the **arithmetic was only ~0.1 s while the wall clock was ~2.2 s**: ~91% of the cost was JAX *tracing* the stamp program again on every call (the solver was plain Python, so each invocation rebuilt the same jaxpr from scratch). With the solver built once and reused — which `infer_parameters` does, and which is why `solve_log_amplitudes` documents that repeated callers should use `_make_amplitude_solver` directly — the solve takes **0.116 s** (18.8× faster), with identical results.

- **All sources are solved jointly** across the whole exposure stack, so overlaps/blending and the multi-exposure constraint (one amplitude per source) are handled exactly.
- **Jacobi scaling (always on, not optional).** The normal matrix is extremely ill-scaled: because $M$ is expressed in *absolute* amplitude units, $\mathrm{diag}(M^\top M)$ spans ~65 orders of magnitude in the mock. The solver applies a symmetric change of variables $a = Dc$ with $D = \mathrm{diag}(M^\top M)^{-1/2}$ (so the scaled normal matrix has unit diagonal). This is a *correctness* requirement, not a speed option — in the unscaled variables the normal matrix spans ~12 decades and float32 CG cannot converge — so there is deliberately no flag to disable it.
- **Information threshold.** A source whose postage stamp lies (almost) entirely beyond the detector edge has a negligible design diagonal but is still technically "observed". Solving for it is unconstrained and produces absurd values, so sources with $\mathrm{diag} < 10^{-6}\max(\mathrm{diag})$ are excluded and keep their current amplitude. (Before this, ~10% of sources were driven to $a \approx 0$, which also *froze* them: the gradient $\partial \mathcal{L}/\partial \log A = a\,\partial\mathcal{L}/\partial a$ vanishes as $a\to0$, so SGD could never revive them.)
- **Regularization / failure safety.** A tiny ridge (`damping`) is applied in the scaled space, guaranteeing the operator is strictly positive definite (sources in no exposure get an O(1) shift instead of a zero row, which previously made the system singular and produced `NaN` from CG). Failure is handled *branchlessly* inside the jit — if the solution is non-finite the previous amplitudes are kept — so a bad linear solve can never corrupt the fit and there is no Python-level `if` to trigger re-tracing.
- **Inner iteration cap.** `cg_max_steps` defaults to 50; the preconditioned system typically converges in 1–5 iterations.

### 5.6 SGD interlace

`infer_parameters` interleaves the two updates by default: every `amp_solve_every` SGD steps (default 16) it replaces the log-amplitude column with the `solve_log_amplitudes` result (shapes and backgrounds held fixed), plus one final solve after the loop. The jitted solver is **built once before the loop** (`_make_amplitude_solver`), so the interlace never re-traces or recompiles it. SGD's optimiser state is untouched by this external update. `amp_verbose=True` prints the loss before/after each solve, and the interlace is automatically disabled (with a warning) if the generators do not expose `source_stamps`. This is remarkably effective — in the full-scale mock the first interleaved solve drops $\chi^2/\text{pixel}$ from $1.9\times10^5$ to $\sim\!42$ in a single call, where pure SGD needs many steps to do the same.

### 5.7 Diagnostic: `compute_lm_loss`

Evaluates the **exact same** residual function that LM uses internally (`_make_residual_fn`), returning `sum(r²)/n_pixels`. This can be checked against `compute_loss` (which uses `_per_exposure_sumsq`) to verify consistency. Both should agree to machine precision.

---

## 6. Mock Data Pipeline

`scripts/mock_spherex_images.py` implements a full end-to-end simulation:

### 6.1 Configuration

Key globals at the top of the file:

| Parameter | Default | Meaning |
|---|---|---|
| `DOWNSAMPLE` | 16 | Detector binning factor (2048 → 128 pixels) |
| `PSF_SCALE` | 1.0 | Multiplier on PSF FWHM |
| `N_SOURCES` | 128 | Number of catalog sources |
| `N_EXPOSURES` | 32 | Number of random detector pointings |
| `N_LAMBDA` | 1 | Wavelength integration samples (quantile-based) |
| `HALF_STAMP_FLOOR` | 3 | Minimum postage stamp radius (pixels) |
| `BACKGROUND_FACTOR` | 2.0 | Background = FACTOR × peak rate of faintest star |
| `SPECTRUM_KIND` | `"blackbody"` | Shape template: `"blackbody"` or `"nn"` (§6.4) |
| `NN_N_PARAMS` | 1 | Number of shape parameters θ of the neural network |
| `NN_N_HIDDEN_LAYERS` | 3 | Hidden layers of the neural network |
| `NN_HIDDEN_SIZE` | 32 | Neurons per hidden layer |
| `NN_SEED` | 0 | Seed for the (frozen) random weights |
| `NN_TARGET_LOG_SHAPE_STD` | 0.5 | Target spread of the rescaled in-band log-shape |

### 6.2 Pipeline steps

1. **Amplitude range** (`_report_photon_counts`): amplitudes are specified *directly* on the physical $f_\lambda(\lambda_0)$ scale via the `AMPLITUDE_MIN`/`AMPLITUDE_MAX` constants — there is no photon-count *targeting*. The implied detected photon counts are only *reported*, by integrating the active model's shape (at its reference shape parameters — 3000 K for the blackbody, θ = 0 for the network) over each band with the Gaussian filter, times `EXPOSURE_TIME`. The same reference shape sets the background level (`_compute_background`), so both diagnostics track the active model automatically.

2. **Generate catalog** (`_generate_catalog`): sources uniformly distributed on the sphere within a spherical cap, with shape parameters drawn from the **active model's prior** (log of U(3000, 8000) K for the blackbody, θ ~ N(0, 1) for the network) and power-law amplitudes P(A) ∝ A^{-1.5}.

3. **Generate exposures** (`_generate_exposures`): random detector pointings with random bands (1–6). Computes `half_stamp = max(ceil(5×FWHM/pixel_scale), HALF_STAMP_FLOOR)`.

4. **Filter sources** (`_filter_sources`): for each exposure, select sources within `half_stamp` of the detector edges. Builds per-exposure `(positions_pix, params, band, wcs, half_stamp, src_idx)` tuples where `params` has shape `(n_in, 1 + P)` — amplitude first, then the `P` shape parameters — and `src_idx` maps into the **full** (unfiltered) catalog.

5. **Generate images** (`_generate_and_save`):
   - Creates band-generator instances with `lambda_slope_scale=DOWNSAMPLE` and the SAME `spectrum_model` instance for every band
   - Generates noiseless rate images via `gen(positions, params, ...)`
   - Adds background (constant per band) and multiplies by `EXPOSURE_TIME` to get counts
   - Adds Gaussian noise: `σ = sqrt(true_counts + noise_floor²)` with `noise_floor = 1.0`
   - Saves true/noisy/predicted/residual PNGs

6. **Inference** (see §5). The initial guess uses a **model-specific draw that deliberately differs from the prior**: amplitudes uniform in `[AMPLITUDE_MIN, AMPLITUDE_MAX]`, blackbody shape parameters uniform in log T, network parameters `N(0, NN_INIT_STD²)`.

7. **Diagnostic plots**: loss history (asinh y-scale), true-vs-recovered scatter, predicted/residual images, and — for a flexible model — a true-vs-recovered *spectrum* overlay (`plots/spectra_comparison.svg`, see §6.4).

   The true-vs-recovered scatter (`plots/comparison.svg`) has **one panel per parameter** (`1 + P` panels, caller-supplied labels; with the blackbody's `P = 1` this is the familiar log A / log T pair). It highlights "bright" sources, defined as those **detected at S/N > `SNR_THRESHOLD` (5) in at least `MIN_BANDS` (3) distinct SPHEREx bands** — bands, not exposures, so that a source repeatedly observed in one band (which constrains only one point of its spectrum) is not promoted. The per-(source, exposure) S/N is the expected matched-filter significance of the source's own noiseless model counts,

   $$\text{S/N} = \sqrt{\sum_i \frac{(m_i\, t_\text{exp})^2}{\sigma_i^2}}$$

   where $m_i$ is the source-only model rate in pixel $i$ (from its unit-amplitude postage stamp, so PSF and amplitude included) and $\sigma_i$ the per-pixel noise. Using the model rather than the noisy data keeps the label free of noise bias, and taking the expectation of the optimal flux estimator makes this simply "how many sigma the source's flux stands above the noise" (zero-flux pixels drop out automatically). All of it uses the **true** parameters: deriving the label from recovered parameters would be circular, since a source whose amplitude estimate diverges upward would then be labelled "bright" precisely because of the convergence failure being diagnosed. See `_source_snr` / `_compute_bright_mask`.

### 6.3 Wavelength gradient scaling with downsampling

When `DOWNSAMPLE > 1`, the detector has fewer pixels covering the same physical area. The WCS pixel scale remains 6.2"/pixel, so the physical detector extent shrinks to `(2048/DOWNSAMPLE) × 6.2"`. Without correction, the wavelength range across the detector is compressed by `DOWNSAMPLE`. The fix: `lambda_slope_scale = DOWNSAMPLE` in the generator constructor, which multiplies the µm/arcsec slope so the wavelength range across the (smaller) detector matches the full band.

### 6.4 Optional neural-network spectrum model (`--spectrum nn`)

The mock can be run with a **frozen random `NeuralNetSpectrum`** instead of the analytic blackbody, to test that the pipeline recovers spectra whichever shape template generated them:

```bash
python scripts/mock_spherex_images.py                          # blackbody (default)
python scripts/mock_spherex_images.py --spectrum nn            # frozen random net
python scripts/mock_spherex_images.py --spectrum nn --nn-shape-check
```

**Why so little changed.** `_build_band` gained one optional `spectrum_model` argument (§3.6), the mock builds **one** model instance and passes it to every band's generator, and everything else was already generic:

* `image.py` / `image3.py` / `psf.py` / `transmission.py` never look at the model beyond calling `spectrum_model(λ, shape_params)`.
* The amplitude solve is untouched — it only needs the model to be *linear in amplitude* and the shape to be *independent of it*, both of which `NeuralNetSpectrum` guarantees.
* SGD/LM are untouched: they differentiate only `(log_params, log_backgrounds)`.
* Parameter bookkeeping is written for `(N, 1 + P)` throughout, so `--nn-params 3` works with no further changes.

**The weights are frozen by construction.** The model is never handed to an optimiser, and the generator is a closed-over constant rather than a JIT argument — that is the *only* way the weights could become traced/differentiated values. `end_to_end_mock` hashes every leaf of the model before and after inference and reports `Spectrum model weights unchanged (frozen)`, and `tests/test_mock_spectrum_model.py` asserts the same thing so a future refactor cannot silently unfreeze it.

**Why the output layer is rescaled.** A randomly initialised MLP produces an arbitrary spectrum: measured over the full 0.75–5 µm range, the raw network's log-shape spread is ~1/100 of the blackbody's, so left alone the frozen net would give near-featureless, decades-wide (or decades-narrow) spectra and the directly specified `AMPLITUDE_MIN`/`AMPLITUDE_MAX` range, the background level and the resulting detection S/N would no longer correspond to the blackbody run. `_rescale_nn_output` therefore scales the final layer's weight block so the λ₀-normalised log-shape has standard deviation `NN_TARGET_LOG_SHAPE_STD = 0.5`. This is *exact* rather than iterative: the shape is `exp(raw(λ,θ) − raw(λ₀,θ))`, which is linear in the final weight block (the final **bias cancels** in the difference), so scaling that block by `f` scales the log-shape by exactly `f`. In practice the factor is ~10².

**`--nn-shape-check` (run this first).** A flexible model can fail *before* any fitting happens — either because θ does not move the in-band shape (θ is then unidentifiable however long you fit) or because the frozen net's spectra span too many decades to be comparable to the blackbody run. The check answers both:

* per-band log-shape range over draws from the prior, e.g. (seed 0) band 1 `[-0.02, +0.06]`, band 6 `[-1.37, -1.20]`;
* the singular values of $\partial \log \text{shape} / \partial \theta$ on the sampled wavelength grid: near-zero singular values are directions that leave the spectrum unchanged. Seed 0 gives `[46.0]` for `P = 1` (rank 1, condition number 1) and `[31.8, 8.1, 5.2]` for `P = 3` (rank 3, condition number 6.1), so θ is well constrained by the band coverage at both settings — but at `P = 3` the parameters are near the information limit of six bands, so treat large `P` as a test of the plumbing rather than as a physically meaningful model;
* `shape(λ; θ)` for 64 prior draws, plus the implied photon counts.

**Reading the results.** Since a random net's θ has no physical meaning, `plots/comparison.svg` (θ vs θ) is only a parameter-recovery check; the interpretable diagnostic is `plots/spectra_comparison.svg`, which overlays the true and recovered *spectra* for the brightest sources and reports the median $|\Delta \log \text{shape}|$ over them. In the small-scale smoke runs (`N_SOURCES = 12`, 4 exposures, 64 SGD steps), both models reach $\chi^2/\text{pixel} = 0.993$ at the *true* parameters and 0.994–0.996 at the recovered ones, with a median $|\Delta \log \text{shape}|$ of ~0.12–0.31 dex for the few bright sources.

**Cost.** A 3×32 MLP costs much more arithmetic per wavelength sample than the closed-form Planck function, and the shape is evaluated at every sample of every sub-pixel of every source. Measured on an identical problem (8 sources, band 3, `half_stamp = 10`, `n_lambda = 1`, oversampling 2), the forward model takes 215 ms with the blackbody and 263 ms with the network — only +22%, because the PSF evaluation, sub-pixel bookkeeping and scatter-add dominate the per-pixel work rather than the spectrum evaluation. End to end at the small scale above (12 sources, 4 exposures), 64 SGD steps took 12.0 s versus 16.0 s. Expect the gap to widen with `hidden_size` and `n_lambda`.

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

### 7.13 Amplitude units and the need for preconditioning

The spectrum models return a *shape* normalised to 1 at $\lambda_0$, and the generator multiplies in $\exp(\log A)$ itself. Consequently the design operator of the amplitude solve (§5.5) is expressed in **absolute** amplitude units, and since catalog amplitudes can span many decades, the unpreconditioned normal matrix $M^\top M$ overflows float32 (observed: CG returning a zero vector after one step). The Jacobi scaling transform in `solve_log_amplitudes` is therefore a *correctness* requirement, not just a speed optimisation: keep `precondition=True` (the default) whenever `source_stamps` is available. Note also that the shape is pinned to exactly 1 at $\lambda_0$, so $d(\text{shape})/dT = 0$ there — any gradient test must evaluate at a wavelength away from $\lambda_0$.

### 7.14 Three failure modes of the amplitude solve, and their symptoms

1. **Singular system → `NaN` from CG.** Sources absent from every exposure have a zero design diagonal, so an unscaled/regularized operator is singular and `lineax.CG` raises "linear solver returned non-finite output". Fixed by a tiny ridge in the scaled space plus an O(1) shift on the excluded rows, and by running the solve with `throw=False` and checking finiteness.
2. **Negligible-but-nonzero stamps → absurd amplitudes.** A source sitting just outside the detector edge is "observed" (its index appears in `src_idx`) yet contributes essentially nothing, giving $\mathrm{diag}\sim10^{-32}$. Dividing by its own tiny diagonal yields a meaningless amplitude. Fixed by the relative information threshold $\mathrm{diag} > 10^{-6}\max(\mathrm{diag})$.
3. **Frozen amplitudes after clamping.** Clamping a negative least-squares amplitude to a tiny positive value looks harmless but is fatal: $\partial\mathcal{L}/\partial\log A = a\,\partial\mathcal{L}/\partial a \to 0$ as $a\to0$, so SGD can never recover that source. Leave ill-constrained sources at their current value instead of clamping them to ~0.

### 7.15 Band-grouped helpers receive already-group-indexed arrays

`_prepare_band_group` stores `group_idx` as *global* exposure indices, and every group-level helper (e.g. `_prepare_band_group`'s consumers) is passed arrays already indexed by it. Re-indexing inside a helper (`log_backgrounds[idx][idx]`) is therefore wrong — and worse, JAX silently *clamps* out-of-range gather indices instead of erroring, so it corrupts results without raising. The bug was masked in testing because the mock generates one constant background per band: within a band group every entry is identical, so the mis-indexing was a no-op. The regression test therefore uses **two exposures sharing a band with different per-exposure backgrounds**.

### 7.16 Tracing overhead, not arithmetic, dominates naive JAX helper loops

The amplitude solve originally ran as plain Python. Profiling showed the *arithmetic* took ~0.1 s while the wall clock was ~2.2 s: **~91% was JAX tracing** — every call rebuilt the same jaxpr for the `vmap(over exposures) ∘ lax.map(over sources)` stamp program, and `lineax`'s jitted `linear_solve` re-compiled too because the `matvec` closure captured fresh arrays. The fix is structural, not algorithmic: compile the whole computation once (`_make_amplitude_solver`) and pass the changing values in as *arguments*, not closures. Concretely:

- Rebuilding a jitted callable per invocation (as a naive `solve_log_amplitudes` wrapper does) still costs ~1.2 s per call, versus ~0.12 s when the compiled solver is reused. Keep the factory call **outside** any optimisation loop.
- The same pattern applies to any helper that is called repeatedly; prefer building the compiled callable once and caching it.
- Note the contrast with §7.4: jitting the whole *LM residual* is still a bad idea (minutes of compile for a huge graph). The amplitude solve compiles in ~1.3 s because its graph is small — the rule is "measure the compile time", not "never jit".

### 7.17 Spectrum models must not batch themselves

The spectrum-model interface is *one source × many wavelengths*: `spherex.image._one_subpixel_rate` calls `spectrum_model(lambdas, shape_params)` per source and per sub-pixel and expects `(N_λ,)` back, having already applied `jax.vmap` over sub-pixels and `lax.scan`/`lax.map` over sources. A model whose `__call__` only accepts a *scalar* wavelength therefore cannot be dropped in — every call site would need its own `vmap` — and a model that tries to concatenate a wavelength *vector* with the parameters gets extra *features* rather than extra examples (an MLP layer wants the feature axis last). The pattern that works: a scalar core plus an internal `jax.vmap` over wavelengths, with source batching left to the caller. `BlackbodySpectrum` and `NeuralNetSpectrum` both follow it, which is what makes the generators model-agnostic.

---

### 7.18 "Frozen" is a property of the call graph, not of the object

An optimiser can only move what it is handed, so the neural network's weights are frozen for a purely structural reason: `infer_parameters` receives `(log_params, log_backgrounds)`, while the generator — and hence the model — is a closed-over Python constant inside the loss. The failure mode to avoid is accidental widening of that graph. Anything that turns the generator into a *dynamic* JIT argument (re-jitting the loss with the generator as a traced argument rather than a static/closed-over one, or splicing the model's arrays into the parameter pytree) converts its weights into tracers that the optimiser's update is free to alter, and "frozen" silently stops being true — with no error, just slowly drifting spectra.

Cheap guard: fingerprint every leaf of the model before and after the fit and compare. `_model_fingerprint` hashes dtype, shape and bytes of each leaf, so it is sensitive to a single changed value; `end_to_end_mock` prints the verdict and `tests/test_mock_spectrum_model.py` asserts it. Both a spectrum *prior* and the *initial guess* also need to be model-specific — the blackbody draws `log T = log U(3000, 8000 K)` but initialises uniformly in `log T`, while the network draws and initialises `N(0, σ²)`.

### 7.19 Random initialisation needs an explicit output scale — and an identifiability check

A randomly initialised MLP has an arbitrary spectrum. Measured here, the raw 3×32 network's λ₀-normalised log-shape varied ~100× less across 0.75–5 µm than the blackbody's over its temperature prior, so it would have produced a nearly featureless, near power-law spectrum, and the directly specified amplitude range, the background level and the resulting detection S/N would no longer have been comparable to the blackbody run they were chosen for. Fix: rescale the final layer's **weight block** so the in-band log-shape has the intended spread (`NN_TARGET_LOG_SHAPE_STD = 0.5`). This is exact, not iterative, because the shape is `exp(raw(λ,θ) − raw(λ₀,θ))` — linear in that block, with the final **bias cancelling** in the difference — so scaling the block by `f` scales the log-shape by exactly `f`.

Second, a flexible model can be unfittable regardless of the optimiser: if a direction in θ leaves the in-band shape unchanged, that parameter is unidentifiable *in principle*, and a poor recovery is not evidence of a convergence problem. Check it before fitting by looking at the singular values of $\partial \log \text{shape}/\partial\theta$ over the observed wavelengths (unit-independent, because the log-shape is dimensionless): near-zero singular values are the null directions. Practically, a parameter count approaching the number of distinct bands should be treated as a plumbing test, not a physical model.

Also worth remembering when *interpreting* the fit: with a random network, θ has no physical meaning, so the θ-vs-θ scatter is only a parameter-recovery plot; the scientifically meaningful diagnostic is whether the recovered *spectrum* matches the true one (`plot_spectra_comparison`), and its error should be quoted in dex of log-shape rather than as a parameter offset.

---

## 8. File Map

| File | Purpose |
|---|---|
| `spherex/constants.py` | Physical constants in codebase-native units (kJ, µm, arcsec) |
| `spherex/spectrum.py` | `BlackbodySpectrum` (analytic) and `NeuralNetSpectrum` (MLP) — shape-only source spectrum templates, both normalised to 1 at the global `LAMBDA_0`, plus the `[log_amplitude, shape…]` layout helpers |
| `spherex/psf.py` | `GaussianPSF` — wavelength-dependent Gaussian PSF |
| `spherex/transmission.py` | `GaussianFilterTransmission` — LVF transmission with quantile interface |
| `spherex/image.py` | `_one_subpixel_rate` (quantile integration; splits amplitude out of `source_params`), `ImageGenerator` (scan-based) |
| `spherex/image3.py` | `ImageGenerator3` (chunked-batch, uses `_one_subpixel_rate` from `image.py`), plus `source_stamps` for the amplitude-solve preconditioner |
| `spherex/config.py` | `SpherexImageGenerator`, `SpherexImageGenerator3` — pre-configured per-band generators, `_build_band(..., spectrum_model=None)` (the spectrum-model injection point), band table |
| `spherex/__init__.py` | Public API exports |
| `scripts/inference.py` | `infer_parameters` (SGD, with amplitude interlace), `infer_parameters_lm` (LM), `solve_log_amplitudes` (direct amplitude least-squares), `compute_loss`, `compute_lm_loss`, `plot_loss_history`, `plot_comparison` (one panel per parameter, `1 + P`), residual functions, custom LM solver |
| `scripts/mock_spherex_images.py` | End-to-end mock: spectrum-model factory + shape-parameter bookkeeping (Step 0), catalog, exposures, image generation, hybrid SGD+LM inference, diagnostics (`--spectrum {blackbody,nn}`, `--nn-shape-check`), benchmarks |
| `tests/test_image3.py` | Numerical tests verifying ImageGenerator3 ≡ ImageGenerator |
| `tests/test_mock_spectrum_model.py` | Model factory (determinism, output rescaling), generic `(N, 1 + P)` bookkeeping, generator injection, and the frozen-weights guarantee |
| `pyproject.toml` | Dependencies: `jax`, `equinox`, `optimistix`, `lineax`, `numpy` |
| `/memories/jax_optimization_lessons.md` | Persistent notes on LM trust-region tuning and memory safety |
