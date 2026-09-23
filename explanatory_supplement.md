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

- **Quantile-based wavelength integration** (`_one_subpixel_rate` in `spherex/image.py`): instead of a fixed `linspace` + trapezoid rule (which mostly samples the near-zero tails of a narrow bandpass), the code draws wavelength samples at evenly spaced quantiles of the transmission profile's normalised CDF via `transmission.quantile(q, omega_p)`. This concentrates samples where the transmission actually has support. At `N_LAMBDA = 1` the RMS residual against a 63-sample reference is only `2.3e-5` — far more accurate than naive quadrature at the same sample count, *provided the spectrum varies slowly across the bandpass*. That proviso is model-dependent: it holds comfortably for the blackbody (~1e-5) and for an unembedded network (~1e-6), but a heavily Fourier-embedded network varies by a factor of ~2 across a single bandpass, which costs ~4e-3 relative RMS (§6.4).
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

where $B_\lambda$ is the Planck function and $\lambda_0$ is a single **global** reference wavelength (`LAMBDA_0 = 1.0 µm` in `spherex/spectrum.py`) shared by every band — so the modelled spectrum stays smooth in amplitude across the whole wavelength range. The dimensionless shape $\hat{B}_\lambda$ equals exactly 1 at $\lambda_0$.

**The models return an unnormalised log-flux, and the CALLERS do the normalising.** A spectrum model returns $\log f_\lambda$ up to an additive constant that is independent of wavelength:

```
spectrum_model(wavelength, shape_params) -> log f_lambda + const
```

and the image generators turn that into the physical spectrum by anchoring it at $\lambda_0$:

$$f_\lambda(\lambda) = \exp\big(\log A + \texttt{model}(\lambda, \theta) - \texttt{model}(\lambda_0, \theta)\big)$$

Nothing about this is specific to a blackbody — a blackbody, a neural network, or any future template differ only in what they return. Three properties make it work:

* **The models are anchor-free.** They know nothing about `LAMBDA_0` and never exponentiate, so a model is a pure (log-)kernel. The normalisation lives in `spherex.spectrum.reference_log_flux` / `normalized_shape` / `normalized_source_params`, and the constant is evaluated **once per source** by the generators, then folded into the amplitude column (`normalized_source_params`) so the per-sub-pixel kernel is a single `exp` and a single add. Doing it any later (inside the model) re-evaluates a wavelength-independent quantity once per sub-pixel — see §7.20.
* **Log space, not linear.** An unnormalised *linear* kernel would mean exponentiating an arbitrary offset and dividing: the network's raw output has an offset that is independent of wavelength and that the data cannot constrain (it is a null direction of the likelihood), so it can drift during inference and overflow float32. Exponentiating the (bounded) *difference* cannot overflow.
* **The convention survives.** Because the caller divides by the model's value at $\lambda_0$, $A = f_\lambda(\lambda_0)$ is still a physical flux density and $\log A$ keeps its direct interpretation.

**Amplitude is not part of the spectrum model.** It is carried as a separate per-source parameter, always the **first** column of `source_params` (`LOG_AMPLITUDE_INDEX = 0`); `source_params[:, 1:]` are the shape parameters $\theta$. All parameters are stored in log-space to enforce positivity and improve optimizer conditioning.

`BlackbodySpectrum` therefore has a single free parameter, $\log(T/\text{kK})$. Crucially, the forward model is **exactly linear in the amplitude** $a = \exp(\log A)$ — a source's contribution is just $a$ times its unit-amplitude postage stamp — which is what makes the direct amplitude least-squares solve of §5.5 possible. Note that this linearity is preserved by the normalisation: the folded constant is independent of $a$.

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

- **Attributes**: `n_params = 1` (log-T only — the amplitude lives *outside* the model). There is no `lambda_0` attribute: the model is anchor-free.
- **`__call__(wavelength, shape_params)`**: returns the **unnormalised log-flux** $\log B_\lambda$ minus the constant $\log(2hc^2)$, i.e. $-5\log\lambda - \log(e^x - 1)$ with $x = hc/(\lambda k_B T)$. Only *differences* of this quantity are physical, which is exactly what the caller's normalisation at $\lambda_0$ produces: $\hat{B}_\lambda = \exp(\texttt{model}(\lambda,T) - \texttt{model}(\lambda_0,T))$.
- **Layout / normalisation helpers**: `LOG_AMPLITUDE_INDEX`, `split_source_params`, `join_source_params` define the `[log_amplitude, shape…]` convention; `reference_log_flux`, `normalized_log_shape`, `normalized_shape` and `normalized_source_params` implement the anchoring at `LAMBDA_0` (§2.1).
- **Implementation detail**: uses `inv_T_scale = HC/(λ·KB)` to avoid float32 underflow in the Planck gradient; the exponent is clipped at 50 so `expm1` cannot overflow.

### 3.2 `NeuralNetSpectrum` (`spherex/spectrum.py`)

A flexible alternative to the blackbody. A small MLP maps the *embedded wavelength* to a log-flux, with the shape parameters entering through **FiLM** modulation of every hidden layer (§3.2.1); `__call__` returns that log-flux directly — the model neither normalises nor exponentiates:

$$\texttt{model}(\lambda; \theta) = \mathrm{NN}(\lambda; \theta), \qquad
\hat{B}_\lambda(\lambda) = \exp\big(\mathrm{NN}(\lambda; \theta) - \mathrm{NN}(\lambda_0; \theta)\big)$$

which is *exactly* 1 at $\lambda_0$ for **any** $\theta$. Keeping the blackbody's normalisation convention is what makes the amplitude (first column of `source_params`) remain $f_\lambda(\lambda_0)$, and it is why **the entire amplitude pipeline works unchanged**: the least-squares solve only requires the model to be linear in amplitude and the shape to be independent of it, both of which hold (verified numerically — the solve recovers amplitudes to ~2×10⁻² in log on a synthetic problem, driving $\chi^2/\text{pixel}$ from ~2×10⁴ to ~1.1).

- **Attributes**: `layers` (list of `eqx.nn.Linear`), `film` (list of `FiLMLayer`, one per *activated* layer, i.e. `n_hidden_layers + 1`), `norms` (list of `eqx.nn.LayerNorm`, one per activated layer when `layer_norm` is on, else empty), `layer_norm` (bool), `n_params` (number of *shape* parameters), `n_hidden_layers`, `hidden_size`, plus the FiLM hyperparameters `film_hidden_layers` and `film_hidden_size_factor` and the resolved `film_hidden_size`
- **`__init__(n_params, n_hidden_layers, hidden_size, n_embeddings=8, delta_ln_wavelength=ln(5/0.75), film_hidden_layers=1, film_hidden_size_factor=1.0, layer_norm=False, *, key)`**: `key` is **required** (`eqx.nn.Linear` needs a PRNG key); one key per layer is derived with `jax.random.split`, and the FiLM branches from an independent subkey, so adding or resizing them cannot change the main MLP's weights for a given seed. All fields are declared as annotations, which is mandatory for `equinox.Module` (pytree) behaviour. The network input is `1 + 2·n_embeddings` wide — **`theta` is not an input feature**.
- **Batching contract** (identical to `BlackbodySpectrum`): `__call__(wavelength (N_λ,), shape_params (P,)) -> (N_λ,)`, i.e. **one source, many wavelengths**. The image generators already supply the other axes — `jax.vmap` over sub-pixels and `lax.scan`/`lax.map` over sources — and call `spectrum_model(lambdas, shape_params)` once per source per sub-pixel, so the model must not try to batch them itself.
- **Internals**: the network is evaluated one wavelength at a time and vectorised with `jax.vmap` — a layer expects the feature axis last, and the scalar Fourier embedding is the natural core of the computation. The FiLM parameters are computed **once per call** (they do not depend on wavelength) and passed into the vmapped core, so the branches are never re-evaluated per wavelength sample. To batch over sources as well, compose a second vmap at the call site: `jax.vmap(model, in_axes=(None, 0))(wavelengths, shape_params_batch)`.
- **Cost**: the kernel is evaluated at every wavelength sample of every sub-pixel of every source, and an MLP costs far more per point than the closed-form blackbody, so this slows the forward model *and* each interleaved amplitude solve (which rebuilds the postage stamps). Since the caller supplies the $\lambda_0$ value, the MLP runs exactly `n_wavelength_samples` times per sub-pixel (measured: 2.05x fewer MLP evaluations than the old contract at `n_wavelength_samples = 1`); keep `hidden_size` modest. Measured on image generation **alone** (band 3, 57 sources, 128² detector, `n_lambda = 1`, steady state): 146 ms for the blackbody, **199 ms (+36%)** for the FiLM network without Fourier embeddings and **202 ms (+38%)** with 8 — so the branches add little next to the main MLP, and the embedding is nearly free at `n_lambda = 1`.
- The output is a log-flux, so it is unbounded — but the caller exponentiates the *difference* to its value at $\lambda_0$, which is bounded. No clipping is needed.

**Wavelength embedding (Fourier features).** The wavelength is not fed to the network raw. It is expanded into `ln(lambda)` plus `sin`/`cos` of `n_embeddings` geometrically spaced frequencies, $k_i = 2^i\pi/\Delta\ln\lambda$ with `delta_ln_wavelength` defaulting to $\ln(5/0.75)$ (the full range). The i-th frequency completes $2^{i-1}$ cycles across the span, so the finest one has a period of $2\,\Delta\ln\lambda/2^{n}$ — a constant *fraction* of the wavelength (~3% at the defaults, i.e. ~128 samples per e-folding of $\lambda$).

Log-wavelength space rather than $\lambda$ is deliberate, and matches the instrument: a SPHEREx bandpass has a constant fractional width, $\sigma_\lambda/\lambda_c = 1/(2.355R)$, just as emission lines, absorption edges and dust features are quasi-log-periodic. Embedding $\lambda$ linearly gives each Fourier feature a fixed *absolute* period, so its richness relative to the bandpass drifts across the range — the fluctuations become relatively too rapid at long wavelengths and too coarse at short ones. Measured in-bandpass peak-to-peak $\log(\text{shape})$ at `n_embeddings = 8` (across $\lambda_c \pm 2\sigma_t$, median over 32 draws from the prior): linear embedding 0.28 / 0.72 / 1.00 for bands 1 / 3 / 6, versus log embedding 0.39 / 0.90 / 0.33. The log version is the physically sensible one — band 6, with the highest resolving power ($R = 130$), now shows the *least* relative variation, whereas the linear version made it the most.

This expands what the model can represent but not how many free parameters it has: the frequencies are fixed, so $\theta$ still selects a one-parameter family of curves. The extra cost is $2\,n_\text{embeddings}$ transcendentals per call. **Caveat:** finer features mean the `n_lambda = 1` quantile integration is less faithful — see §6.4.

#### 3.2.1 How $\theta$ enters: FiLM modulation

$\theta$ is **not** concatenated to the wavelength features. Every hidden layer has its own `FiLMLayer` — a small MLP mapping $\theta$ to a per-neuron scale and offset — applied **before** the nonlinearity:

$$z_l = W_l h_{l-1} + b_l, \qquad \gamma_l, \beta_l = \mathrm{FiLM}_l(\theta), \qquad h_l = \mathrm{siLU}(\gamma_l \odot z_l + \beta_l)$$

so $\theta$ modulates the hidden features multiplicatively instead of entering as one more input feature alongside the wavelength. Four properties matter:

- **The branch width scales with $\theta$.** Each branch's hidden layers have width $W = \max(\mathrm{round}(\texttt{film\_hidden\_size\_factor} \cdot P), 1)$, with the factor defaulting to 1.0 — i.e. exactly as wide as $\theta$. A *fixed* small width (4, say) would have capped the modulation at that many independent combinations of the shape parameters however large $P$ is, so the model would have behaved as though $\theta$ were lower-dimensional. The factor is the lever for more capacity; `film_hidden_layers = 0` degenerates a branch to a single linear map $\theta \to (\gamma, \beta)$.
- **One branch per *activated* layer**, i.e. `n_hidden_layers + 1` of them, because the input projection is itself a hidden layer. That is also why $\theta$ is never ignored, even at `n_hidden_layers = 0`, where that branch is its only route into the network.
- **Plain random initialisation**, like every other layer. The "identity at initialisation" scheme used when *training* FiLM ($\gamma = 1$, $\beta = 0$, i.e. a zeroed output projection) is deliberately avoided here: with a linear output projection it makes $\partial\gamma/\partial\theta = 0$ for **every** $\theta$, so the shape parameters would have no effect on the spectrum at all — fatal when the weights are frozen and $\theta$'s entire job is to select the spectrum. `tests/test_spectrum.py` pins this down with a non-zero-$\theta$-Jacobian test.
- **No guaranteed null direction — but $\theta$ can still lose its effect.** If every $\gamma$ ended up near zero, the hidden layers would stop depending on wavelength and the shape would collapse to a constant for any $\theta$. Random initialisation makes that unlikely, and the mock's output rescale (§6.4) turns it into a conspicuous rescaling factor, but it is the failure mode to look for if recovered $\theta$ values scatter wildly.

The branches are cheap: $(\gamma, \beta)$ are wavelength-independent, so they are computed once per call and broadcast over the wavelength axis, costing roughly $W(P + 2H)$ arithmetic per call against $O(H^2)$ *per wavelength sample* for the main MLP.

#### 3.2.2 Optional per-wavelength LayerNorm (`layer_norm=True`)

With `layer_norm=True`, an `eqx.nn.LayerNorm` is inserted after the activation of **every activated layer** (so it is index-aligned with `film`, `n_hidden_layers + 1` of them). Both the normalisation and the FiLM modulation act on a *single wavelength's* feature vector, which is the property that makes the option safe: the value returned for a given $(\theta, \lambda)$ cannot depend on which other wavelengths or $\theta$ values share the batch. Normalising over the **wavelength** axis would instead make $f_\lambda$ a function of the caller's grid — exactly what the interface forbids, and what `tests/test_spectrum.py::test_neural_net_layer_norm_is_call_time_deterministic` rules out (it checks both a reduced wavelength grid and a batch of $\theta$; the rejected alternative fails the same check by $O(1)$, since a $\lambda$-axis statistic swings by 0.26–0.75 under it).

The reason it is an *option*: it changes how much spectral structure a **randomly initialised** network has, which matters only while the weights are frozen and used to generate mock data. Measured raw (unrescaled) per-source log-shape std over 0.75–5 µm, median over 64 prior draws:

| architecture | off | `last` layer only | every layer | before activation |
|---|---|---|---|---|
| 1×32, 8 embeddings | 0.015 | 0.042 | **0.100** | 0.042 |
| 2×16, 8 embeddings | 0.001–0.004 | 0.003–0.014 | 0.029–0.113 | 0.005–0.018 |
| 2×16, no embeddings | 0.001 | 0.003 | 0.016–0.048 | 0.003–0.006 |
(i.e. the equivalent *factor the output rescaling would need*, 0.5/std: ~33 → ~5 at the mock's defaults; the ranges are over seeds 314159 / 7 / 11.)

Three conclusions worth keeping: (i) LayerNorm does **not** remove the need for the output rescale — it divides by a *wavelength-dependent* quantity, so it cannot pin a statistic defined over wavelengths, and the achieved spread still varies with the seed (a factor 4.4–17.6 over three seeds at 2×16); (ii) it is not a cosmetic change — it alters the family of spectra the model can produce, so it should be re-checked with `--nn-shape-check` and `--compare-n-lambda`; (iii) at the mock's defaults it costs $2H$ parameters per activated layer and leaves identifiability essentially unchanged (singular values $[51.0]$ v. $[54.7]$ with it off). Once the weights are *trained* the whole flatness argument disappears and the normalisation becomes an ordinary architectural choice.

The affine scale and offset are initialised to 1 and 0 (identity), which is what makes the option a pure standardisation at initialisation; they are ordinary parameters, so a caller that trains the model trains them, and the mock's frozen-weights fingerprint covers them.

#### 3.2.3 `BlackbodyPlusNNSpectrum`: continuum plus modulation

A composite of the two models above, obtained by **splitting the shape parameters**: $\theta_0$ is the blackbody's log-temperature, $\theta_{1:}$ go to the network, and the two log-flux kernels are *added*:

$$\texttt{model}(\lambda; \theta) = BB(\lambda; \theta_0) + NN(\lambda; \theta_{1:}), \qquad n_\text{params} = 1 + P_\text{nn}$$

In linear terms the blackbody is multiplied by the network's dimensionless modulation, i.e. the familiar "physical continuum plus flexible correction" model in which the network absorbs whatever the blackbody cannot describe. No new machinery is needed: both parts obey the log-flux contract (§2.1), so their sum does too, and the caller's normalisation at $\lambda_0$ still yields a shape of exactly 1 there. The network's own output rescaling (§6.4) applies to the network part *before* wrapping, and it then sets the amplitude of the modulation on top of the continuum.

Two consequences worth remembering:

- **The continuum dominates the total shape.** A 3–8 kK blackbody has a per-source log-shape std of ~1.75 over 0.75–5 µm against a target of 0.5 for the rescaled random network, so the network is a modulation of relative size $\sim e^{0.5}$ rather than the whole spectrum; the shape check reports the composite against the blackbody reference row accordingly (measured at the mock's defaults with `--spectrum blackbody+nn`: composite 1.70, blackbody 1.75).
- **The split is positional**, so everything that draws, initialises, labels or summarises $\theta$ must treat column 0 as `log T`. In the mock this is centralised in `_model_blocks`, which returns the `(kind, n_params)` blocks in column order — the blackbody prior (`log T = log U(3,8) kK`), initial guess (uniform in `log T`) and label are then applied to column 0 and the network's (`N(0, NN_PRIOR_STD²)` prior, `N(0, NN_INIT_STD²)` init, `theta_k` labels) to the rest, for any `P` and without touching the single-model code paths.

Identifiability is not a problem: with `P_\text{nn} = 2` the singular values of $\partial\log\text{shape}/\partial\theta$ are `[184.9, 164.6, 23.1]` (rank 3, condition number 8), i.e. the temperature direction and the network directions are all constrained by the band coverage.

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
                 ├─ normalized_source_params(model, params)   ← ONCE per source:
                 │     log_amplitude -= model(lambda_0, theta)   folds the
                 │     LAMBDA_0 normalisation into the amplitude
                 └─ jax.vmap(_one_subpixel_rate) over K² sub-pixels
                      └─ _one_subpixel_rate() per sub-pixel
                           ├─ transmission.quantile(q, omega_p) → K_λ wavelengths
                           ├─ spectrum_model(wavelengths, theta) → log-flux
                           ├─ exp(log_amplitude + log_flux) → f_λ
                           ├─ psf(omega_p, omega_s, wavelengths) → PSF vals
                           └─ jnp.mean((λ/HC) * f_λ * PSF_val) * norm
       └─ scatter-add stamps into full image
```

Note the placement of the fold: it is the *only* step that needs $\lambda_0$, it depends on neither the wavelength nor the pixel, and it happens once per source. Putting it inside the spectrum model would re-run it for every sub-pixel of every source.

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
- **One amplitude least-squares solve runs BEFORE the first SGD step**, then one every `amp_solve_every` steps (default **16**), then one final solve after the loop (cf. §5.6). The initial solve matters more than it looks: the drawn amplitudes are uniform in log A, i.e. wrong by an arbitrary factor, so the first loss is enormous and every early gradient — including the per-parameter statistics `optax.scale_by_rms` accumulates from them — is dominated by amplitude error rather than by the spectral shape. Because the model is exactly linear in amplitude the solve is exact, so it costs one CG solve (the solver is built once and reused) and leaves the best possible starting point for the amplitude block. `opt_state` is rebuilt afterwards so the momentum/RMS state belongs to the solved parameters. Measured on the mock's defaults (P = 3, 16 sources, 6 exposures): $\chi^2/\text{pixel}$ $3.4\times10^5 \rightarrow 1.06$ before any SGD step.
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
| `SPECTRUM_KIND` | `"nn"` | Shape template: `"blackbody"`, `"nn"` or `"blackbody+nn"` (§6.4) |
| `NN_N_PARAMS` | 1 | Number of shape parameters θ of the neural network |
| `NN_N_HIDDEN_LAYERS` | 1 | Hidden layers of the neural network (each modulated by its own FiLM branch) |
| `NN_HIDDEN_SIZE` | 32 | Neurons per hidden layer |
| `NN_FILM_HIDDEN_LAYERS` | 1 | Hidden layers inside each FiLM branch (§3.2.1) |
| `NN_FILM_SIZE_FACTOR` | 1.0 | FiLM branch width as a multiple of `len(theta)`, so θ is never bottlenecked |
| `NN_LAYER_NORM` | `False` | Per-wavelength LayerNorm after every hidden activation (§3.2.2) |
| `NN_SEED` | 314159 | Seed for the (frozen) random weights |
| `NN_TARGET_LOG_SHAPE_STD` | 0.5 | Target *per-source* (over wavelength) spread of the rescaled log-shape |
| `NN_RESCALE_SAMPLES` | 256 | Draws used to calibrate the rescaling (a 0.99 quantile needs enough samples to be meaningful) |
| `NN_RESCALE_TAIL_QUANTILE` | 0.99 | Quantile of the per-source spread that the rescaling caps (§7.19) |
| `NN_MAX_LOG_SHAPE_STD` | 4.0 | Ceiling for that quantile — bounds the tail so `exp` can never overflow to `inf`/`NaN` |
| `NN_PRIOR_STD` | 0.35 | Width of the network's θ prior, `θ ~ N(0, σ²)` — sets the *tail* of the per-source spread |
| `NN_INIT_STD` | `= NN_PRIOR_STD` | Width of the initial-guess θ draw |

### 6.2 Pipeline steps

1. **Amplitude range** (`_report_photon_counts`): amplitudes are specified *directly* on the physical $f_\lambda(\lambda_0)$ scale via the `AMPLITUDE_MIN`/`AMPLITUDE_MAX` constants — there is no photon-count *targeting*. The implied detected photon counts are only *reported*, by integrating the active model's shape (at its reference shape parameters — 3000 K for the blackbody, θ = 0 for the network) over each band with the Gaussian filter, times `EXPOSURE_TIME`. The same reference shape sets the background level (`_compute_background`), so both diagnostics track the active model automatically.

**Spectrum preview (before any image is generated).** Step 1 is followed by `plots/spectra_preview.svg`: `n_spectra = 8` shapes drawn from the active model's prior overlaid on one semilog-y axes, all in the *same* colour (the point is the family of curves, not telling them apart), with the band boundaries marked so it is obvious which part of each curve the six bands sample. It needs nothing but the model, so it costs milliseconds and runs **before** the expensive image generation — a model that is too flat (little spectral information to recover), too wiggly (poor `n_lambda = 1` quadrature), or whose prior range is mis-set is visible at a glance, and it is the quickest way to see what the `--nn-*` options actually do. The curves are the dimensionless *shapes*, normalised to 1 at `LAMBDA_0`; amplitudes are drawn separately from a ~4-decade power law and would swamp the figure without adding shape information (their range is reported by `_report_photon_counts`).

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

7. **Diagnostic plots**: loss history (asinh y-scale), true-vs-recovered scatter, predicted/residual images, — for a flexible model — a true-vs-recovered *spectrum* overlay (`plots/spectra_comparison.svg`, see §6.4), and **one spectrum figure per selected source** (`plots/source_spectrum_{NN}.svg`, see §6.5).

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
python scripts/mock_spherex_images.py --spectrum blackbody+nn  # continuum + modulation
python scripts/mock_spherex_images.py --spectrum nn --nn-shape-check
python scripts/mock_spherex_images.py --spectrum nn --nn-params 3 --nn-layers 2 \
    --nn-hidden 16 --nn-film-layers 1 --nn-film-size-factor 2.0 --nn-layer-norm
```

With `--spectrum blackbody+nn` the model is a `BlackbodyPlusNNSpectrum` (§3.2.3): $\theta_0$ is the blackbody's `log T` and the remaining `NN_N_PARAMS` parameters go to the network part, which every `NN_*` hyperparameter configures exactly as it does for `--spectrum nn`. The mock's prior, initial guess, labels and reference parameters all treat column 0 as `log T` (see `_model_blocks`), and the network's output rescaling is applied to the network part before the composite wraps it.

`plots/spectra_preview.svg` (§6.2) is the quickest way to see what these flags do: it is drawn from the model's prior before any image is generated.

θ reaches the network through **FiLM** modulation (§3.2.1), so the shape hyperparameters include the two FiLM ones: `--nn-film-layers` (hidden layers inside each branch; `0` = a single linear map) and `--nn-film-size-factor` (branch width as a multiple of `len(theta)`, i.e. the knob that decides how much of θ can reach γ and β). `--nn-layer-norm` / `--no-nn-layer-norm` additionally inserts a per-wavelength LayerNorm after every hidden activation (§3.2.2); it makes a random net much less flat — at the defaults the output rescaling needs a factor ~5 instead of ~33 — but does **not** remove the need for that rescaling.

**Why so little changed.** `_build_band` gained one optional `spectrum_model` argument (§3.6), the mock builds **one** model instance and passes it to every band's generator, and everything else was already generic:

* `image.py` / `image3.py` / `psf.py` / `transmission.py` never look at the model beyond calling `spectrum_model(λ, shape_params)`.
* The amplitude solve is untouched — it only needs the model to be *linear in amplitude* and the shape to be *independent of it*, both of which `NeuralNetSpectrum` guarantees.
* SGD/LM are untouched: they differentiate only `(log_params, log_backgrounds)`.
* Parameter bookkeeping is written for `(N, 1 + P)` throughout, so `--nn-params 3` works with no further changes.

**The weights are frozen by construction.** The model is never handed to an optimiser, and the generator is a closed-over constant rather than a JIT argument — that is the *only* way the weights could become traced/differentiated values. `end_to_end_mock` hashes every leaf of the model before and after inference and reports `Spectrum model weights unchanged (frozen)`, and `tests/test_mock_spectrum_model.py` asserts the same thing so a future refactor cannot silently unfreeze it.

**Why the output layer is rescaled.** A randomly initialised MLP produces an arbitrary spectrum: the raw network's λ₀-normalised log-shape spread is ~1/50 of the blackbody's, so left alone the frozen net would give near-featureless spectra and the directly specified `AMPLITUDE_MIN`/`AMPLITUDE_MAX` range, the background level and the resulting detection S/N would no longer correspond to the blackbody run. `_rescale_nn_output` therefore scales the final layer's weight block so a TYPICAL source's log-shape has standard deviation `NN_TARGET_LOG_SHAPE_STD = 0.5`. This is *exact* rather than iterative: the shape is `exp(raw(λ;θ) − raw(λ₀;θ))`, which is linear in the final weight block (the final **bias cancels** in the difference, because it does not depend on λ *or* θ), so scaling that block by `f` scales the log-shape by exactly `f`. In practice the factor is ~30–50 — smaller (~5) with `--nn-layer-norm`, i.e. the LayerNorm of §3.2.2 makes a random net much less flat, but it still leaves the *statistic* to be pinned by this rescale rather than guaranteeing it.

The factor is **capped** as well, so that the `NN_RESCALE_TAIL_QUANTILE` (0.99) quantile of the per-source spread cannot exceed `NN_MAX_LOG_SHAPE_STD = 4.0`, and θ itself is drawn from a narrow prior (`NN_PRIOR_STD = 0.35`). Matching the *median* alone leaves the tail free, and the tail is what breaks a run: see §7.19 and §7.21.

The statistic matched is the **per-source** spread — the std over wavelength for one drawn θ, then the median over draws — and not the spread *pooled* over (θ, λ). The pooled number also contains the spread *between* sources, which is a property of the θ prior; because FiLM modulates hidden activations multiplicatively, that between-source spread can dominate, so a run could hit a pooled target while every individual spectrum stayed nearly flat — precisely the failure the rescaling exists to prevent. `nn_shape_check` reports the per-source statistic for the active model **and** for a reference blackbody, so the target constant can be judged from data: at the mock's defaults the FiLM network lands at 0.51 (target 0.5), while a 3–8 kK blackbody gives 1.75 — it varies 3.4× more across the full 0.75–5 µm range, although its *in-band* spread is smaller (0.27 vs 0.38), which is what the per-band columns of the shape check show.

**`--nn-shape-check` (run this first).** A flexible model can fail *before* any fitting happens — either because θ does not move the in-band shape (θ is then unidentifiable however long you fit) or because the frozen net's spectra span too many decades to be comparable to the blackbody run. The check answers both:

* per-band log-shape range over draws from the prior, e.g. at the mock's defaults (seed 314159, 1×32, FiLM 1×1, 8 embeddings) band 1 `[-0.55, +1.03]` and band 6 `[+0.51, +2.13]` (medians);
* the per-source vs pooled log-shape spread, with the blackbody reference row described above;
* the singular values of $\partial \log \text{shape} / \partial \theta$ on the sampled wavelength grid: near-zero singular values are directions that leave the spectrum unchanged. That configuration gives `[54.7]` for `P = 1` (rank 1, condition number 1) — comfortably non-degenerate; an earlier pre-FiLM seed with a 3x32 network gave `[31.8, 8.1, 5.2]` at `P = 3` (rank 3, condition number 6.1), so $\theta$ is well constrained by the band coverage at both settings — but at `P = 3` the parameters are near the information limit of six bands, so treat large `P` as a test of the plumbing rather than as a physically meaningful model. Note that with FiLM, a *degenerate* `[0]` is exactly the signature to fear (all $\gamma \approx 0$: the shape stops depending on wavelength at all);
* `shape(λ; θ)` for 64 prior draws, plus the implied photon counts. The grid is log-spaced (see `_wavelength_grid`), because the Fourier embedding is log-periodic and a linear grid would under-sample its finest features at short wavelengths.

**Reading the results.** Since a random net's θ has no physical meaning, `plots/comparison.svg` (θ vs θ) is only a parameter-recovery check; the interpretable diagnostic is `plots/spectra_comparison.svg`, which overlays the true and recovered *spectra* for the brightest sources and reports the median $|\Delta \log \text{shape}|$ over them. In the small-scale smoke runs (`N_SOURCES = 12`, 4 exposures, 48 SGD steps), both models reach $\chi^2/\text{pixel} = 0.9932$ at the *true* parameters and 0.9933–0.9948 at the recovered ones, with a median $|\Delta \log \text{shape}|$ of ~0.03–0.31 dex for the few bright sources.

**Cost.** The kernel is evaluated at every wavelength sample of every sub-pixel of every source, and an MLP costs far more arithmetic per point than the closed-form Planck function, so this slows the forward model *and* each interleaved amplitude solve (which rebuilds the postage stamps). Two things keep the overhead modest:

* The caller supplies the $\lambda_0$ value, so the MLP runs exactly `n_wavelength_samples` times per sub-pixel instead of `n_wavelength_samples + 1`. Measured on one band-3 stamp (1764 sub-pixel entries, `n_lambda = 1`), the model-side cost drops from 5.01 ms to 2.44 ms — **2.05× fewer MLP evaluations** (1.55× at `n_lambda = 5`, as expected from $(N_\lambda+1)/N_\lambda$).
* The FiLM branches are computed **once per call**, not per wavelength sample, so they cost a few percent of the main MLP.
* That work is small next to the rest of the per-sub-pixel cost (quantile sampling, PSF evaluation, sub-pixel bookkeeping, scatter-add). Measured on image generation **alone** (band 3, 57 sources, 128² detector, `n_lambda = 1`, steady state): 146 ms for the blackbody, **199 ms (+36%)** for the FiLM network without Fourier embeddings and **202 ms (+38%)** with the default 8 embeddings. At `n_lambda = 5` the same models take 152 / 248 / 292 ms, i.e. the NN's extra cost grows with the wavelength sample count because it is per-sample, while the blackbody's does not.

**Quadrature is the price of a flexible shape.** A bandpass integral taken with `n_lambda = 1` is exact only for a shape that varies slowly across the filter. Fourier features deliberately allow faster variation, so the forward model's fidelity is model-dependent — measured as the relative RMS of a band-3 image between `n_lambda = 1` and `n_lambda = 63` (normalized by the image's L2 norm), it is ~5e-6 for a 3–8 kK blackbody, ~1e-6 at `n_embeddings = 0`, ~6e-6 at 2, ~3e-5 at 4 and ~2e-3 at the default 8 — the last being a **28% error in total flux**, against 0% for the blackbody. Routing θ through FiLM did not change these materially: the modulation changes how strongly the shape bends, not which *frequencies* the embedding can express, so `n_embeddings = 8` was always the outlier. This does *not* bias a mock run, because generation and fitting share the same `n_lambda`; it does mean the simulated bands stop being faithful band-integrated fluxes, so `--compare-n-lambda` should be re-checked after any change to `n_embeddings`, and either that or `N_LAMBDA` adjusted (raising `N_LAMBDA` multiplies the MLP cost by the same factor, so lowering `n_embeddings` is usually the cheaper lever — at `n_embeddings = 4` the flux error is already back to +0.4%).

### 6.5 Per-source spectrum figures (`plots/source_spectrum_{NN}.svg`)

The global diagnostics (§6.2 step 7) answer "did the fit work in aggregate?". These figures answer "what does the fit *do* to one source?" — one standalone figure per source, named by its **global catalog index** (`{:02d}` is a minimum field width, so source 7 gives `source_spectrum_07.svg` and a large catalog gives `source_spectrum_127.svg`).

**Which sources.** The `n_top = 8` highest-amplitude *bright* sources — ranked by the **true** amplitude, so the selection cannot be circular — plus `n_random = 8` further bright sources drawn without replacement from what remains (seeded, so the file set is reproducible). Only sources in the bright subsample (§6.2 step 7) are eligible. Degenerate cases are handled rather than special-cased: a catalog with fewer than 16 bright sources simply yields fewer files, and with no bright sources at all the step is skipped with a message. See `_select_plot_sources`.

**What is in a figure.**

* the **true** spectrum $f_\lambda(\lambda; \theta_\text{true})$ and the **recovered** spectrum $f_\lambda(\lambda; \hat\theta)$ over the full 0.75–5 µm range, on log–log axes;
* one **observation point per exposure** the source appears in, placed at that exposure's bandpass **central wavelength at the source's own position**, $\lambda_c(y)$ — the linear variable filter ramps along $y$, so a source observed several times is sampled at a different $\lambda_c$ each time, which is exactly what gives it its multi-band coverage. The point's height is the **recovered** model flux $\exp(\widehat{\log A})\, \texttt{normalized\_shape}(\lambda_c; \hat\theta)$, so any vertical offset from the true curve is a recovery error and not a plotting artifact;
* a **vertical error bar** $\pm f_\text{true}(\lambda_c) / (\text{S/N})$: for a matched-filter measurement the *fractional* flux uncertainty is $1/\text{S/N}$, so this builds the uncertainty out of exactly what is available — the noise level, the PSF, and the true flux — using the same matched-filter S/N defined in §6.2 step 7;
* a **horizontal bar** of width $\sigma_\lambda$ (the bandpass sigma), because the point constrains an average over that window rather than the flux exactly at $\lambda_c$;
* band boundaries with band numbers, the $\lambda_0$ anchor, and a text box with the true vs recovered shape parameters.

Points are coloured per band and **filled** when detected (S/N > `SNR_THRESHOLD`) and **hollow** when not, so an unconstraining observation is visible as such instead of being silently dropped.

**Two edge cases, and why they are drawn differently.**

1. *The stamp falls off the detector.* `_filter_sources` deliberately keeps sources within `half_stamp` of the detector edges, so a source can be in an exposure's source list while contributing no measurable flux. Its matched-filter S/N is then a numerical zero (~1e-11 – 1e-6) rather than a small detection, and $f_\text{true}/\text{S/N}$ would be ~$10^{10}$ times the flux. Such an exposure is **not an observation** and is skipped (`SNR_FLOOR = 1e-3`, some six decades below the faintest real catalog source and well above the numerical zeros).
2. *A partially clipped stamp.* A source just off the edge, or with a narrow PSF, can keep a small but real overlap, giving a legitimate but tiny S/N (e.g. 0.01 for a source that is bright in its other bands). The point is **kept** — it is a real measurement, just an unconstraining one — and its bar, which is far taller than the axes, is **truncated to the axis range** and marked with an **arrow head pointing the way the bar ran off**. Without that marker a full-height thin line is easily mistaken for a band boundary.

**Cost.** The S/N sweep is the expensive half of these diagnostics (one forward model per exposure), so it is computed **once** per run, in `_per_exposure_source_snr`, and shared between `_compute_bright_mask` and `plot_source_spectra` instead of sweeping twice. `plot_source_spectra` accepts the precomputed list as `snr_per_exposure` and only falls back to computing it itself when called standalone.

**Note on `PLOTS_DIR`.** `_source_spectrum_fname` and `plot_source_spectra` take `out_dir=None` and resolve it to the module-level `PLOTS_DIR` *inside the body*. Binding it as a default argument (`out_dir=PLOTS_DIR`) captures the directory at import time, so a driver that patches `mock.PLOTS_DIR` would still write to `plots/` — which is exactly the bug this signature avoids.

The one deliberately unoptimised factor that remains is that the shape is evaluated separately for every pixel *column* of a stamp even though the wavelength samples depend only on the row — see §7.20 for why, and for what it would take to remove it.

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

**A median is not a bound.** Matching a *typical* source is only half the job, because the θ→shape map is steeply nonlinear: the per-source spread is heavy-tailed, and one global factor multiplies that tail too. Measured on the mock's 4×16 net, the spread's maximum over a few hundred draws was ~170× its median, i.e. a log-shape range of ~80 (36 decades of flux). At that point `exp` overflows float32 (limit ≈ 88.7), `inf` meets a zero PSF or mask value and becomes `NaN`, and every parameter set that touches that source has a `nan` χ² — the whole failure chain is dissected in §7.21. Two measures now bound it:

* the **prior width** is narrow (`NN_PRIOR_STD = 0.35`, and the initial guess uses the same width), because the spread grows steeply with |θ|: max/median falls from ~170 at a unit-width prior to ~40 at 0.35, and the rescaling re-normalises the median either way, so only the absurd spectra are removed; and
* the factor is **capped** so a high quantile of the spread cannot exceed `NN_MAX_LOG_SHAPE_STD = 4.0` (`NN_RESCALE_TAIL_QUANTILE = 0.99`, calibrated from `NN_RESCALE_SAMPLES = 256` draws — a 0.99 quantile needs enough samples to mean anything).

The median target still sets the factor whenever the ceiling allows it, so a well-behaved net is unchanged (a 6×32 net lands on the full 0.5 median; the heavy-tailed 4×16 one is cap-limited to ≈0.2). Worst-case `log f_λ` at the mock's brightest amplitude (float32 `exp` overflows at 88.7): **292 before, −3 for the 4×16 net and +10 for a 6×32 net** after.

Second, a flexible model can be unfittable regardless of the optimiser: if a direction in θ leaves the in-band shape unchanged, that parameter is unidentifiable *in principle*, and a poor recovery is not evidence of a convergence problem. Check it before fitting by looking at the singular values of $\partial \log \text{shape}/\partial\theta$ over the observed wavelengths (unit-independent, because the log-shape is dimensionless): near-zero singular values are the null directions. Practically, a parameter count approaching the number of distinct bands should be treated as a plumbing test, not a physical model.

Also worth remembering when *interpreting* the fit: with a random network, θ has no physical meaning, so the θ-vs-θ scatter is only a parameter-recovery plot; the scientifically meaningful diagnostic is whether the recovered *spectrum* matches the true one (`plot_spectra_comparison`), and its error should be quoted in dex of log-shape rather than as a parameter offset.

### 7.20 Where the `LAMBDA_0` normalisation lives — and why not a shared wavelength grid

The spectrum models return an unnormalised log-flux and know nothing about `LAMBDA_0`; the caller anchors them by folding the constant into the amplitude **once per source** (§2.1, §4). Two decisions behind that are worth recording.

**Log space, not linear space.** If the models returned an unnormalised *linear* kernel, the caller would have to compute `exp(raw(λ)) / exp(raw(λ₀))`. A randomly initialised network's raw output carries an offset that is (i) independent of wavelength, (ii) unconstrained by the data — an exact null direction of the likelihood, so θ can random-walk along it — and (iii) potentially large, because the output rescaling of §6.4 scales only the final *weight* block and leaves the final *bias* (which is precisely that offset) untouched. The ratio could therefore overflow float32 to `inf/inf`. Exponentiating a *difference* in log space cannot: the difference is bounded by the shape's own dynamic range. It also removes the `exp` from the model entirely, and drops the number of `λ₀` evaluations from one per sub-pixel to one per source.

**Why the shape is still evaluated per sub-pixel rather than per sub-pixel row.** The bandpass central wavelength is `λ_c(y) = λ_intercept + λ_slope·y`, so the quantile wavelengths used for entry `[i, j, v, u]` of a stamp depend only on `(i, u)`: a stamp of size `S` has just `S·K` distinct λ vectors, while the shape is evaluated `S²·K²` times — a 42× (band 1) to 186× (band 6) redundancy at the mock's defaults. Collapsing it (evaluate the shape on those distinct rows, broadcast over the column axis) is exact *for a purely y-dependent calibration* — but the real SPHEREx wavelength calibration has a slight x-dependence, so each sub-pixel keeps its own quantile evaluation. The cost of that correctness is the factor above.

This refines rather than contradicts the frozen note in `image2.py`: that note rules out hoisting over the *whole* stamp (different rows really do have different `λ_c`), whereas the row collapse only shares λ across the column axis. If the remaining cost ever matters, the exact fix is to evaluate the shape once on the union of quantile wavelengths actually used by the stamp and gather into place; the approximate fix is to tabulate the shape per source on a λ grid and interpolate — the shape depends only on λ, so the calibration's x-dependence does not invalidate that route.

### 7.21 An unbounded exponent is a `NaN`, not an `inf` — and `NaN` is contagious

A model that runs away numerically should look like an absurdly large number, not like poison. Here it was poison, and the symptom was actively misleading: the mock printed `Loss (chi^2 / pixel) at TRUE parameters: 1.0002` and then `... at INITIAL GUESS: nan`, which reads like an optimiser or a data problem. It is neither — it is a forward-model failure, and the two evaluations differ only in θ.

The chain, once measured (§7.19 has the numbers):

1. `_rescale_nn_output` set its factor from the **median** per-source log-shape spread, which pins a typical source and leaves the tail free. At the old unit-width θ prior a few draws per thousand reached a log-shape range of ~80.
2. `f_lam = exp(log_amplitude + log_flux)` in `_one_subpixel_rate` therefore overflowed float32 — `exp` returns `inf` above ≈88.7 — **in the bands whose sampled wavelengths happened to probe the wild part of that draw's shape** (measured: the wildest init rows gave `inf` pixels in bands 3–6 and `NaN` pixels in band 3).
3. `inf` alone would merely be large (the rate is `inf`, χ² is `inf`) *until* it meets an exactly zero factor: a PSF value, a masked postage-stamp element, or a zero sub-pixel weight. `inf * 0 = NaN` in IEEE arithmetic, so one such pixel makes χ² `NaN` for **every** parameter set containing that source, permanently. `inf` is recoverable; `NaN` is not, and gradient clipping does not help because the gradient is `NaN` too.
4. Which source hits it depends on the draw, so a run can pass its true-parameter sanity check and fail at the initial guess — the catalog draw was tame, the init draw was not. That is the most confusing possible presentation of a deterministic bug.

The origin is the *scaling* of §7.19: the heavy tail of the θ→shape map, multiplied by a single median-calibrated factor of ~4000. Two layers of defence now exist, and they are deliberately different in kind:

* **Bounds on the distribution** (`NN_PRIOR_STD = 0.35` and the `NN_MAX_LOG_SHAPE_STD = 4.0` ceiling, §7.19) keep the parameters in a region where the exponent never approaches the float32 limit. This is the actual fix: the worst-case `log f_λ` goes from +292 to −3 (4×16 net) or +10 (6×32 net) against a limit of 88.7.
* **A hard bound on the exponent in the library** — `MAX_LOG_FLUX = 40` in `spherex/image.py::_one_subpixel_rate`, applied as `exp(min(log_flux, MAX_LOG_FLUX))`. 40 comes from the arithmetic of the pixel rate, which multiplies `exp(log_flux)` by `(λ/hc)·PSF·∫T ≲ 1e19`: the integrand then stays below ~1e36, inside float32, while never touching a physical source, since the mock's brightest amplitude is `5e-11 W m⁻² µm⁻¹` (log-flux ≈ −24) and a typical shape contributes O(1). Only the upper end is clipped; a very negative log-flux harmlessly underflows to 0. The guard turns an unrecoverable `NaN` into a bounded, visible error — a seat belt, not the fix.

**What the guard does not do.** It bounds the *exponent*, so it cannot rescue a network that already returns `inf`/`NaN` from its own arithmetic (a θ astronomically far outside the prior can saturate a FiLM scaling inside a hidden layer), and it cannot make χ² finite — an absurd θ still gives a huge or `inf` χ². It is also a *clipping*, so its gradient is exactly zero above the bound: a source pinned there cannot be pulled back by an optimiser. That is why the distribution-level measures, not this constant, are what make the mock safe; the constant only makes any residual failure loud and local instead of silent and global.

**How it would have been caught sooner.** A pre-flight report of the *tail* (not the median) of the per-source spread, together with the implied worst-case exponent at the brightest amplitude, would have flagged the configuration before the first image was generated. Note the two sanity checks that exist did pass: the true-parameter χ² was 1.0002, and `nn_shape_check` reports the *median* per-source spread (0.51 vs a 0.5 target) — a statistic that is blind to the tail by construction.

---

## 8. File Map

| File | Purpose |
|---|---|
| `spherex/constants.py` | Physical constants in codebase-native units (kJ, µm, arcsec) |
| `spherex/spectrum.py` | `BlackbodySpectrum` (analytic), `NeuralNetSpectrum` (MLP with FiLM conditioning and an optional LayerNorm) and `BlackbodyPlusNNSpectrum` (continuum + modulation composite) — unnormalised log-flux templates — plus the `[log_amplitude, shape…]` layout helpers and the caller-side `LAMBDA_0` anchors (`reference_log_flux`, `normalized_log_shape`, `normalized_shape`, `normalized_source_params`) |
| `spherex/psf.py` | `GaussianPSF` — wavelength-dependent Gaussian PSF |
| `spherex/transmission.py` | `GaussianFilterTransmission` — LVF transmission with quantile interface |
| `spherex/image.py` | `_one_subpixel_rate` (quantile integration; applies `exp(log_amplitude + log_flux)` for pre-normalised params, with the exponent bounded by `MAX_LOG_FLUX`, §7.21), `photon_rate_per_pixel` (normalises once per source), `ImageGenerator` (scan-based) |
| `spherex/image3.py` | `ImageGenerator3` (chunked-batch, uses `_one_subpixel_rate` from `image.py`; folds the `LAMBDA_0` normalisation once per source in `_stamp_batch`), plus `source_stamps` for the amplitude-solve preconditioner |
| `spherex/config.py` | `SpherexImageGenerator`, `SpherexImageGenerator3` — pre-configured per-band generators, `_build_band(..., spectrum_model=None)` (the spectrum-model injection point), band table |
| `spherex/__init__.py` | Public API exports |
| `scripts/inference.py` | `infer_parameters` (SGD, with amplitude interlace), `infer_parameters_lm` (LM), `solve_log_amplitudes` (direct amplitude least-squares), `compute_loss`, `compute_lm_loss`, `plot_loss_history`, `plot_comparison` (one panel per parameter, `1 + P`), residual functions, custom LM solver |
| `scripts/mock_spherex_images.py` | End-to-end mock: spectrum-model factory + shape-parameter bookkeeping (Step 0), catalog, exposures, image generation, hybrid SGD+LM inference, diagnostics (`--spectrum {blackbody,nn}`, `--nn-shape-check`), per-source spectrum figures (§6.5: `_per_exposure_source_snr`, `_observed_central_wavelengths`, `_select_plot_sources`, `_source_observation_points`, `_plot_one_source_spectrum`, `plot_source_spectra`), benchmarks |
| `tests/test_image3.py` | Numerical tests verifying ImageGenerator3 ≡ ImageGenerator |
| `tests/test_mock_spectrum_model.py` | Model factory (determinism, output rescaling), generic `(N, 1 + P)` bookkeeping, generator injection, the frozen-weights guarantee, and the per-source spectrum diagnostics (source selection, `λ_c` from the LVF, the `f/S/N` bar, off-detector observations, one file per source) |
| `pyproject.toml` | Dependencies: `jax`, `equinox`, `optimistix`, `lineax`, `numpy` |
| `/memories/jax_optimization_lessons.md` | Persistent notes on LM trust-region tuning and memory safety |
