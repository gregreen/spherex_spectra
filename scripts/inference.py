"""Infer source parameters from SPHEREx exposures via gradient descent.

Provides ``infer_parameters`` which optimises log-temperature,
log-amplitude and log-background using preconditioned SGD (gradient
clipping + RMS-scaling + momentum) with a warmup-cosine learning-rate
schedule.  The entire training step (loss, gradient, and optimiser update)
is compiled into a single ``jax.jit`` function, analogous to wrapping a
whole training step in ``tf.function``.  Also includes plotting utilities
for loss history and true-vs-recovered comparisons.

Loss convention
----------------
The reported loss is chi^2 *per pixel*, computed as::

    loss = (sum of squared normalised residuals over ALL pixels
             in ALL exposures) / (total number of pixels in ALL exposures)

This is important: summing *per-exposure means* (as an earlier version of
this module did) implicitly upweights small exposures and downweights large
ones, since each exposure's mean gets equal weight regardless of its pixel
count.  Normalising by the *global* pixel count instead means every pixel
contributes equally, and the loss is directly interpretable
(~1 at convergence, >>1 indicates a poor fit, <<1 indicates a bug, e.g.
over-counting degrees of freedom).
"""

import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import optax

EXPOSURE_TIME = 15.0             # s

# Sentinel pixel position used to pad the source arrays of an exposure up to
# a group's common source count.  Placed far outside any realistic detector
# so that the postage-stamp ``valid`` mask (see ``spherex.image``) always
# excludes it, contributing exactly zero flux and zero gradient.
_PAD_POSITION = -1.0e6


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _per_exposure_sumsq(mean_img, sigma_img, positions, log_params_global,
                        source_idx, gen_module, log_background, half_stamp,
                        n_lambda, oversampling):
    """Sum of squared normalised residuals for one exposure.

    NOT normalised by pixel count - callers must sum this across all
    exposures and divide by the *total* pixel count over all exposures to
    obtain a properly weighted chi^2 / pixel (see module docstring).
    """
    lp_exp = log_params_global[source_idx]  # select sources in this exposure
    pred = gen_module(
        positions,
        lp_exp,
        postage_stamp_half_size=half_stamp,
        n_wavelength_samples=n_lambda,
        oversampling=oversampling,
    )
    # pred/background are photon RATES (s^-1); scale both by EXPOSURE_TIME
    # to get expected counts, matching the data-generation convention in
    # mock_spherex_images.py (previously only the background was scaled,
    # which under-counted the source flux relative to the background).
    pred_bg = (pred + jnp.exp(log_background)) * EXPOSURE_TIME
    diff = (mean_img - pred_bg) / sigma_img
    return jnp.sum(diff ** 2)


# ---------------------------------------------------------------------------
# Per-band-grouped batched loss (Part B)
# ---------------------------------------------------------------------------

def _group_exposures_by_band(exposures):
    """Group exposure indices by the identity of their ``gen_module``.

    Exposures sharing the same generator instance (i.e. the same SPHEREx
    band) are grouped together, since only they can safely share a common
    postage-stamp half-size and be processed by a single ``lax.scan``
    without padding across bands (different bands legitimately need
    different stamp sizes, since PSF FWHM grows with wavelength).

    Returns
    -------
    dict : id(gen_module) -> list of exposure indices (into ``exposures``)
    """
    groups = defaultdict(list)
    for k, exposure in enumerate(exposures):
        gen_mod = exposure[3]
        groups[id(gen_mod)].append(k)
    return groups


def _pad_source_arrays(positions, src_idx, s_max):
    """Pad ``positions`` (S, 2) and ``src_idx`` (S,) up to ``s_max`` sources.

    Padding uses an off-detector sentinel position so that the padded
    sources contribute exactly zero flux (and zero gradient) via the
    existing postage-stamp ``valid`` mask.
    """
    s = positions.shape[0]
    n_pad = s_max - s
    if n_pad == 0:
        return positions, src_idx
    pos_pad = jnp.full((n_pad, 2), _PAD_POSITION, dtype=positions.dtype)
    idx_pad = jnp.zeros((n_pad,), dtype=src_idx.dtype)
    positions = jnp.concatenate([positions, pos_pad], axis=0)
    src_idx = jnp.concatenate([src_idx, idx_pad], axis=0)
    return positions, src_idx


def _prepare_band_group(exposures, group_idx):
    """Python-side (untraced) padding/stacking for one band group.

    Returns a dict of stacked static arrays plus the shared ``gen_module``
    and group-local postage-stamp half-size, ready to be ``lax.scan``-ed
    over inside a jitted loss function.
    """
    gen_mod = exposures[group_idx[0]][3]
    half_stamp_group = max(exposures[k][4] for k in group_idx)
    s_max = max(exposures[k][5].shape[0] for k in group_idx)

    mean_imgs = jnp.stack([exposures[k][0] for k in group_idx])
    sigma_imgs = jnp.stack([exposures[k][1] for k in group_idx])

    positions_list = []
    src_idx_list = []
    for k in group_idx:
        _, _, positions, _, _, src_idx = exposures[k]
        pos_p, idx_p = _pad_source_arrays(positions, src_idx, s_max)
        positions_list.append(pos_p)
        src_idx_list.append(idx_p)

    return {
        "group_idx": group_idx,
        "gen_mod": gen_mod,
        "half_stamp": half_stamp_group,
        "mean_imgs": mean_imgs,
        "sigma_imgs": sigma_imgs,
        "positions": jnp.stack(positions_list),
        "src_idx": jnp.stack(src_idx_list),
    }


def _group_sumsq(group, log_params, log_backgrounds_group, n_lambda, oversampling):
    """Sum of squared normalised residuals over all exposures in one
    band group (via a single ``jax.lax.scan``)."""

    def body(carry, xs):
        mean_img, sigma_img, positions, src_idx, log_bg = xs
        s = _per_exposure_sumsq(
            mean_img, sigma_img, positions, log_params, src_idx,
            group["gen_mod"], log_bg, group["half_stamp"],
            n_lambda, oversampling,
        )
        return carry + s, None

    total, _ = jax.lax.scan(
        body, jnp.array(0.0),
        (group["mean_imgs"], group["sigma_imgs"], group["positions"],
         group["src_idx"], log_backgrounds_group),
    )
    return total


# ---------------------------------------------------------------------------
# Flat residual vector (for Levenberg-Marquardt / Gauss-Newton, Part D)
# ---------------------------------------------------------------------------

def _per_exposure_diff(mean_img, sigma_img, positions, log_params_global,
                       source_idx, gen_module, log_background, half_stamp,
                       n_lambda, oversampling):
    """Flattened (pre-square) normalised residual vector for one exposure.

    Same forward model/convention as :func:`_per_exposure_sumsq`, but
    returns the residual array itself (flattened to 1-D) rather than the
    summed square - needed as the ``fn(y, args) -> residuals`` callable
    expected by ``optimistix.least_squares``.
    """
    lp_exp = log_params_global[source_idx]
    pred = gen_module(
        positions,
        lp_exp,
        postage_stamp_half_size=half_stamp,
        n_wavelength_samples=n_lambda,
        oversampling=oversampling,
    )
    pred_bg = (pred + jnp.exp(log_background)) * EXPOSURE_TIME
    diff = (mean_img - pred_bg) / sigma_img
    return diff.reshape(-1)


def _group_diffs(group, log_params, log_backgrounds_group, n_lambda, oversampling):
    """Flattened (pre-square) normalised residual vector for all exposures
    in one band group (via a single ``jax.lax.scan``, stacking each
    exposure's residual array then flattening)."""

    def body(carry, xs):
        mean_img, sigma_img, positions, src_idx, log_bg = xs
        d = _per_exposure_diff(
            mean_img, sigma_img, positions, log_params, src_idx,
            group["gen_mod"], log_bg, group["half_stamp"],
            n_lambda, oversampling,
        )
        return carry, d

    _, diffs = jax.lax.scan(
        body, None,
        (group["mean_imgs"], group["sigma_imgs"], group["positions"],
         group["src_idx"], log_backgrounds_group),
    )
    return diffs.reshape(-1)


def _make_residual_fn(exposures, n_lambda, oversampling):
    """Build ``residuals(params) -> flat residual vector`` over the ENTIRE
    exposure stack simultaneously (all bands/exposures jointly), for use
    with ``optimistix.least_squares`` (Levenberg-Marquardt/Gauss-Newton).

    ``params`` is ``(log_params, log_backgrounds)``, exactly as used by
    :func:`infer_parameters`.  Exposures are grouped by band purely as an
    implementation efficiency (batching same-band exposures through a
    single ``lax.scan``) - all exposures still contribute to the ONE
    joint residual vector / least-squares problem; there is no patch
    decomposition or independent per-band fitting here.
    """
    band_groups = [
        _prepare_band_group(exposures, idx)
        for idx in _group_exposures_by_band(exposures).values()
    ]

    def residuals(params):
        log_params, log_backgrounds = params
        pieces = []
        for group in band_groups:
            idx_arr = jnp.asarray(group["group_idx"])
            lbg_group = log_backgrounds[idx_arr]
            pieces.append(
                _group_diffs(group, log_params, lbg_group, n_lambda, oversampling)
            )
        return jnp.concatenate(pieces)

    return residuals


def compute_lm_loss(
    exposures, log_params, log_backgrounds, n_lambda=5, oversampling=2,
):
    """Loss value that the Levenberg-Marquardt fitter sees, normalised
    to chi² / pixel, computed through the **exact same** residual function
    used internally by :func:`infer_parameters_lm`.

    This builds a fresh :func:`_make_residual_fn` (the same band-grouped,
    padded, flattened residual builder that LM optimises over), evaluates
    it at the given parameters, and returns ``sum(residuals²) / n_pixels``
    — i.e. the same quantity that :func:`compute_loss` returns, but
    computed through the LM residual path for consistency checks.

    The raw ``optimistix`` loss (printed per-step when ``verbose=True``)
    is ``0.5 * sum(residuals²)``.  To cross-reference::

        raw_optx_loss = 0.5 * n_pixels * compute_lm_loss(...)

    Parameters
    ----------
    exposures : list of (mean_img, sigma_img, positions_pix,
                         gen_module, half_stamp, source_idx)
        Same format as accepted by :func:`infer_parameters_lm`.
    log_params : (S, P) array
        Log-parameters [log(T/kK), log(A/(W/m2/um))].
    log_backgrounds : (E,) array
        Log-background (rate, s^-1) per exposure.
    n_lambda : int
        Wavelength-integration samples (must match the LM fit).
    oversampling : int
        Sub-pixel oversampling factor (must match the LM fit).

    Returns
    -------
    float
        ``chi² / n_pixels`` — ~1.0 at the true parameters if the noise
        model is correctly specified.
    """
    residuals_fn = _make_residual_fn(exposures, n_lambda, oversampling)
    log_params_j = jnp.asarray(log_params, dtype=jnp.float32)
    log_backgrounds_j = jnp.asarray(log_backgrounds, dtype=jnp.float32)
    r = residuals_fn((log_params_j, log_backgrounds_j))
    sumsq = jnp.sum(r ** 2)
    total_pixels = float(sum(
        m.shape[0] * m.shape[1] for m, *_ in exposures
    ))
    return float(sumsq / total_pixels)


# ---------------------------------------------------------------------------
# Single fully-jitted training step
# ---------------------------------------------------------------------------

def _make_loss_fn(exposures, n_lambda, oversampling, batched):
    """Build a ``jax.jit``-compiled ``(log_params, log_backgrounds) -> loss``.

    ``loss`` is chi^2 / pixel, normalised by the *total* pixel count across
    all exposures (see module docstring) - not a sum/mean of per-exposure
    means.  Shared by :func:`_make_train_step` and :func:`compute_loss`.
    """
    E = len(exposures)
    total_pixels = float(sum(m.shape[0] * m.shape[1] for m, *_ in exposures))

    if batched:
        band_groups = [
            _prepare_band_group(exposures, idx)
            for idx in _group_exposures_by_band(exposures).values()
        ]

        def _total_sumsq(log_params, log_backgrounds):
            total = jnp.array(0.0)
            for group in band_groups:
                idx_arr = jnp.asarray(group["group_idx"])
                lbg_group = log_backgrounds[idx_arr]
                total = total + _group_sumsq(
                    group, log_params, lbg_group, n_lambda, oversampling
                )
            return total
    else:
        def _total_sumsq(log_params, log_backgrounds):
            total = jnp.array(0.0)
            for k in range(E):
                mean_img, sigma_img, positions, gen_mod, hs, src_idx = exposures[k]
                total = total + _per_exposure_sumsq(
                    mean_img, sigma_img, positions, log_params, src_idx,
                    gen_mod, log_backgrounds[k], hs, n_lambda, oversampling,
                )
            return total

    @jax.jit
    def _loss_fn(log_params, log_backgrounds):
        return _total_sumsq(log_params, log_backgrounds) / total_pixels

    return _loss_fn


def compute_loss(
    exposures, log_params, log_backgrounds, n_lambda=5, oversampling=2,
    batched=False,
):
    """Evaluate chi^2 / pixel for a given set of (log) parameters.

    Useful as a sanity check: evaluating this at the *true* source and
    background parameters used to generate mock data should give a loss
    of order 1 (see module docstring) - a much larger or smaller value
    indicates a bug in the forward model or in how noise/sigma was
    generated, rather than an optimisation failure.

    Parameters
    ----------
    exposures : list of (mean_img, sigma_img, positions_pix,
                         gen_module, half_stamp, source_idx)
        Same format as accepted by :func:`infer_parameters`.
    log_params : (S, P) array
        Log-parameters to evaluate at (e.g. the true simulation values).
    log_backgrounds : (E,) array
        Log-background (rate, s^-1) per exposure to evaluate at.
    n_lambda, oversampling, batched
        Same meaning as in :func:`infer_parameters`.

    Returns
    -------
    float
        chi^2 / pixel at the given parameters.
    """
    loss_fn = _make_loss_fn(exposures, n_lambda, oversampling, batched)
    log_params = jnp.asarray(log_params, dtype=jnp.float32)
    log_backgrounds = jnp.asarray(log_backgrounds, dtype=jnp.float32)
    return float(loss_fn(log_params, log_backgrounds))


def _make_train_step(exposures, optimiser, n_lambda, oversampling, batched):
    """Build one ``jax.jit``-compiled training step.

    The returned ``train_step(params, opt_state) -> (params, opt_state, loss)``
    performs the loss/gradient computation *and* the optimiser update
    inside a single compiled program (analogous to wrapping a whole
    training step in ``tf.function``), avoiding per-operation Python
    dispatch overhead between e.g. gradient summation and the optimiser
    update.

    ``loss`` is chi^2 / pixel, normalised by the *total* pixel count across
    all exposures (see module docstring) - not a sum/mean of per-exposure
    means.
    """
    loss_fn = _make_loss_fn(exposures, n_lambda, oversampling, batched)
    value_and_grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

    @jax.jit
    def train_step(params, opt_state):
        log_params, log_backgrounds = params
        loss, (grad_params, grad_bg) = value_and_grad_fn(
            log_params, log_backgrounds
        )
        grads = (grad_params, grad_bg)
        updates, opt_state = optimiser.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    return train_step


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------

def infer_parameters(
    exposures,
    init_log_params,
    n_steps=500,
    learning_rate=1e-3,
    warmup_steps=50,
    momentum=0.5,
    n_lambda=5,
    oversampling=2,
    batched=False,
    precondition_rms=True,
):
    """Infer log(temperature), log(amplitude) and log(background) via
    preconditioned SGD.

    Parameters
    ----------
    exposures : list of (mean_img, sigma_img, positions_pix,
                         gen_module, half_stamp, source_idx)
        One tuple per exposure.  ``positions_pix`` are source positions
        in pixel coordinates (pre-computed by caller from WCS).
    init_log_params : (S, P) array
        Initial log-parameters  [log(T/kK), log(A/(W/m2/um))].
    n_steps : int
        Number of SGD iterations.
    learning_rate : float
        Peak learning rate (after warmup, before cosine decay).
    warmup_steps : int
        Number of linear warmup steps (0 = no warmup).
    momentum : float
        SGD momentum coefficient.
    n_lambda : int
        Wavelength-integration samples.
    oversampling : int
        Sub-pixel oversampling factor.
    batched : bool
        If True, group exposures by band (shared ``gen_module``), pad each
        group to a common postage-stamp half-size and source count, and
        evaluate loss/grad with one ``lax.scan`` per band group (instead
        of an unrolled loop over every exposure).  Both variants are
        compiled into the same single training-step ``jax.jit`` either
        way; ``batched=True`` mainly helps when there are many exposures
        sharing a small number of bands, by reducing the size of the
        traced/compiled program.
    precondition_rms : bool
        If True (default), precondition gradients with
        ``optax.scale_by_rms()`` before the SGD+momentum update - this
        rescales each parameter's gradient by its running RMS, giving
        Adam-like per-parameter adaptive step sizes (useful here since
        temperature, amplitude and background live on very different
        scales) while keeping a plain SGD+momentum update rule.  The
        optimiser chain is, in order: ``clip`` (bound raw gradient
        magnitude before estimating its RMS) -> ``scale_by_rms``
        (per-parameter adaptive scaling) -> ``sgd`` (momentum + the
        warmup-cosine learning-rate schedule).

    Returns
    -------
    log_params : (S, P) array
        Optimised log-parameters.
    log_backgrounds : (E,) array
        Optimised log-background per exposure.
    losses : list of float
        Loss (chi^2 / pixel) at each step.
    learning_rates : list of float
        Learning rate at each step.
    """
    # ---- initialise trainable parameters ----------------------------------
    log_params = jnp.asarray(init_log_params, dtype=jnp.float32)

    # Background init: median of each noisy image, converted from counts
    # back to a rate (s^-1) by dividing by EXPOSURE_TIME, since
    # log_background is used as a rate (see ``_per_exposure_sumsq``).
    log_backgrounds = jnp.array([
        jnp.log(jnp.maximum(jnp.median(mean_img) / EXPOSURE_TIME, 1e-6))
        for mean_img, _, _, _, _, _ in exposures
    ], dtype=jnp.float32)
    # ---- optimiser ---------------------------------------------------------
    # Linear warmup then cosine decay to 0.
    # Note: optax's decay_steps is TOTAL steps (warmup + decay); it
    # subtracts warmup_steps internally.
    warmup = min(warmup_steps, max(n_steps - 1, 0))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup,
        decay_steps=n_steps,
        end_value=0.0,
    )
    transforms = [optax.clip(1.0)]
    if precondition_rms:
        transforms.append(optax.scale_by_rms())
    transforms.append(optax.sgd(learning_rate=schedule, momentum=momentum))
    optimiser = optax.chain(*transforms)

    params = (log_params, log_backgrounds)
    opt_state = optimiser.init(params)

    train_step = _make_train_step(
        exposures, optimiser, n_lambda, oversampling, batched
    )

    losses = []
    learning_rates = []

    pbar = tqdm(range(n_steps), desc="SGD")
    for step in pbar:
        params, opt_state, loss_val = train_step(params, opt_state)

        if not jnp.isfinite(loss_val):
            print(f"  Loss became non-finite at step {step}, stopping.")
            break

        log_params, log_backgrounds = params

        losses.append(float(loss_val))
        lr = float(schedule(step))
        learning_rates.append(lr)

        pbar.set_postfix(loss=f"{float(loss_val):.4e}",
                         lr=f"{float(lr):.2e}")

    return log_params, log_backgrounds, losses, learning_rates


# ---------------------------------------------------------------------------
# Levenberg-Marquardt / Gauss-Newton inference (via optimistix)
# ---------------------------------------------------------------------------

def _make_lm_solver(rtol, atol, cg_rtol, cg_atol, cg_max_steps, initial_step_size,
                    trust_region_low_constant, trust_region_high_constant,
                    max_step_size, verbose):
    """Build a Levenberg-Marquardt solver with a CONFIGURABLE initial
    trust-region radius (damping) and a maximum trust-region radius.

    ``optimistix.LevenbergMarquardt`` hardcodes the initial trust-region
    step size to 1.0 (i.e. an initially near-undamped Gauss-Newton step) -
    this is far too aggressive for our exponentiated (``exp(log_T)``,
    ``exp(log_A)``) parameterisation when starting far from the optimum:
    a "reasonable-looking" step in log-parameter space can correspond to
    an enormous change in predicted flux, causing the very first step to
    wildly overshoot (observed: loss exploding from ~1e5 to ~1e8, then the
    inner CG linear solve returning non-finite output at a badly
    conditioned point).  Subclassing lets us start with a much smaller,
    more conservative initial step (heavier damping / closer to steepest
    descent), while keeping everything else about
    ``optimistix.LevenbergMarquardt`` (damped-Newton descent, classical
    trust-region accept/reject + grow/shrink logic) unchanged.

    The ``max_step_size`` cap prevents the trust region from growing
    unboundedly after a few accepted steps (the default growth factor 3.5
    means 0.01 → 0.035 → 0.12 → 0.43 → 1.5 → 5.3 in just 5 steps).  Once
    the trust region becomes too large the Levenberg-Marquardt damping
    (λ ~ 1/step_size²) effectively vanishes, the normal-equations matrix
    ``J^T J + λI`` becomes ``J^T J`` (which may be near-singular for
    ill-conditioned problems), and the inner CG solver spins forever
    trying to converge.
    """
    import lineax as lx
    import optimistix as optx

    class _InitStepTrustRegion(optx.ClassicalTrustRegion):
        """``ClassicalTrustRegion`` with a configurable initial step size
        (upstream hardcodes this to 1.0 - see ``_AbstractTrustRegion.init``)
        and a maximum step size cap."""

        max_step_size: float = 1.0

        def init(self, y, f_info_struct):
            del f_info_struct
            return type(super().init(y, None))(
                step_size=jnp.array(initial_step_size)
            )

        def step(self, first_step, y, y_eval, f_info, f_eval_info, state):
            new_step_size, accept, result, new_state = (
                super().step(first_step, y, y_eval, f_info, f_eval_info, state)
            )
            new_step_size = jnp.minimum(new_step_size, self.max_step_size)
            return new_step_size, accept, result, new_state

    class _DampedLevenbergMarquardt(optx.AbstractGaussNewton):
        """Same as ``optimistix.LevenbergMarquardt``, but with a
        configurable initial trust-region radius, trust-region
        grow/shrink constants, and maximum step size."""

        rtol: float
        atol: float
        norm: object
        descent: optx.DampedNewtonDescent
        search: _InitStepTrustRegion
        verbose: frozenset

        def __init__(self, rtol, atol, linear_solver, norm=optx.max_norm,
                    verbose=frozenset()):
            self.rtol = rtol
            self.atol = atol
            self.norm = norm
            self.descent = optx.DampedNewtonDescent(linear_solver=linear_solver)
            self.search = _InitStepTrustRegion(
                low_constant=trust_region_low_constant,
                high_constant=trust_region_high_constant,
                max_step_size=max_step_size,
            )
            self.verbose = verbose

    return _DampedLevenbergMarquardt(
        rtol=rtol,
        atol=atol,
        linear_solver=lx.Normal(lx.CG(rtol=cg_rtol, atol=cg_atol, max_steps=cg_max_steps)),
        verbose=frozenset({"loss", "step_size"}) if verbose else frozenset(),
    )


def infer_parameters_lm(
    exposures,
    init_log_params,
    init_log_backgrounds=None,
    n_lambda=5,
    oversampling=2,
    rtol=1e-4,
    atol=1e-6,
    max_steps=64,
    cg_rtol=1e-2,
    cg_atol=1e-4,
    cg_max_steps=None,
    initial_step_size=1.0,
    trust_region_low_constant=0.25,
    trust_region_high_constant=3.5,
    max_step_size=1.0,
    verbose=False,
):
    """Infer log(temperature), log(amplitude) and log(background) via
    Levenberg-Marquardt (damped Gauss-Newton), using ``optimistix``.

    Unlike :func:`infer_parameters` (first-order preconditioned SGD), this
    solves the nonlinear least-squares problem using the model's Jacobian
    (via matrix-free Jacobian-vector products - ``lineax.Normal(lineax.CG(...))``
    as the inner linear solve, so a dense Jacobian is never materialised), which is the
    natural, much faster-converging choice for a smooth chi^2 objective
    like this one.

    Takes EXACTLY the same ``exposures``/``init_log_params`` inputs as
    :func:`infer_parameters` (no patch/tiling logic here - that is handled
    elsewhere).  The entire exposure stack is fit SIMULTANEOUSLY (required
    to constrain source spectra from multiple bands/exposures - this
    mirrors the joint fit `infer_parameters(batched=...)` already
    performs, just with a different optimiser), and background parameters
    are fit jointly alongside the source parameters (may change if
    backgrounds become more complex in future, e.g. a Gaussian process).

    Parameters
    ----------
    exposures : list of (mean_img, sigma_img, positions_pix,
                         gen_module, half_stamp, source_idx)
        Same format as accepted by :func:`infer_parameters`.
    init_log_params : (S, P) array
        Initial log-parameters [log(T/kK), log(A/(W/m2/um))].
    init_log_backgrounds : (E,) array or None
        Initial log-background (rate, s^-1) per exposure.  If ``None``
        (default), initialises from ``log(median(image) / EXPOSURE_TIME)``,
        matching :func:`infer_parameters`.  Pass an explicit array when
        warm-starting from a previous optimisation phase (e.g. a short SGD
        pre-fit before LM takes over).
    n_lambda : int
        Wavelength-integration samples.
    oversampling : int
        Sub-pixel oversampling factor.
    rtol, atol : float
        Relative/absolute tolerance for the Levenberg-Marquardt solve's
        convergence criterion.
    max_steps : int
        Maximum number of Levenberg-Marquardt iterations.
    cg_rtol, cg_atol : float
        Relative/absolute tolerance for the inner matrix-free
        conjugate-gradient linear solve (``lineax.Normal(lineax.CG(...))``)
        used at each LM
        step to solve the damped Gauss-Newton normal equations without
        ever forming a dense Jacobian or J^T J.  CG is terminated early
        once the residual drops below these tolerances.  Unlike an exact
        linear solve, LM only needs a *direction* - tolerances looser than
        the outer LM tolerances (default 1e-2 / 1e-4) are usually
        sufficient and much faster.
    cg_max_steps : int or None
        Maximum number of CG iterations per LM step.  ``None`` (default)
        lets ``lineax`` choose the number automatically (typically the
        size of the parameter vector).  Reducing this can dramatically
        speed up early LM steps where the trust-region damping already
        limits how far we can move anyway.
    initial_step_size : float
        Initial trust-region radius (upstream ``optimistix`` hardcodes
        this to 1.0, i.e. a nearly undamped first Gauss-Newton step,
        which can badly overshoot for this exponentiated parameterisation
        when starting far from the optimum).  Smaller values start closer
        to steepest descent (heavier damping); the trust-region logic
        will grow/shrink it adaptively from there.
    trust_region_low_constant, trust_region_high_constant : float
        Shrink/growth factors applied to the trust-region radius on a
        rejected/accepted step respectively (``optimistix`` defaults:
        0.25 / 3.5).  Lower ``trust_region_high_constant`` grows the
        radius more conservatively after a good step.
    max_step_size : float
        Hard cap on the trust-region radius.  Without this, after ~5
        accepted steps the radius grows from 0.01 to >5 (due to the 3.5×
        growth factor), the Levenberg-Marquardt damping λ ~ 1/r²
        effectively vanishes, and the inner CG solve works on a
        near-singular ``J^T J`` — which can hang indefinitely.  Default
        1.0 keeps non-zero damping throughout the solve.
    verbose : bool
        If True, print per-step LM progress (loss, step size, etc.) via
        optimistix's built-in verbosity.

    Returns
    -------
    log_params : (S, P) array
        Optimised log-parameters.
    log_backgrounds : (E,) array
        Optimised log-background per exposure.
    result : optimistix.RESULTS
        Solver result/status flag; index into ``optimistix.RESULTS`` for
        a human-readable message (e.g. success, or why it failed/stopped).
    stats : dict
        Solver statistics (e.g. number of steps taken), from
        ``optimistix.Solution.stats``.
    """
    import optimistix as optx

    log_params = jnp.asarray(init_log_params, dtype=jnp.float32)

    if init_log_backgrounds is not None:
        log_backgrounds = jnp.asarray(init_log_backgrounds, dtype=jnp.float32)
    else:
        log_backgrounds = jnp.array([
            jnp.log(jnp.maximum(jnp.median(mean_img) / EXPOSURE_TIME, 1e-6))
            for mean_img, _, _, _, _, _ in exposures
        ], dtype=jnp.float32)

    residuals_fn = _make_residual_fn(exposures, n_lambda, oversampling)

    def _fn(y, args):
        del args
        return residuals_fn(y)

    y0 = (log_params, log_backgrounds)

    solver = _make_lm_solver(
        rtol=rtol, atol=atol, cg_rtol=cg_rtol, cg_atol=cg_atol,
        cg_max_steps=cg_max_steps,
        initial_step_size=initial_step_size,
        trust_region_low_constant=trust_region_low_constant,
        trust_region_high_constant=trust_region_high_constant,
        max_step_size=max_step_size,
        verbose=verbose,
    )

    sol = optx.least_squares(
        _fn, solver, y0=y0, max_steps=max_steps, throw=False,
    )

    log_params, log_backgrounds = sol.value
    return log_params, log_backgrounds, sol.result, sol.stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_history(losses, learning_rates, fname, final_loss=None):
    """Plot loss (left axis) and learning rate (right axis) vs step.

    Uses an asinh-based y-scale that transitions smoothly between linear
    and logarithmic behaviour, centred on ``loss = 1`` (the expected
    chi² / pixel at the true parameters).  The width of the linear region
    is set to ``b = 5 * abs(final_loss - 1)`` (or ``5`` if ``final_loss``
    is not provided), so well-converged runs get a tighter linear window
    that makes the late-stage convergence easier to see.
    """
    from matplotlib.ticker import FixedLocator, NullLocator

    losses = np.asarray(losses, dtype=float)
    if final_loss is None:
        final_loss = float(losses[-1]) if len(losses) > 0 else 1.0
    b = max(5.0 * abs(final_loss - 1.0), 0.5)  # floor at 0.5 to avoid near-zero

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    steps = np.arange(len(losses))
    ax1.plot(steps, losses, "b-", alpha=0.7, linewidth=0.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss  (chi² / pixel)", color="b")
    ax1.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # asinh scale: linear near y=1, logarithmic for |y-1| >> b
    forward = lambda y: b * np.arcsinh((y - 1.0) / b)
    inverse = lambda t: b * np.sinh(t / b) + 1.0
    ax1.set_yscale("function", functions=(forward, inverse))

    # ---- y-tick positions ---------------------------------------------------
    # Major ticks at powers of 10, and at 1.0 (the true-parameter baseline).
    loss_max = float(np.max(losses))
    majors = [1.0]
    k = 1
    while 10 ** k <= loss_max * 1.1:
        majors.append(10.0 ** k)
        k += 1
    # Include one below 1 if losses dip there.
    if loss_max >= 1.0 and float(np.min(losses)) < 0.95:
        majors.insert(0, 0.5)
    ax1.yaxis.set_major_locator(FixedLocator(majors))

    # Minor ticks at n × 10^m for n = 2..9 within each decade.
    minors = []
    for power in range(k):
        decade = 10.0 ** power
        for n in range(2, 10):
            v = n * decade
            if v <= loss_max * 1.05:
                minors.append(v)
    ax1.yaxis.set_minor_locator(FixedLocator(minors))

    ax2.plot(steps, learning_rates, "r-", alpha=0.7)
    ax2.set_ylabel("Learning rate", color="r")

    ax1.set_title("Loss history + cosine LR schedule")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def plot_comparison(true_params, recovered_params, bright_mask, fname):
    """Scatter true vs recovered log(T) and log(A).  Bright sources
    (``bright_mask``) are plotted in a different colour."""
    true_T = true_params[:, 0]
    true_A = true_params[:, 1]
    rec_T = recovered_params[:, 0]
    rec_A = recovered_params[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, true_vals, rec_vals, label in [
        (ax1, true_T, rec_T, "log(T / kK)"),
        (ax2, true_A, rec_A, "log(A / (W/m2/um))"),
    ]:
        ax.scatter(true_vals[~bright_mask], rec_vals[~bright_mask],
                   s=1, color="gray", alpha=0.3, rasterized=True)
        ax.scatter(true_vals[bright_mask], rec_vals[bright_mask],
                   s=4, color="red", alpha=0.8, label="Bright")
        vmin = min(true_vals.min(), rec_vals.min())
        vmax = max(true_vals.max(), rec_vals.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.5)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Recovered {label}")
        ax.legend(markerscale=3)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")
