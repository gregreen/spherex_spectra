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

def _per_exposure_diff_params(mean_img, sigma_img, positions, params_exp,
                              gen_module, log_background, half_stamp,
                              n_lambda, oversampling):
    """Flattened normalised residual for one exposure.

    Takes an ALREADY SELECTED/ASSEMBLED per-exposure parameter array
    ``params_exp`` of shape ``(S, 1 + P)`` (log-amplitude first - see
    ``spherex.spectrum``).  Shared by :func:`_per_exposure_diff` (which
    selects rows from the global log-parameter array) and by the direct
    amplitude solve (:func:`solve_log_amplitudes`), which instead assembles
    ``params_exp`` from a per-source amplitude vector plus fixed shapes.
    """
    pred = gen_module(
        positions,
        params_exp,
        postage_stamp_half_size=half_stamp,
        n_wavelength_samples=n_lambda,
        oversampling=oversampling,
    )
    pred_bg = (pred + jnp.exp(log_background)) * EXPOSURE_TIME
    return ((mean_img - pred_bg) / sigma_img).reshape(-1)


def _per_exposure_diff(mean_img, sigma_img, positions, log_params_global,
                       source_idx, gen_module, log_background, half_stamp,
                       n_lambda, oversampling):
    """Flattened (pre-square) normalised residual vector for one exposure.

    Same forward model/convention as :func:`_per_exposure_sumsq`, but
    returns the residual array itself (flattened to 1-D) rather than the
    summed square - needed as the ``fn(y, args) -> residuals`` callable
    expected by ``optimistix.least_squares``.
    """
    return _per_exposure_diff_params(
        mean_img, sigma_img, positions, log_params_global[source_idx],
        gen_module, log_background, half_stamp, n_lambda, oversampling,
    )


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


# ---------------------------------------------------------------------------
# Direct amplitude least-squares solve
# ---------------------------------------------------------------------------
#
# The forward model is exactly LINEAR in the per-source amplitude
# ``a = exp(log_amplitude)``: a source's contribution to the image is just
# ``a`` times its unit-amplitude postage stamp (the spectrum model provides
# only the *shape*).  So, holding the spectral shapes and backgrounds fixed,
# finding the optimal log-amplitude of every source is a linear weighted
# least-squares problem:
#
#       minimise_a  || (d - t_exp (G a + bg)) / sigma ||^2
#
# with ``G`` the (sparse, per-source postage-stamp) design operator.  Writing
# ``f(a) = y - M a`` for the residual vector (affine in ``a``), the normal
# equations are ``(M^T M) a = M^T y``, solved matrix-free with CG.  This
# removes the amplitude from the nonlinear optimisation entirely.

def _group_amplitude_data(group, shape_params, log_backgrounds_group,
                          n_lambda, oversampling):
    """Precompute the per-source unit-amplitude postage stamps for one band
    group, together with their pixel indices, the noise and the target
    residual.

    These arrays let the amplitude design operator ``M`` (and its adjoint and
    the Jacobi diagonal) be applied with cheap scatter/gather arithmetic, at
    a fraction of the cost of re-running the forward model on every CG
    iteration.

    Returns ``None`` if the generator does not expose ``source_stamps``.
    """
    gen_mod = group["gen_mod"]
    if not hasattr(gen_mod, "source_stamps"):
        return None
    image_width = getattr(gen_mod, "image_width", None)
    image_height = getattr(gen_mod, "image_height", None)
    pixel_scale = getattr(gen_mod, "pixel_scale", None)
    if image_width is None or image_height is None or pixel_scale is None:
        return None

    half = group["half_stamp"]
    positions = group["positions"]      # (E, S, 2)
    src_idx = group["src_idx"]          # (E, S)
    sigma = group["sigma_imgs"]         # (E, H, W)
    mean_imgs = group["mean_imgs"]      # (E, H, W)
    # ``log_backgrounds_group`` is ALREADY indexed by ``group["group_idx"]``
    # by the caller (one entry per exposure in this group).
    log_bg = log_backgrounds_group      # (E,)

    # Unit-amplitude parameters: log_amplitude = 0, shapes held fixed.
    shape_exp = shape_params[src_idx]                          # (E, S, P-1)
    params = jnp.concatenate(
        [jnp.zeros((*src_idx.shape, 1), dtype=positions.dtype), shape_exp],
        axis=-1,
    )

    def _one_exposure(pos, par):
        return gen_mod.source_stamps(
            pos, par,
            image_width=image_width,
            image_height=image_height,
            pixel_scale=pixel_scale,
            postage_stamp_half_size=half,
            n_wavelength_samples=n_lambda,
            oversampling=oversampling,
        )

    stamps, i_all, j_all = jax.vmap(_one_exposure)(positions, params)

    # Target residual at zero amplitude:  y = (d - t_exp * bg) / sigma.
    y = (mean_imgs - jnp.exp(log_bg)[:, None, None] * EXPOSURE_TIME) / sigma

    return {
        "stamps": stamps,
        "i_all": i_all,
        "j_all": j_all,
        "sigma": sigma,
        "y": y,
        "src_idx": src_idx,
    }


def _build_amplitude_problem(exposures, shape_params, log_backgrounds,
                             n_lambda, oversampling):
    """Build the per-band-group data needed by the amplitude solve.

    Returns a list of per-group dicts, or ``None`` if any generator lacks the
    ``source_stamps`` hook (caller should then skip the amplitude step).
    """
    data = []
    for idx in _group_exposures_by_band(exposures).values():
        group = _prepare_band_group(exposures, idx)
        log_bg = log_backgrounds[jnp.asarray(group["group_idx"])]
        gd = _group_amplitude_data(
            group, shape_params, log_bg, n_lambda, oversampling
        )
        if gd is None:
            return None
        data.append(gd)
    return data


def _amplitude_forward(amplitudes, groups_data):
    """Apply the amplitude design operator ``M``.

    Each source contributes ``amplitude * unit_stamp``, scattered into its
    postage-stamp pixels; the result is multiplied by ``t_exp / sigma`` to
    match the residual convention of the fit.
    """
    def _one_exposure(amp_e, stamps_e, i_e, j_e, sigma_e):
        contrib = stamps_e * amp_e[:, None, None]
        img = jnp.zeros_like(sigma_e)
        img = img.at[i_e, j_e].add(contrib)
        return EXPOSURE_TIME * img / sigma_e

    out = []
    for g in groups_data:
        amp = amplitudes[g["src_idx"]]                     # (E, S)
        out.append(jax.vmap(_one_exposure)(
            amp, g["stamps"], g["i_all"], g["j_all"], g["sigma"]
        ))
    return out


def _amplitude_adjoint(pieces, groups_data, n_src):
    """Apply the adjoint ``M^T`` to a per-group list of weighted arrays."""
    def _one_exposure(v_e, stamps_e, i_e, j_e, sigma_e):
        sig_patch = sigma_e[i_e, j_e]
        v_patch = v_e[i_e, j_e]
        weighted = EXPOSURE_TIME * stamps_e / sig_patch
        return jnp.sum(v_patch * weighted, axis=(1, 2))      # (S,)

    out = jnp.zeros((n_src,), dtype=jnp.float32)
    for g, v_g in zip(groups_data, pieces):
        contrib = jax.vmap(_one_exposure)(
            v_g, g["stamps"], g["i_all"], g["j_all"], g["sigma"]
        )                                                   # (E, S)
        out = out.at[g["src_idx"].ravel()].add(contrib.ravel())
    return out


def _amplitude_rhs(groups_data, n_src):
    """Right-hand side of the normal equations, ``M^T y``."""
    return _amplitude_adjoint(
        [g["y"] for g in groups_data], groups_data, n_src
    )


def _amplitude_diagonal(groups_data, n_src):
    """Diagonal of the weighted normal matrix ``M^T M``.

    Entry ``s`` is ``Σ_e Σ_i (t_exp · g_{s,i} / σ_{e,i})²`` over every
    exposure and postage-stamp pixel of source ``s`` — the Jacobi
    preconditioner diagonal.  Computed from the same precomputed unit-
    amplitude stamps used by the design operator.
    """
    diag = jnp.zeros((n_src,), dtype=jnp.float32)
    for g in groups_data:
        def _one_exposure(stamps_e, i_e, j_e, sigma_e):
            sig_patch = sigma_e[i_e, j_e]
            weighted = EXPOSURE_TIME * stamps_e / sig_patch
            return jnp.sum(weighted ** 2, axis=(1, 2))          # (S,)

        contrib = jax.vmap(_one_exposure)(
            g["stamps"], g["i_all"], g["j_all"], g["sigma"]
        )                                                       # (E, S)
        diag = diag.at[g["src_idx"].ravel()].add(contrib.ravel())
    return diag


def _cg_solve(matvec, b, max_steps, rtol, atol):
    """Conjugate-gradient solve of ``A x = b`` for symmetric positive
    definite ``A`` (given by the linear map ``matvec``).

    Written to run INSIDE an outer ``jax.jit``: the iteration is a
    ``jax.lax.while_loop`` (no Python-level loop), so the whole solve is one
    fused program.  Degenerate steps (``p^T A p <= 0``) freeze the iterate
    instead of producing NaNs, and the loop also stops if the squared
    residual becomes non-finite.

    Returns
    -------
    x : (N,) array
    n_steps : scalar int array
        Number of CG iterations actually performed.
    """
    b_norm2 = jnp.sum(b * b)
    tol = (rtol ** 2) * b_norm2 + atol ** 2

    x = jnp.zeros_like(b)
    r = b - matvec(x)
    p = r
    rs = jnp.sum(r * r)

    def cond(state):
        i, _x, _r, _p, rs = state
        return (i < max_steps) & (rs > tol) & jnp.isfinite(rs)

    def body(state):
        i, x, r, p, rs = state
        ap = matvec(p)
        pap = jnp.sum(p * ap)
        good = pap > 0
        alpha = jnp.where(good, rs / jnp.where(good, pap, 1.0), 0.0)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = jnp.sum(r * r)
        beta = jnp.where(
            good & (rs > 0), rs_new / jnp.where(rs > 0, rs, 1.0), 0.0
        )
        return (i + 1, x, r, r + beta * p, rs_new)

    i, x, r, p, rs = jax.lax.while_loop(
        cond, body, (jnp.zeros((), jnp.int32), x, r, p, rs)
    )
    return x, i


def _amplitude_solver_available(exposures):
    """True if every exposure's generator exposes ``source_stamps`` and the
    detector geometry attributes the amplitude solver needs."""
    for exp in exposures:
        gen = exp[3]
        if not hasattr(gen, "source_stamps"):
            return False
        for name in ("image_width", "image_height", "pixel_scale"):
            if getattr(gen, name, None) is None:
                return False
    return True


def _make_amplitude_solver(exposures, n_lambda=5, oversampling=2, damping=1e-8,
                           cg_rtol=1e-2, cg_atol=1e-4, cg_max_steps=50):
    """Build a SINGLE jitted amplitude solver for a set of exposures.

    Returns a callable ``(log_params, log_backgrounds) -> (log_amplitude,
    n_steps, finite)`` that compiles the entire pipeline -- per-source
    unit-amplitude stamps, the normal-equation diagonal and right-hand side,
    the Jacobi scaling, and a fixed-iteration conjugate-gradient solve -- into
    **one** XLA program.

    Building the solver once and reusing it across the optimisation is
    essential for performance: a fresh, non-jitted call re-traces the whole
    stamp program on every invocation, and measurement showed that ~91% of
    the wall-clock cost of the amplitude solve was JAX tracing rather than
    arithmetic.  The exposure data is closed over; it never changes during a
    fit, so JAX compiles once and later calls with new parameters reuse the
    same executable.
    """
    if not _amplitude_solver_available(exposures):
        raise ValueError(
            "the amplitude solver requires generators exposing "
            "`source_stamps` and the detector geometry attributes "
            "(e.g. ImageGenerator3 / SpherexImageGenerator3)"
        )
    damp = max(float(damping), 1e-12)

    @jax.jit
    def _solve(log_params, log_backgrounds):
        shape_params = log_params[:, 1:]
        n_src = log_params.shape[0]

        groups_data = _build_amplitude_problem(
            exposures, shape_params, log_backgrounds, n_lambda, oversampling
        )

        diag = _amplitude_diagonal(groups_data, n_src)
        rhs = _amplitude_rhs(groups_data, n_src)

        diag = _amplitude_diagonal(groups_data, n_src)

        # Jacobi (unit-diagonal) change of variables.  This is REQUIRED for
        # correctness, not merely for speed: the design operator is expressed
        # in *absolute* amplitude units, so in the unscaled variables the
        # normal matrix spans many decades and float32 CG cannot converge.
        # Sources whose stamps are (almost) entirely off-detector have a
        # negligible diagonal and no amplitude information; they are excluded
        # (zero scale) and handled by the shift below.
        observed = diag > jnp.max(diag) * 1e-6
        scale = jnp.where(
            observed, 1.0 / jnp.sqrt(jnp.maximum(diag, 1e-30)), 0.0
        )
        # Unit ridge on observed rows; O(1) shift on the excluded, decoupled
        # (zero-rhs) rows, so the operator is strictly positive definite.
        shift = jnp.where(observed, damp, 1.0)

        def scaled_matvec(c):
            v = scale * c
            v = _amplitude_adjoint(
                _amplitude_forward(v, groups_data), groups_data, n_src
            )
            return scale * v + shift * c

        x, n_steps = _cg_solve(
            scaled_matvec, scale * rhs, cg_max_steps, cg_rtol, cg_atol
        )

        amplitudes = scale * x
        finite = jnp.all(jnp.isfinite(amplitudes))
        amplitudes = jnp.where(observed, jnp.maximum(amplitudes, 1e-30), 1.0)
        log_amplitude = jnp.where(
            observed, jnp.log(amplitudes), log_params[:, 0]
        )
        # Branchless failure handling: a non-finite solve keeps the old value.
        log_amplitude = jnp.where(finite, log_amplitude, log_params[:, 0])
        return log_amplitude, n_steps, finite

    return _solve


def solve_log_amplitudes(
    exposures,
    log_params,
    log_backgrounds,
    n_lambda=5,
    oversampling=2,
    damping=1e-8,
    cg_rtol=1e-2,
    cg_atol=1e-4,
    cg_max_steps=50,
):
    """Solve DIRECTLY for each source's optimal log-amplitude.

    Holds the spectral shapes (``log_params[:, 1:]``) and the backgrounds
    fixed, and solves the linear weighted least-squares problem for the
    amplitudes ``a = exp(log_amplitude)`` of ALL sources *jointly* across the
    whole exposure stack (so overlaps/blending and multi-exposure constraints
    are handled exactly).

    Because the model is linear in ``a``, the design operator ``M`` is built
    directly from the per-source unit-amplitude postage stamps (computed via
    ``gen_module.source_stamps``); the normal equations ``(M^T M) a = M^T y``
    are then solved matrix-free by conjugate gradient.  Each CG iteration is a
    cheap scatter/gather over the cached stamps rather than a forward-model
    evaluation.

    The whole computation is wrapped in a single ``jax.jit`` (see
    :func:`_make_amplitude_solver`).  **Callers that run this repeatedly
    should build the solver once with :func:`_make_amplitude_solver` and reuse
    it**, as :func:`infer_parameters` does; otherwise each call re-traces the
    stamp program, which dominates the runtime.

    By default the system is Jacobi-preconditioned by a symmetric scaling
    transform ``a = D b`` with ``D = diag(M^T M)^{-1/2}`` (so the scaled
    normal matrix has unit diagonal).  Preconditioning is not merely an
    optimisation: the design operator is expressed in *absolute* amplitude
    units, so an unscaled float32 solve overflows.  There is therefore no
    option to disable it.

    Parameters
    ----------
    exposures : list of (mean_img, sigma_img, positions_pix, gen_module,
                         half_stamp, source_idx)
        Same format as accepted by :func:`infer_parameters`.
    log_params : (N, 1 + P) array
        Current log-parameters, amplitude in column 0.  Only the shape
        columns affect the operator; the amplitude column is a fallback for
        sources that appear in no exposure.
    log_backgrounds : (E,) array
        Log-background (rate, s^-1) per exposure, held fixed.
    n_lambda, oversampling
        Must match the values used elsewhere in the fit.
    damping : float
        Ridge term (applied in the scaled space) guaranteeing the normal
        matrix is positive definite.  Defaults to a tiny value with
        negligible bias.
    cg_rtol, cg_atol, cg_max_steps
        Inner CG tolerances / iteration cap.  The preconditioned system is
        well scaled and converges in a handful of iterations, so the default
        cap of 50 bounds the cost without limiting accuracy.

    Returns
    -------
    log_amplitude : (N,) array
        Optimal log-amplitude per source (unchanged for sources that appear in
        no exposure).  If the inner solve produces non-finite values, the
        current log-amplitudes are returned unchanged and
        ``stats["success"]`` is False.
    stats : dict
        Inner CG solver statistics (iterations, ``success``).
    """
    solver = _make_amplitude_solver(
        exposures, n_lambda=n_lambda, oversampling=oversampling,
        damping=damping, cg_rtol=cg_rtol, cg_atol=cg_atol,
        cg_max_steps=cg_max_steps,
    )
    log_params = jnp.asarray(log_params, dtype=jnp.float32)
    log_backgrounds = jnp.asarray(log_backgrounds, dtype=jnp.float32)
    log_amplitude, n_steps, finite = solver(log_params, log_backgrounds)
    return log_amplitude, {
        "num_steps": int(n_steps), "success": bool(finite),
    }


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
    log_params : (S, 1 + P) array
        Log-parameters, amplitude in the first column.
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
    log_params : (S, 1 + P) array
        Log-parameters to evaluate at (e.g. the true simulation values),
        amplitude in the first column.
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


def _make_train_step(loss_fn, optimiser):
    """Build one ``jax.jit``-compiled training step from a prebuilt
    ``loss_fn`` (see :func:`_make_loss_fn`).

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
    amp_solve_every=10,
    amp_cg_max_steps=50,
    amp_verbose=False,
):
    """Infer log(amplitude), log(temperature) and log(background) via
    preconditioned SGD, interleaved with a direct amplitude least-squares
    solve.

    The parameter vector per source is ``[log_amplitude, log_temperature]``
    (amplitude FIRST - see ``spherex.spectrum``).  In addition to plain SGD
    on all parameters, every ``amp_solve_every`` steps the log-amplitude
    column is replaced by the closed-form least-squares solution from
    :func:`solve_log_amplitudes` (shapes and backgrounds held fixed) - see
    that function for why this is exact rather than approximate.

    Parameters
    ----------
    exposures : list of (mean_img, sigma_img, positions_pix,
                         gen_module, half_stamp, source_idx)
        One tuple per exposure.  ``positions_pix`` are source positions
        in pixel coordinates (pre-computed by caller from WCS).
    init_log_params : (S, 1 + P) array
        Initial log-parameters: log-amplitude in the first column, then the
        spectrum-shape parameters (e.g. log(T/kK)).
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
    amp_solve_every : int or None
        Interlace the direct amplitude least-squares solve every this many
        SGD steps (default 10), plus one final solve after the loop.
        ``None`` (or 0) disables the interlace.
    amp_cg_max_steps : int or None
        Iteration cap for the amplitude solve's inner CG.
    amp_verbose : bool
        If True, print the loss before/after each interleaved amplitude
        solve (requires an extra loss evaluation per solve).

    Returns
    -------
    log_params : (S, 1 + P) array
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

    loss_fn = _make_loss_fn(exposures, n_lambda, oversampling, batched)
    train_step = _make_train_step(loss_fn, optimiser)

    # ---- amplitude least-squares interlace ---------------------------------
    # The forward model is exactly linear in the source amplitude, so every
    # ``amp_solve_every`` SGD steps we replace the amplitude column with the
    # direct least-squares solution (shapes and backgrounds held fixed).
    # ``amp_solve_every=None`` or 0 disables the interlace.
    amp_solve_every = 0 if amp_solve_every is None else int(amp_solve_every)
    if amp_solve_every > 0 and not _amplitude_solver_available(exposures):
        print("  [amp-solve] generators do not expose `source_stamps`; "
              "disabling the amplitude interlace.")
        amp_solve_every = 0
    do_amp_solve = amp_solve_every > 0

    # Build (and compile) the amplitude solver ONCE, so the interlace loop
    # never re-traces or recompiles it.
    amp_solver = None
    if do_amp_solve:
        amp_solver = _make_amplitude_solver(
            exposures, n_lambda=n_lambda, oversampling=oversampling,
            cg_max_steps=amp_cg_max_steps,
        )

    def _amp_solve(log_params, log_backgrounds):
        log_amp, _n_steps, finite = amp_solver(log_params, log_backgrounds)
        if not bool(finite):
            print("  [amp-solve] inner CG solve failed to produce a finite "
                  "solution; amplitudes left unchanged.")
        return log_params.at[:, 0].set(log_amp)

    losses = []
    learning_rates = []

    pbar = tqdm(range(n_steps), desc="SGD")
    for step in pbar:
        params, opt_state, loss_val = train_step(params, opt_state)

        if not jnp.isfinite(loss_val):
            print(f"  Loss became non-finite at step {step}, stopping.")
            break

        log_params, log_backgrounds = params

        if do_amp_solve and (step + 1) % amp_solve_every == 0:
            log_params = _amp_solve(log_params, log_backgrounds)
            params = (log_params, log_backgrounds)
            if amp_verbose:
                post = float(loss_fn(log_params, log_backgrounds))
                print(f"  [amp-solve @ step {step}] "
                      f"loss {float(loss_val):.4e} -> {post:.4e}")

        losses.append(float(loss_val))
        lr = float(schedule(step))
        learning_rates.append(lr)

        pbar.set_postfix(loss=f"{float(loss_val):.4e}",
                         lr=f"{float(lr):.2e}")

    if do_amp_solve:
        log_params = _amp_solve(log_params, log_backgrounds)

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
    init_log_params : (S, 1 + P) array
        Initial log-parameters: log-amplitude in the first column, then the
        spectrum-shape parameters (e.g. log(T/kK)).
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
    log_params : (S, 1 + P) array
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
    """Scatter true vs recovered log(A) and log(T).  Bright sources
    (``bright_mask``) are plotted in a different colour.

    Parameter layout is ``[log_amplitude, log_temperature]`` (amplitude
    first - see ``spherex.spectrum``).
    """
    true_A = true_params[:, 0]
    true_T = true_params[:, 1]
    rec_A = recovered_params[:, 0]
    rec_T = recovered_params[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, true_vals, rec_vals, label in [
        (ax1, true_A, rec_A, "log(A / (W/m2/um))"),
        (ax2, true_T, rec_T, "log(T / kK)"),
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
