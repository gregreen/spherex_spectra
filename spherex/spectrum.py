"""Source spectrum models for SPHEREx photometry.

Each spectrum model is an ``equinox.Module`` that acts as a *parameterised
template* describing the **shape** of a source spectrum: it stores no
per-source state, and its ``__call__`` method accepts explicit parameter
arrays so that gradients flow through them naturally.

Log-flux contract
-----------------
A model returns the source's **unnormalised log-flux**: the logarithm of the
spectral flux density, up to an additive constant that is independent of
wavelength::

    spectrum_model(wavelength, shape_params) -> log f_lambda + const

Models know nothing about :data:`LAMBDA_0` and they never exponentiate: both
are the *caller's* business, because only the caller knows which wavelengths,
and which reference wavelength, it needs.

Callers recover a physical flux density by normalising at the global
reference wavelength :data:`LAMBDA_0`::

    f_lambda(lambda) = exp( log_amplitude
                            + spectrum_model(lambda, theta)
                            - spectrum_model(LAMBDA_0, theta) )

so the dimensionless shape is exactly 1 at ``LAMBDA_0``, and ``log_amplitude``
keeps its direct physical meaning: the log of the source's flux density at
``LAMBDA_0``.  The helpers below implement this in one place - see
:func:`reference_log_flux`, :func:`normalized_shape` and
:func:`normalized_source_params`.

The image generators normalise **once per source**: they fold the constant
into the amplitude with :func:`normalized_source_params` and then evaluate
``exp(log_amplitude + spectrum_model(lambda, theta))`` per wavelength.  Doing
it any later - inside the model, i.e. once per sub-pixel - would re-evaluate a
wavelength-independent quantity millions of times per image.

Working in *log* space, rather than having models return an unnormalised
linear kernel, matters for a flexible model: the network's raw output carries
an offset that is independent of wavelength and that the data cannot
constrain, so exponentiating it and dividing by its value at ``LAMBDA_0``
would overflow float32.  Subtracting in log space and exponentiating only the
(bounded) difference is immune to that.

Amplitude convention
--------------------
The flux-density scale (``log_amplitude``) is deliberately *not* part of a
spectrum model.  It is carried as a separate per-source parameter, always
stored as the **first** column of ``source_params`` (see
:data:`LOG_AMPLITUDE_INDEX`); the remaining columns
(``source_params[:, 1:]``) are the shape parameters ``theta`` handed to the
model.

Splitting the amplitude out this way lets the optimiser solve for the optimal
amplitude of every source *directly*: the forward model is exactly linear in
amplitude, so it becomes a linear least-squares problem (see
``scripts.inference.solve_log_amplitudes``).
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from .constants import HC_JAX, KB_JAX

# ---------------------------------------------------------------------------
# Amplitude / normalisation conventions
# ---------------------------------------------------------------------------

#: Reference wavelength (microns) at which CALLERS normalise every spectrum
#: model's log-flux, so that the dimensionless shape equals ``f_lambda = 1``
#: there.  This is a single GLOBAL constant shared by all bands so that the
#: modelled spectrum stays smooth in amplitude across the whole wavelength
#: range: changing band only changes the *sampling* of the shape, never its
#: normalisation.  The models themselves are anchor-free (see the module
#: docstring); this constant is used by the helpers below and by the callers.
LAMBDA_0 = 1.0

#: Index of the log-amplitude column in ``source_params``.  By convention the
#: amplitude is ALWAYS the first parameter; the remaining columns
#: (``source_params[:, 1:]``) are the ones passed to the spectrum model.
LOG_AMPLITUDE_INDEX = 0

#: Default wavelength span covered by the Fourier embedding of an
#: :class:`NeuralNetSpectrum`, expressed in **natural log of the wavelength**:
#: ``ln(5.0 / 0.75)`` spans the full SPHEREx wavelength range (0.75 - 5.0 um).
#: Working in log-wavelength space makes the embedding scale-invariant, which
#: is what the instrument's bandpasses do: a filter's fractional width
#: ``sigma / lambda_c`` is the same in every band.
DEFAULT_DELTA_LN_WAVELENGTH = math.log(5.0 / 0.75)


def split_source_params(source_params: jnp.ndarray):
    """Split ``source_params`` into ``(log_amplitude, shape_params)``.

    Parameters
    ----------
    source_params : jnp.ndarray, shape (..., 1 + P)
        Per-source parameters with the log-amplitude in the first column.

    Returns
    -------
    log_amplitude : jnp.ndarray, shape (..., 1)
    shape_params : jnp.ndarray, shape (..., P)
    """
    start = LOG_AMPLITUDE_INDEX
    log_amplitude = source_params[..., start:start + 1]
    shape_params = source_params[..., start + 1:]
    return log_amplitude, shape_params


def join_source_params(log_amplitude: jnp.ndarray, shape_params: jnp.ndarray):
    """Inverse of :func:`split_source_params` (amplitude first)."""
    return jnp.concatenate([log_amplitude, shape_params], axis=-1)


# ---------------------------------------------------------------------------
# Caller-side normalisation at LAMBDA_0
# ---------------------------------------------------------------------------
# The models return an unnormalised log-flux (see the module docstring).  These
# helpers turn that into the dimensionless shape, which is exactly 1 at
# LAMBDA_0.  All of them describe a SINGLE source; ``vmap`` for batches.

def reference_log_flux(
    spectrum_model: eqx.Module,
    shape_params: jnp.ndarray,
    lambda_0: float = LAMBDA_0,
) -> jnp.ndarray:
    """Log-flux of ``spectrum_model`` at the reference wavelength.

    This is the normalisation constant that the caller subtracts, and it
    depends only on the shape parameters - never on wavelength or position -
    so callers should compute it ONCE per source and reuse it.

    Parameters
    ----------
    spectrum_model : equinox.Module
    shape_params : jnp.ndarray, shape (P,)
        Shape parameters of a single source.
    lambda_0 : float, optional
        Reference wavelength in microns (default: the global ``LAMBDA_0``).

    Returns
    -------
    jnp.ndarray
        Scalar log-flux at ``lambda_0``.
    """
    return spectrum_model(jnp.atleast_1d(lambda_0), shape_params)[..., 0]


def normalized_log_shape(
    spectrum_model: eqx.Module,
    wavelength: jnp.ndarray,
    shape_params: jnp.ndarray,
    lambda_0: float = LAMBDA_0,
) -> jnp.ndarray:
    """Log of the dimensionless shape, anchored to 0 at ``lambda_0``.

    Equivalent to ``log f_lambda(lambda) - log f_lambda(lambda_0)``, i.e. the
    quantity that is exponentiated to obtain a physical spectrum.

    Parameters
    ----------
    spectrum_model : equinox.Module
    wavelength : jnp.ndarray, shape (N_lambda,)
    shape_params : jnp.ndarray, shape (P,)
        Shape parameters of a single source.
    lambda_0 : float, optional

    Returns
    -------
    jnp.ndarray, shape (N_lambda,)
    """
    return (spectrum_model(wavelength, shape_params)
            - reference_log_flux(spectrum_model, shape_params, lambda_0))


def normalized_shape(
    spectrum_model: eqx.Module,
    wavelength: jnp.ndarray,
    shape_params: jnp.ndarray,
    lambda_0: float = LAMBDA_0,
) -> jnp.ndarray:
    """Dimensionless spectrum shape, exactly 1 at ``lambda_0``.

    The physical flux density is ``exp(log_amplitude) * normalized_shape(...)``.
    This is the convenience form for diagnostics, plots and tests - the image
    generators instead use :func:`normalized_source_params`, which avoids a
    second model evaluation per call site.
    """
    return jnp.exp(normalized_log_shape(
        spectrum_model, wavelength, shape_params, lambda_0
    ))


def normalized_source_params(
    spectrum_model: eqx.Module,
    source_params: jnp.ndarray,
    lambda_0: float = LAMBDA_0,
) -> jnp.ndarray:
    """Fold the ``lambda_0`` normalisation into the log-amplitude column.

    Returns ``source_params`` with column 0 replaced by
    ``log_amplitude - reference_log_flux(spectrum_model, theta)``, so that the
    cheap per-wavelength expression

        exp(params[0]) * exp(spectrum_model(lambda, params[1:]))

    is the physical flux density whose value at ``lambda_0`` is
    ``exp(log_amplitude)``.  Callers should apply this ONCE PER SOURCE, before
    looping or vmapping over wavelengths or sub-pixels: the folded constant is
    wavelength-independent, so recomputing it per sub-pixel is pure waste.

    Note that the returned array is for INTERNAL use with a log-flux model -
    its first column is no longer the physical ``log f_lambda(lambda_0)``.

    Parameters
    ----------
    spectrum_model : equinox.Module
    source_params : jnp.ndarray, shape (1 + P,)
        Parameters of a SINGLE source (amplitude first).
    lambda_0 : float, optional

    Returns
    -------
    jnp.ndarray, shape (1 + P,)
    """
    log_amplitude, shape_params = split_source_params(source_params)
    log_reference = spectrum_model(
        jnp.atleast_1d(lambda_0), shape_params
    )[..., 0:1]
    return join_source_params(log_amplitude - log_reference, shape_params)


class BlackbodySpectrum(eqx.Module):
    """Blackbody spectrum template (unnormalised log-flux).

    Returns the log of the Planck function up to the additive constant
    ``log(2 h c^2)``, i.e. a *log-flux kernel*::

        log_kernel(lambda) = -5 * log(lambda) - log(expm1(x))
        x = hc / (lambda * k_B * T)

    That constant is independent of both wavelength and temperature, so the
    caller removes it by subtracting the value at the reference wavelength
    :data:`LAMBDA_0` (see :func:`normalized_shape` and
    :func:`normalized_source_params`).  The resulting dimensionless shape is
    the familiar Planck ratio::

        shape(lambda) = (lambda_0 / lambda)^5 * (exp(x_0) - 1) / (exp(x) - 1)

    Parameters
    ----------
    wavelength : jnp.ndarray, shape (...,)
        Wavelength(s) in microns.
    shape_params : jnp.ndarray, shape (..., 1)
        Per-source shape parameters in **log-space**.
        ``shape_params[..., 0]`` = log(temperature / kK).

    Attributes
    ----------
    n_params : int
        Number of spectrum-*shape* parameters (1: log-temperature).  The
        amplitude is not counted here (it is held outside the model).
    """

    n_params: int = 1

    def __call__(
        self,
        wavelength: jnp.ndarray,
        shape_params: jnp.ndarray,
    ) -> jnp.ndarray:
        # Slice with a trailing dummy axis for broadcasting.
        log_temperature = shape_params[..., 0:1]   # log(kK)  (..., 1)
        temperature = jnp.exp(log_temperature)     # kK

        # Exponent x = hc / (lambda * k_B * T), written via a pre-factored
        # inv_T_scale = hc / (lambda * k_B) so that its gradient w.r.t. T
        # becomes -inv_T_scale / T^2 - avoiding the (k_B * T)^2 term that
        # underflows float32 at T ~ 5 kK (see explanatory supplement 7.10).
        inv_T_scale = HC_JAX / (wavelength * KB_JAX)   # K
        exponent = inv_T_scale / temperature           # dimensionless

        # Guard exp overflow (short wavelengths / low temperatures).
        exponent = jnp.clip(exponent, None, 50.0)

        # log B_lambda = log(2 h c^2) - 5 log(lambda) - log(exp(x) - 1).  The
        # log(2 h c^2) term is dropped here; the caller removes it (together
        # with any other wavelength-independent offset) by normalising at
        # LAMBDA_0.
        return -5.0 * jnp.log(wavelength) - jnp.log(jnp.expm1(exponent))


# ---------------------------------------------------------------------------
# FiLM conditioning
# ---------------------------------------------------------------------------

class FiLMLayer(eqx.Module):
    """Feature-wise linear modulation of one hidden layer.

    Maps a source's shape parameters ``theta`` to a per-neuron scale
    ``gamma`` and offset ``beta`` for ONE hidden layer of
    :class:`NeuralNetSpectrum`; that layer then computes
    ``silu(gamma * (W h + b) + beta)``.  This is how ``theta`` reaches the
    network at all - it is *not* concatenated to the wavelength features.

    The branch is deliberately tiny, and its hidden width is tied to the size
    of ``theta`` rather than being a fixed constant::

        hidden width  W = max(round(film_hidden_size_factor * n_params), 1)

    A fixed width would cap how much of ``theta`` can reach ``gamma``/``beta``:
    with ``P`` shape parameters and a 4-unit hidden layer, the modulation could
    only depend on 4 independent combinations of them, so the model would
    behave as if ``theta`` were 4-dimensional however large ``P`` is.  With
    ``film_hidden_size_factor = 1`` the branch is exactly as wide as ``theta``
    (no compression at all); larger factors widen it.

    Parameters
    ----------
    n_params : int
        Size ``P`` of ``theta`` (the branch's input width).
    film_hidden_layers : int
        Number of hidden layers inside the branch.  ``0`` gives a single
        linear map ``theta -> (gamma, beta)``.
    film_hidden_size : int
        Width ``W`` of those hidden layers (see the width rule above).
    hidden_size : int
        Width of the hidden layer being modulated; the branch outputs
        ``2 * hidden_size`` numbers (a scale and an offset per neuron).
    key : jax.Array
        PRNG key for the branch's initialisation.

    Notes
    -----
    The weights use plain random initialisation, like every other layer.  An
    "identity at initialisation" scheme (``gamma = 1``, ``beta = 0``, i.e. a
    zeroed output projection) is deliberately *not* used: with a linear output
    projection it would make ``d gamma / d theta == 0`` for every ``theta``, so
    the shape parameters would have no effect on the spectrum at all - fatal
    for a model whose weights are frozen and whose entire purpose is that
    ``theta`` selects the spectrum.
    """

    layers: list
    n_params: int
    film_hidden_layers: int
    film_hidden_size: int
    hidden_size: int

    def __init__(
        self,
        n_params: int,
        film_hidden_layers: int,
        film_hidden_size: int,
        hidden_size: int,
        *,
        key: jax.Array,
    ):
        self.n_params = n_params
        self.film_hidden_layers = film_hidden_layers
        self.film_hidden_size = film_hidden_size
        self.hidden_size = hidden_size

        # theta -> W -> ... -> W -> (gamma, beta).  With no hidden layers this
        # collapses to a single Linear(n_params, 2 * hidden_size), so the
        # factor simply does not apply.
        dims = ([n_params] + [film_hidden_size] * film_hidden_layers
                + [2 * hidden_size])
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(in_dim, out_dim, key=layer_key)
            for in_dim, out_dim, layer_key in zip(dims[:-1], dims[1:], keys)
        ]

    def __call__(self, shape_params: jnp.ndarray):
        """Map ``theta`` to ``(gamma, beta)``, both of shape ``(hidden_size,)``.

        SiLU is applied between the branch's hidden layers only: the final
        projection is left linear, so ``(gamma, beta)`` are an affine function
        of the last hidden features.
        """
        x = shape_params
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        gamma, beta = jnp.split(self.layers[-1](x), 2)
        return gamma, beta


class NeuralNetSpectrum(eqx.Module):
    """Neural-network spectrum template (unnormalised log-flux).

    A flexible alternative to :class:`BlackbodySpectrum`.  A small MLP maps the
    wavelength to a log-flux, with the shape parameters entering through FiLM
    modulation of every hidden layer, and ``__call__`` returns it directly::

        log_kernel(lambda) = NN(lambda; theta)

    Like the blackbody, this is defined only up to a wavelength-independent
    offset: the caller normalises it by subtracting ``NN(LAMBDA_0, theta)``
    (see :func:`normalized_shape` / :func:`normalized_source_params`), which is
    what makes the amplitude (the first column of ``source_params``) keep its
    meaning as ``f_lambda(LAMBDA_0)``.  The amplitude machinery needs no
    changes: it only requires the shape to be independent of the amplitude and
    the model to be *linear in amplitude*, both of which hold here (see
    ``scripts.inference.solve_log_amplitudes``).

    The model neither normalises nor exponentiates.  Both matter: ``exp`` and
    the ``LAMBDA_0`` evaluation are the *caller's* job, so the extra network
    evaluation happens once per source instead of once per sub-pixel, and the
    (unconstrained, wavelength-independent) offset never gets exponentiated.

    Wavelength embedding
    --------------------
    The wavelength is not fed to the network raw: it is first expanded into a
    Fourier (positional) feature vector containing ``ln(lambda)`` itself plus
    ``sin`` and ``cos`` of ``n_embeddings`` geometrically spaced frequencies in
    **log-wavelength space**, ``k_i = 2**i * pi / delta_ln_wavelength`` (see
    :meth:`_embed_wavelength`).  The i-th frequency completes ``2**(i-1)``
    cycles across ``delta_ln_wavelength``, so the highest one resolves
    fractional wavelength structure on scales of roughly
    ``2**(1 - n_embeddings)`` in ``lambda2 / lambda1`` - i.e. a *fixed fraction*
    of the wavelength, the same in every band.

    Log-wavelength space is deliberate.  A SPHEREx filter's width scales with
    wavelength (``sigma / lambda_c = 1 / (2.355 R)`` is constant across the
    range), and a spectrum's interesting structure - emission lines, absorption
    edges, dust features - is likewise quasi-log-periodic.  Embedding
    ``lambda`` linearly instead would make a fixed feature period an ever
    smaller *fraction* of the bandpass as wavelength grows, so the fluctuations
    would fade away at long wavelengths and be over-represented at short ones.
    In log space one setting of ``n_embeddings`` gives the same relative
    resolution in every band.

    This increases the model's *representational* flexibility (the shape can
    bend on more than one scale) but not the number of free parameters: the
    frequencies are fixed, so ``theta`` alone still selects a curve.  Note two
    consequences when choosing ``n_embeddings``: a higher setting makes the
    shape wigglier within a bandpass (see the quadrature caution below), and
    the embedding costs ``2 * n_embeddings`` transcendentals per call, which is
    comparable to the arithmetic of a small MLP.

    How ``theta`` enters (FiLM modulation)
    --------------------------------------
    ``theta`` is **not** concatenated to the wavelength features.  Instead,
    every hidden layer has its own :class:`FiLMLayer` - a small MLP mapping
    ``theta`` to a per-neuron scale and offset::

        z_l         = W_l h_{l-1} + b_l
        gamma, beta = FiLM_l(theta)
        h_l         = silu(gamma * z_l + beta)

    so ``theta`` modulates the hidden features multiplicatively (and shifts
    them) instead of entering as one more input feature alongside the
    wavelength.  Two properties are worth keeping in mind:

    * The branch's hidden width scales with ``theta``
      (``film_hidden_size_factor * n_params``, see :class:`FiLMLayer`), so a
      large ``P`` is never squeezed through a fixed narrow bottleneck.
    * ``(gamma, beta)`` are wavelength-independent, so they are computed ONCE
      per call and broadcast over the wavelength axis (see :meth:`__call__`):
      the branches add no per-wavelength cost.

    There is one branch per *activated* layer, i.e. ``n_hidden_layers + 1`` of
    them, because the input projection is itself a hidden layer.  That also
    means ``theta`` is never ignored, even at ``n_hidden_layers = 0``, where
    the input projection's branch would be its only route into the network.

    Optional layer normalization
    ----------------------------
    With ``layer_norm=True`` an ``eqx.nn.LayerNorm`` is inserted after the
    activation of every activated layer.  This is a **per-wavelength**
    operation: it standardises the ``hidden_size`` features of ONE wavelength's
    hidden vector, so the value returned for a given ``(theta, wavelength)``
    cannot depend on which other wavelengths or ``theta`` values share the
    batch.  Normalising over the *wavelength* axis would do exactly that and is
    deliberately not offered - it would make ``f_lambda`` a function of the
    caller's wavelength grid.

    Why it is an option rather than always on: it changes how much spectral
    structure a *randomly initialised* network has.  Measured raw per-source
    log-shape std over 0.75-5 um at the mock's defaults (1x32, 8 embeddings,
    seed 314159): 0.015 without normalization versus 0.100 with it - i.e. the
    output rescaling the mock applies afterwards needs a factor of ~33 instead
    of ~5.  It does NOT remove the need for that rescaling, because
    normalization divides by a *wavelength-dependent* quantity and so cannot
    pin a statistic defined over wavelengths; the achieved spread still varies
    with the seed (a factor 4.4-17.6 over three seeds at 2x16).  Once the
    weights are *trained* that argument disappears and the normalization is an
    ordinary architectural choice.

    It costs ``2 * hidden_size`` parameters per activated layer (the affine
    scale and offset, initialised to 1 and 0 - identity, which is what makes it
    a pure standardisation at initialisation) plus a feature-axis mean and
    variance per wavelength sample.

    Batching convention
    -------------------
    ``__call__`` handles the wavelengths of a SINGLE source::

        wavelength   : (N_lambda,)
        shape_params : (n_params,)
        returns      : (N_lambda,)

    This is exactly what the image generators need:
    ``spherex.image._one_subpixel_rate`` calls
    ``spectrum_model(lambdas, shape_params)`` once per source and per
    sub-pixel, and expects ``(N_lambda,)`` back.  Batching over sources and
    sub-pixels is the *caller's* job there (``jax.vmap`` over sub-pixels,
    ``lax.scan``/``lax.map`` over sources), so the model must not try to batch
    those axes itself.

    Internally the network is evaluated one wavelength at a time and vectorised
    with ``jax.vmap``.  That keeps the network definition simple: a layer
    expects the feature axis last, so the scalar wavelength embedding is the
    natural core of the computation.  The FiLM parameters are computed once for
    the whole source (they do not depend on wavelength) and passed to the
    vmapped core, so no branch is re-evaluated per wavelength.  To batch over
    sources as well, compose a second ``vmap`` at the call site::

        jax.vmap(model, in_axes=(None, 0))(wavelengths, shape_params_batch)

    Parameters
    ----------
    n_params : int
        Number of spectrum-*shape* parameters per source.  The amplitude is
        not counted here; the total width of ``source_params`` is
        ``1 + n_params``.
    n_hidden_layers : int
        Number of hidden layers in the neural network.
    hidden_size : int
        Number of neurons in each hidden layer.
    n_embeddings : int, optional
        Number of Fourier frequencies used to embed the wavelength (default
        8, i.e. 1 + 16 input features from the wavelength).  Set to 0 to feed
        ``ln(lambda)`` only.    delta_ln_wavelength : float, optional
        Span of the embedding in log-wavelength, i.e. the range of
        ``ln(lambda)`` the Fourier features should cover (default
        ``ln(5.0 / 0.75)``, the full 0.75 - 5.0 um SPHEREx range).
    film_hidden_layers : int, optional
        Hidden layers inside each FiLM branch (default 1).  ``0`` makes every
        branch a single linear map from ``theta`` to ``(gamma, beta)``.
    film_hidden_size_factor : float, optional
        Width of each FiLM branch's hidden layers, as a multiple of the number
        of shape parameters: ``W = max(round(factor * n_params), 1)``.  The
        default 1.0 makes the branch exactly as wide as ``theta`` itself, so
        the modulation is never a narrower bottleneck than ``theta``; raise it
        to give the branches more capacity.  Must be positive.
    layer_norm : bool, optional
        Insert an ``eqx.nn.LayerNorm`` after the activation of every activated
        layer (default ``False``).  It normalises the feature axis of ONE
        wavelength, so it never couples the wavelengths or sources evaluated
        together - see the "Optional layer normalization" note above.
    key : jax.Array
        PRNG key used to initialise the layers (``eqx.nn.Linear`` requires
        one).

    Notes
    -----
    * **Cost.**  The kernel is evaluated at every wavelength sample of every
      sub-pixel of every source, and an MLP costs far more per point than the
      analytic blackbody, so this model noticeably slows the forward model
      (and each interleaved amplitude solve, which rebuilds the stamps).  Keep
      ``hidden_size`` modest.  The FiLM branches are cheap by comparison, and
      they are evaluated once per call rather than once per wavelength sample.
      Measured on image generation alone (band 3, 57 sources, 128^2 detector,
      ``n_lambda = 1``, steady state): 146 ms for the blackbody, 199 ms (+36%)
      for this model with ``n_embeddings = 0`` and 202 ms (+38%) with 8.
    * **Branch widths.**  Each FiLM branch is ``P -> W -> (2 * H)`` with
      ``W = max(round(film_hidden_size_factor * P), 1)``, so its arithmetic
      grows like ``W * (P + 2 * H)``: linearly in ``hidden_size`` and
      quadratically in ``P`` (at the default factor 1).
    * **``theta`` has no guaranteed null direction, but it can still lose its
      effect.**  If every ``gamma`` ended up near zero, the hidden layers would
      stop depending on the wavelength and the shape would collapse to a
      constant regardless of ``theta``.  Random initialisation makes that
      unlikely, and the mock's output rescaling (see
      ``scripts/mock_spherex_images.py``) turns it into a conspicuous huge
      rescaling factor, but it is the failure mode to look for if the recovered
      ``theta`` scatters wildly.
    * **No ``exp``, no ``LAMBDA_0`` here.**  The caller subtracts the value at
      ``LAMBDA_0`` and exponentiates afterwards.  Keeping both outside the
      model is what makes a random network safe: its raw offset is
      wavelength-independent and unconstrained by the data, so exponentiating
      it and dividing could overflow, whereas exponentiating the (bounded)
      difference cannot.
    * **Quadrature caution.**  The image generators approximate each pixel's
      bandpass integral with as few as one wavelength sample (``N_LAMBDA = 1``
      in the mock), which is very accurate for a shape that varies slowly
      across the filter but not for a wiggly one.  Fourier features
      deliberately allow wiggles: the finest one has a period of
      ``2 * delta_ln_wavelength / 2**n_embeddings`` in ``ln(lambda)``, i.e. a
      constant *fraction* of the wavelength - about 3% at the defaults.  Since
      a SPHEREx bandpass is also a constant fraction of its central wavelength
      (``sigma / lambda_c = 1 / (2.355 R)``), the shape now varies with roughly
      the same relative richness in every band, and more so in the
      low-resolution bands.  Measured forward-model error (band 3, relative RMS
      of ``n_lambda = 1`` against ``63``, normalized by the image's L2 norm):
      ~1e-6 with ``n_embeddings = 0``, ~6e-6 at 2, ~3e-5 at 4 and ~2e-3 at 8,
      the last being a 28% error in total flux - against ~5e-6 (0%) for a
      3-8 kK blackbody.  Routing ``theta`` through FiLM does not change these
      materially: the modulation changes how strongly the shape bends, not
      which *frequencies* the embedding can express.  Check
      ``--compare-n-lambda`` after changing this model, and either reduce
      ``n_embeddings`` or raise ``n_lambda`` if the residuals are large.
    """

    layers: list
    film: list
    norms: list
    layer_norm: bool
    n_params: int
    n_hidden_layers: int
    hidden_size: int
    film_hidden_layers: int
    film_hidden_size_factor: float
    film_hidden_size: int
    n_embeddings: int
    delta_ln_wavelength: float
    frequencies: jnp.ndarray

    def __init__(
        self,
        n_params: int,
        n_hidden_layers: int,
        hidden_size: int,
        n_embeddings: int = 8,
        delta_ln_wavelength: float = DEFAULT_DELTA_LN_WAVELENGTH,
        film_hidden_layers: int = 1,
        film_hidden_size_factor: float = 1.0,
        layer_norm: bool = False,
        *,
        key: jax.Array,
    ):
        """Initialise the neural network spectrum model.

        Parameters
        ----------
        n_params : int
            Number of spectrum-shape parameters per source.
        n_hidden_layers : int
            Number of hidden layers in the neural network.
        hidden_size : int
            Number of neurons in each hidden layer.
        n_embeddings : int, optional
            Number of Fourier features to add to the input vector (default: 8,
            corresponding to ~128 wavelength elements per e-fold of
            wavelength).
        delta_ln_wavelength : float, optional
            Span of the Fourier embedding in log-wavelength (default:
            ``ln(5.0 / 0.75)``, the full 0.75 - 5.0 um range).
        film_hidden_layers : int, optional
            Hidden layers inside each FiLM branch (default: 1).
        film_hidden_size_factor : float, optional
            Width of each FiLM branch's hidden layers relative to ``theta``
            (default: 1.0, i.e. as wide as ``theta`` itself).
        layer_norm : bool, optional
            Normalise the hidden features of each wavelength (default: False);
            per-wavelength, so it cannot couple batched calls (see the class
            docstring).
        key : jax.Array
            PRNG key for layer initialisation.
        """
        self.n_params = n_params
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.n_embeddings = n_embeddings
        self.delta_ln_wavelength = delta_ln_wavelength
        self.film_hidden_layers = film_hidden_layers
        self.film_hidden_size_factor = float(film_hidden_size_factor)
        self.layer_norm = bool(layer_norm)
        if self.film_hidden_size_factor <= 0.0:
            raise ValueError(
                "film_hidden_size_factor must be positive, got "
                f"{film_hidden_size_factor!r}"
            )

        # FiLM branch width: a MULTIPLE of theta's size, so the modulation can
        # never be a narrower bottleneck than theta itself (see FiLMLayer).
        # Floored at 1 so a small factor cannot produce an empty layer.
        self.film_hidden_size = max(
            int(round(self.film_hidden_size_factor * n_params)), 1
        )

        # Fourier feature frequencies k_i = 2^i * pi / delta_ln_wavelength, so
        # the i-th feature completes 2^(i-1) cycles across
        # delta_ln_wavelength: the lowest is a slow ramp (half a cycle) and the
        # highest resolves the finest log-wavelength structure the embedding
        # can express.  With the default 8 embeddings the top frequency
        # completes 2^6 = 64 cycles over ln(5/0.75), i.e. one cycle per ~128
        # e-foldings of lambda / 64 ~ one cycle per 2% change in lambda.
        # Computed once here rather than on every call.
        self.frequencies = (
            2.0 ** jnp.arange(n_embeddings) * jnp.pi / delta_ln_wavelength
        )

        # One key per Linear layer: input -> hidden, hidden -> hidden (x N),
        # hidden -> 1.  The FiLM branches are initialised from an independent
        # subkey, so that adding or resizing them cannot change the main MLP's
        # weights for a given seed.
        key_main, key_film = jax.random.split(key)
        keys = jax.random.split(key_main, n_hidden_layers + 2)
        input_size = 1 + 2 * n_embeddings      # theta is NOT an input
        layers = [eqx.nn.Linear(input_size, hidden_size, key=keys[0])]
        for i in range(n_hidden_layers):
            layers.append(
                eqx.nn.Linear(hidden_size, hidden_size, key=keys[i + 1])
            )
        layers.append(eqx.nn.Linear(hidden_size, 1, key=keys[-1]))
        self.layers = layers

        # One FiLM branch per ACTIVATED layer, i.e. len(layers) - 1 of them:
        # `self.film[i]` modulates `self.layers[i]`.  The input projection is
        # itself a hidden layer, so with n_hidden_layers = 0 there is still one
        # branch and theta is never ignored.
        film_keys = jax.random.split(key_film, len(layers) - 1)
        self.film = [
            FiLMLayer(
                n_params, film_hidden_layers, self.film_hidden_size,
                hidden_size, key=film_keys[i],
            )
            for i in range(len(layers) - 1)
        ]

        # Optional LayerNorm, one per activated layer and aligned with
        # `self.film`.  eqx's default initialisation is the identity affine
        # (weight = 1, bias = 0), so at this point it is a pure feature-axis
        # standardisation; `weight`/`bias` are nonetheless parameters, so a
        # caller that TRAINS the model trains them too.
        self.norms = (
            [eqx.nn.LayerNorm((hidden_size,)) for _ in range(len(layers) - 1)]
            if self.layer_norm else []
        )

    def _embed_wavelength(self, wavelength: jnp.ndarray) -> jnp.ndarray:
        """Fourier (positional) embedding of ONE wavelength.

        This is part of the scalar core of :meth:`raw_neural_net`, so
        ``wavelength`` is a scalar - the ``vmap`` over wavelengths happens
        outside, in ``__call__``.

        Parameters
        ----------
        wavelength : jnp.ndarray
            A single wavelength in microns (a scalar).

        Returns
        -------
        jnp.ndarray, shape (1 + 2 * n_embeddings,)
            ``ln(lambda)`` followed by ``sin(k_i * ln lambda)`` and
            ``cos(k_i * ln lambda)`` for each frequency ``k_i`` (see the
            ``frequencies`` attribute).

        Notes
        -----
        Everything happens in log-wavelength space, so a feature period is a
        fixed *fraction* of the wavelength and the embedding behaves the same
        way in every band (a bandpass has a constant fractional width).  The
        unembedded ``ln(lambda)`` is kept alongside the features so the network
        can combine a smooth overall trend with the finer structure; without it
        a random first layer would have to synthesise the trend from sinusoids
        whose lowest frequency is already half a cycle across the range.

        Note that with wavelengths in microns, ``ln(lambda)`` is already the
        log-wavelength *relative to LAMBDA_0 = 1 um*, so no extra offset is
        needed: it runs over ``[-0.29, +1.61]`` for the full SPHEREx range.
        """
        ln_wl = jnp.log(jnp.atleast_1d(wavelength))   # (1,)
        angles = self.frequencies * ln_wl             # (n_embeddings,)
        return jnp.concatenate(
            [ln_wl, jnp.sin(angles), jnp.cos(angles)], axis=-1
        )                                            # (1 + 2 n_embeddings,)

    def _film_params(self, shape_params: jnp.ndarray) -> list:
        """``(gamma, beta)`` for every activated layer, from ``theta``.

        Depends only on the shape parameters - never on wavelength - so
        :meth:`__call__` evaluates this ONCE and reuses the result for every
        wavelength sample, instead of recomputing the branches inside the
        per-wavelength ``vmap``.

        Returns
        -------
        list of (jnp.ndarray, jnp.ndarray)
            One ``(gamma, beta)`` pair per activated layer, aligned with
            ``self.layers[:-1]``; each array has shape ``(hidden_size,)``.
        """
        return [branch(shape_params) for branch in self.film]

    def _forward(
        self,
        wavelength_features: jnp.ndarray,
        film_params: list,
    ) -> jnp.ndarray:
        """Log-flux kernel for ONE wavelength of ONE source.

        ``wavelength_features`` is an embedded wavelength (see
        :meth:`_embed_wavelength`) and ``film_params`` the pairs returned by
        :meth:`_film_params`.  Passing them in rather than recomputing keeps
        this a pure function of ``(features, film)``, which is what lets
        :meth:`__call__` ``vmap`` over wavelengths without re-running the FiLM
        branches each time.

        When ``layer_norm`` is on, each activated layer is followed by its
        normalization.  Both operations act on a SINGLE wavelength's feature
        vector, which is what keeps the result independent of the rest of the
        batch.
        """
        x = wavelength_features
        for i, (layer, (gamma, beta)) in enumerate(
            zip(self.layers[:-1], film_params)
        ):
            x = jax.nn.silu(gamma * layer(x) + beta)
            if self.layer_norm:
                x = self.norms[i](x)
        return jnp.squeeze(self.layers[-1](x), axis=-1)

    def raw_neural_net(
        self,
        wavelength: jnp.ndarray,
        shape_params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Log-flux kernel for ONE wavelength of ONE source.

        ``wavelength`` is a scalar and ``shape_params`` a ``(n_params,)``
        vector; the result is a scalar log-flux, which is what ``__call__``
        returns per wavelength.  This is the self-contained scalar path (it
        computes its own FiLM parameters); :meth:`__call__` instead computes
        them once for the whole wavelength vector and calls :meth:`_forward`.
        The two agree exactly.
        """
        return self._forward(
            self._embed_wavelength(wavelength),
            self._film_params(shape_params),
        )

    def __call__(
        self,
        wavelength: jnp.ndarray,
        shape_params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Log-flux kernel for ONE source, at many wavelengths.

        Parameters
        ----------
        wavelength : jnp.ndarray, shape (N_lambda,)
            Wavelength(s) in microns.
        shape_params : jnp.ndarray, shape (n_params,)
            Shape parameters of a single source.

        Returns
        -------
        jnp.ndarray, shape (N_lambda,)
            ``log f_lambda`` up to a wavelength-independent constant.  Subtract
            the value at :data:`LAMBDA_0` to obtain the dimensionless shape,
            which is exactly 1 there - use
            :func:`normalized_log_shape` / :func:`normalized_shape`, or better,
            fold the constant into the amplitude once per source with
            :func:`normalized_source_params`.
        """
        wavelength = jnp.atleast_1d(wavelength)   # (N_lambda,)

        # theta -> (gamma, beta) once per call: the FiLM parameters are
        # wavelength-independent, so they must not be recomputed inside the
        # per-wavelength vmap (``in_axes=None`` broadcasts the nested pytree
        # of (gamma, beta) pairs over the wavelength axis).
        film_params = self._film_params(shape_params)
        features = jax.vmap(self._embed_wavelength)(wavelength)
        return jax.vmap(self._forward, in_axes=(0, None))(
            features, film_params
        )


class BlackbodyPlusNNSpectrum(eqx.Module):
    """Blackbody continuum modulated by a neural-network template.

    A composite of :class:`BlackbodySpectrum` and :class:`NeuralNetSpectrum`
    that SPLITS the shape parameters between them: ``theta[0]`` is the
    blackbody's log-temperature, ``theta[1:]`` are the network's parameters.
    The two log-flux kernels are added::

        log_kernel(lambda; theta) = BB(lambda; theta[0]) + NN(lambda; theta[1:])

    which in linear terms multiplies the blackbody by the network's
    dimensionless modulation - the familiar "continuum plus flexible
    correction" model, in which the network absorbs whatever the blackbody
    cannot describe.  It needs no new machinery: both parts obey the log-flux
    contract (see the module docstring), so their sum does too, and the
    caller's normalisation at :data:`LAMBDA_0` still gives a shape of exactly 1
    there.

    ``n_params = 1 + neural_net.n_params``.  Note that the blackbody's own tilt
    is normally the *larger* contribution to the total log-shape (a 3-8 kK
    blackbody has a per-source log-shape std of ~1.75 over 0.75-5 um, against
    ~0.5 for the mock's rescaled random network), so the network acts as a
    modulation *on top of* the continuum rather than as the whole spectrum.
    The split is positional, so anything that draws, initialises, labels or
    reports these parameters must treat column 0 as ``log T`` - the mock does
    (see ``_model_blocks`` / ``_draw_shape_params``).

    Parameters
    ----------
    neural_net : NeuralNetSpectrum
        The network part, already constructed - and, if its output is to be
        rescaled, already rescaled: such a rescaling modifies the network's own
        output layer, so it has to happen BEFORE the wrapping.
    blackbody : BlackbodySpectrum, optional
        The continuum part; a fresh :class:`BlackbodySpectrum` if omitted.

    Notes
    -----
    Batching follows :class:`NeuralNetSpectrum`: ONE source per call, many
    wavelengths.  To batch over sources, compose a ``vmap`` at the call site::

        jax.vmap(model, in_axes=(None, 0))(wavelengths, shape_params_batch)

    Raises
    ------
    ValueError
        If the network part has no shape parameters of its own: the blackbody
        consumes ``theta[0]``, so the composite needs at least two in total.
    """

    blackbody: BlackbodySpectrum
    neural_net: NeuralNetSpectrum
    n_params: int

    def __init__(
        self,
        neural_net: NeuralNetSpectrum,
        blackbody: BlackbodySpectrum = None,
    ):
        if neural_net.n_params < 1:
            raise ValueError(
                "the network part needs at least one shape parameter of its "
                f"own (got n_params={neural_net.n_params}): the blackbody "
                "takes theta[0], so the composite needs n_params >= 2"
            )
        self.neural_net = neural_net
        self.blackbody = (
            BlackbodySpectrum() if blackbody is None else blackbody
        )
        self.n_params = 1 + neural_net.n_params

    def __call__(
        self,
        wavelength: jnp.ndarray,
        shape_params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Log-flux kernel for ONE source, at many wavelengths.

        Parameters
        ----------
        wavelength : jnp.ndarray, shape (N_lambda,)
            Wavelength(s) in microns.
        shape_params : jnp.ndarray, shape (n_params,)
            ``shape_params[0]`` = log(temperature / kK) for the blackbody;
            ``shape_params[1:]`` go to the network.

        Returns
        -------
        jnp.ndarray, shape (N_lambda,)
            ``log f_lambda`` up to a wavelength-independent constant, i.e. the
            shared sum of the blackbody and network kernels.
        """
        return (self.blackbody(wavelength, shape_params[:1])
                + self.neural_net(wavelength, shape_params[1:]))