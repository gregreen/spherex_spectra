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


class NeuralNetSpectrum(eqx.Module):
    """Neural-network spectrum template (unnormalised log-flux).

    A flexible alternative to :class:`BlackbodySpectrum`.  A small MLP maps
    ``(wavelength, shape_params)`` to a log-flux, and ``__call__`` returns it
    directly::

        log_kernel(lambda) = NN(lambda, theta)

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

    Internally the MLP is evaluated one wavelength at a time and vectorised
    with ``jax.vmap``.  That keeps the network definition simple: an MLP layer
    expects the feature axis last, so ``concatenate([lambda, theta])`` is only
    well defined for a single wavelength (a wavelength *vector* would be read
    as extra features).  To batch over sources as well, compose a second
    ``vmap`` at the call site::

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
        ``ln(lambda)`` only.
    delta_ln_wavelength : float, optional
        Span of the embedding in log-wavelength, i.e. the range of
        ``ln(lambda)`` the Fourier features should cover (default
        ``ln(5.0 / 0.75)``, the full 0.75 - 5.0 um SPHEREx range).
    key : jax.Array
        PRNG key used to initialise the layers (``eqx.nn.Linear`` requires
        one).

    Notes
    -----
    * **Cost.**  The kernel is evaluated at every wavelength sample of every
      sub-pixel of every source, and an MLP costs far more per point than the
      analytic blackbody, so this model noticeably slows the forward model
      (and each interleaved amplitude solve, which rebuilds the stamps).  Keep
      ``hidden_size`` modest.
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
      low-resolution bands.  Measured forward-model error (band 3, relative
      RMS of ``n_lambda = 1`` against ``63``): ~1e-6 with ``n_embeddings = 0``,
      ~1e-5 at 2, ~1e-4 at 4 and ~4e-3 (worst pixels ~10%) at 8.  Check
      ``--compare-n-lambda`` after changing this model, and either reduce
      ``n_embeddings`` or raise ``n_lambda`` if the residuals are large.
    """

    layers: list
    n_params: int
    n_hidden_layers: int
    hidden_size: int
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
        key : jax.Array
            PRNG key for layer initialisation.
        """
        self.n_params = n_params
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.n_embeddings = n_embeddings
        self.delta_ln_wavelength = delta_ln_wavelength

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
        # hidden -> 1.
        keys = jax.random.split(key, n_hidden_layers + 2)
        input_size = n_params + 1 + 2 * n_embeddings
        layers = [eqx.nn.Linear(input_size, hidden_size, key=keys[0])]
        for i in range(n_hidden_layers):
            layers.append(
                eqx.nn.Linear(hidden_size, hidden_size, key=keys[i + 1])
            )
        layers.append(eqx.nn.Linear(hidden_size, 1, key=keys[-1]))
        self.layers = layers

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

    def raw_neural_net(
        self,
        wavelength: jnp.ndarray,
        shape_params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Log-flux kernel for ONE wavelength of ONE source.

        ``wavelength`` is a scalar and ``shape_params`` a ``(n_params,)``
        vector; the result is a scalar log-flux, which is what ``__call__``
        returns per wavelength.
        """
        wl_features = self._embed_wavelength(wavelength)
        x = jnp.concatenate(
            [wl_features, shape_params], axis=-1
        )
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)

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

        # The network consumes a single feature vector, so map over wavelengths.
        return jax.vmap(self.raw_neural_net, in_axes=(0, None))(
            wavelength, shape_params
        )