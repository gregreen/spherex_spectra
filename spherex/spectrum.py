"""Source spectrum models for SPHEREx photometry.

Each spectrum model is an ``equinox.Module`` that acts as a *parameterised
template* describing the **shape** of a source spectrum: it stores no
per-source state, and its ``__call__`` method accepts explicit parameter
arrays so that gradients flow through them naturally.

Amplitude convention
--------------------
The overall flux-density scale (``log_amplitude``) is deliberately *not*
part of a spectrum model.  It is carried as a separate per-source parameter,
always stored as the **first** column of ``source_params`` (see
:data:`LOG_AMPLITUDE_INDEX`).  The image generators apply it *outside* the
spectrum model::

    f_lambda = exp(source_params[:, 0]) * spectrum_model(lambda, source_params[:, 1:])

so ``spectrum_model`` only ever describes the spectral *shape*.  For
:class:`BlackbodySpectrum` the shape is normalised to ``f_lambda = 1`` at
:data:`LAMBDA_0`, so ``log_amplitude`` has a direct physical meaning: it is
the log of the source's flux density at ``LAMBDA_0``.

Splitting the amplitude out this way lets the optimiser solve for the optimal
amplitude of every source *directly*: the forward model is exactly linear in
amplitude, so it becomes a linear least-squares problem (see
``scripts.inference.solve_log_amplitudes``).
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .constants import HC_JAX, KB_JAX

# ---------------------------------------------------------------------------
# Amplitude / normalisation conventions
# ---------------------------------------------------------------------------

#: Reference wavelength (microns) at which every spectrum model's shape is
#: normalised to ``f_lambda = 1``.  This is a single GLOBAL constant shared by
#: all bands so that the modelled spectrum stays smooth in amplitude across
#: the whole wavelength range: changing band only changes the *sampling* of
#: the shape, never its normalisation.
LAMBDA_0 = 1.0

#: Index of the log-amplitude column in ``source_params``.  By convention the
#: amplitude is ALWAYS the first parameter; the remaining columns
#: (``source_params[:, 1:]``) are the ones passed to the spectrum model.
LOG_AMPLITUDE_INDEX = 0


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


class BlackbodySpectrum(eqx.Module):
    """Normalised blackbody spectrum template (shape only).

    Returns the Planck function normalised to ``f_lambda = 1`` at the
    reference wavelength ``lambda_0``::

        shape(lambda) = B_lambda(lambda, T) / B_lambda(lambda_0, T)

    The result is dimensionless; the source's physical amplitude is supplied
    separately as the first column of ``source_params`` (see the module
    docstring).  The 2hc^2 prefactor cancels in the ratio, so the returned
    shape is::

        (lambda_0 / lambda)^5 * (exp(x_0) - 1) / (exp(x) - 1)

    with ``x = hc / (lambda * k_B * T)``.

    Parameters
    ----------
    wavelength : jnp.ndarray, shape (...,)
        Wavelength(s) in microns.
    shape_params : jnp.ndarray, shape (..., 1)
        Per-source shape parameters in **log-space**.
        ``shape_params[..., 0]`` = log(temperature / kK).

    Attributes
    ----------
    lambda_0 : float
        Reference wavelength (microns).  Defaults to the global constant
        :data:`LAMBDA_0` and should be the same for every band, so that the
        normalisation does not introduce band-to-band discontinuities.
    n_params : int
        Number of spectrum-*shape* parameters (1: log-temperature).  The
        amplitude is not counted here (it is held outside the model).
    """

    lambda_0: float = LAMBDA_0
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
        inv_T_scale = HC_JAX / (wavelength * KB_JAX)             # K
        inv_T_scale_0 = HC_JAX / (self.lambda_0 * KB_JAX)        # K

        exponent = inv_T_scale / temperature                      # dimensionless
        exponent_0 = inv_T_scale_0 / temperature                  # dimensionless

        # Guard exp overflow (short wavelengths / low temperatures).
        exponent = jnp.clip(exponent, None, 50.0)
        exponent_0 = jnp.clip(exponent_0, None, 50.0)

        # Ratio of Planck functions; the 2 h c^2 / lambda^5 prefactor cancels.
        shape = (
            (self.lambda_0 / wavelength) ** 5
            * jnp.expm1(exponent_0)
            / jnp.expm1(exponent)
        )
        return shape


class NeuralNetSpectrum(eqx.Module):
    """Neural-network spectrum template (shape only).

    A flexible alternative to :class:`BlackbodySpectrum`.  A small MLP maps
    ``(wavelength, shape_params)`` to a log-flux, and the returned shape is the
    exponential of the difference between that log-flux and its value at
    :data:`LAMBDA_0`::

        shape(lambda) = exp( NN(lambda, theta) - NN(LAMBDA_0, theta) )

    which is exactly 1 at ``LAMBDA_0`` for *any* parameters.  That matches the
    normalisation convention of :class:`BlackbodySpectrum`, so the amplitude
    (the first column of ``source_params``) keeps its meaning as
    ``f_lambda(LAMBDA_0)``.  The amplitude machinery needs no changes: it only
    requires the shape to be independent of the amplitude and the model to be
    *linear in amplitude*, both of which hold here (see
    ``scripts.inference.solve_log_amplitudes``).

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
    key : jax.Array
        PRNG key used to initialise the layers (``eqx.nn.Linear`` requires
        one).

    Notes
    -----
    * **Cost.**  The shape is evaluated at every wavelength sample of every
      sub-pixel of every source, and an MLP costs far more per point than the
      analytic blackbody, so this model noticeably slows the forward model
      (and each interleaved amplitude solve, which rebuilds the stamps).  Keep
      ``hidden_size`` modest.
    * **Normalisation cost.**  Every call also evaluates the network once at
      ``LAMBDA_0``, so with ``n_wavelength_samples = 1`` the MLP runs twice per
      point.  That is the price of exact normalisation without extra state.
    * The network output is unbounded, so ``exp`` of a large log-ratio can
      overflow; clip the exponent if the shape parameters are allowed to
      wander far from their initialisation.
    """

    layers: list
    n_params: int
    n_hidden_layers: int
    hidden_size: int

    def __init__(
        self,
        n_params: int,
        n_hidden_layers: int,
        hidden_size: int,
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
        key : jax.Array
            PRNG key for layer initialisation.
        """
        self.n_params = n_params
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size

        # TODO: Spatial embedding: add hand-rolled Fourier features of the
        # wavelength to the input vector:
        # 
        #    $\sin(k_i \lambda)$ and $cos(k_i \lambda)$
        #    for $k_i = 2^i \pi / \Delta\lambda$ for $i=0,1,...,n_{freqs}-1$.
        # 
        # Model hyperparameters:
        #    * n_freqs: number of Fourier features to add (default = 8,
        #               corresponding to ~128 wavelength elements)
        #    * delta_lambda: wavelength range to cover (default = 4.25 um,
        #                    corresponding to 0.75 - 5.0 microns)

        # One key per Linear layer: input -> hidden, hidden -> hidden (x N),
        # hidden -> 1.
        keys = jax.random.split(key, n_hidden_layers + 2)
        layers = [eqx.nn.Linear(n_params + 1, hidden_size, key=keys[0])]
        for i in range(n_hidden_layers):
            layers.append(
                eqx.nn.Linear(hidden_size, hidden_size, key=keys[i + 1])
            )
        layers.append(eqx.nn.Linear(hidden_size, 1, key=keys[-1]))
        self.layers = layers

    def raw_neural_net(
        self,
        wavelength: jnp.ndarray,
        shape_params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Log-flux for ONE wavelength of ONE source.

        ``wavelength`` is a scalar and ``shape_params`` a ``(n_params,)``
        vector; the result is a scalar log-flux, i.e. the network's raw output
        before the ``LAMBDA_0`` normalisation applied in ``__call__``.
        """
        x = jnp.concatenate(
            [jnp.atleast_1d(wavelength), shape_params], axis=-1
        )
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)

    def __call__(
        self,
        wavelength: jnp.ndarray,
        shape_params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Spectrum shape for ONE source, at many wavelengths.

        Parameters
        ----------
        wavelength : jnp.ndarray, shape (N_lambda,)
            Wavelength(s) in microns.
        shape_params : jnp.ndarray, shape (n_params,)
            Shape parameters of a single source.

        Returns
        -------
        jnp.ndarray, shape (N_lambda,)
            Dimensionless shape, equal to exactly 1 at :data:`LAMBDA_0`.
        """
        wavelength = jnp.atleast_1d(wavelength)   # (N_lambda,)

        # The network consumes a single feature vector, so map over wavelengths.
        ln_flux = jax.vmap(self.raw_neural_net, in_axes=(0, None))(
            wavelength, shape_params
        )
        # Reference log-flux at LAMBDA_0: independent of wavelength, so
        # hoisting it out of the vmap costs nothing extra.
        ln_flux_ref = self.raw_neural_net(LAMBDA_0, shape_params)
        return jnp.exp(ln_flux - ln_flux_ref)      # == 1 at LAMBDA_0