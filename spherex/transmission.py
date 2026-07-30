"""Filter transmission models for SPHEREx photometry.

The SPHEREx linear-variable filter (LVF) has a transmission profile
whose central wavelength varies approximately linearly with position
along one detector axis.

Interface contract
-------------------
Every transmission model used by the image-generation code must
implement, in addition to ``__call__``:

- ``quantile(q, omega_p) -> lambda``: the inverse CDF of the (per
  ``omega_p``) *normalised* transmission profile, i.e. the wavelength
  below which a fraction ``q`` of ``T(lambda | omega_p)``'s integral
  lies.
- ``total_transmission(omega_p) -> scalar``: ``integral T(lambda |
  omega_p) d(lambda)``, the normalising constant of the profile.

These are used to numerically integrate over the local filter bandpass
via quantile / inverse-CDF importance sampling (see
``spherex.image._one_subpixel_rate``), which is far more accurate than
naive fixed-grid quadrature at low sample counts, since it concentrates
wavelength samples where the transmission profile actually has support
instead of wasting them in its tails.
"""

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jss


class GaussianFilterTransmission(eqx.Module):
    """Gaussian filter transmission with linearly-varying central wavelength.

    The central wavelength at detector position ``(x_p, y_p)`` is::

        lambda_c(y_p) = lambda_intercept + lambda_slope * y_p

    and the transmission at wavelength ``lambda`` is::

        T(lambda | omega_p) = exp( -(lambda - lambda_c)^2 / (2 * width^2) )

    Parameters
    ----------
    lambda_intercept : float
        Central wavelength at ``y_p = 0``, in microns.
    lambda_slope : float
        Rate of change of central wavelength with detector y-position,
        in microns / arcsec.
    width : float
        Gaussian sigma of the transmission profile, in microns.

    Call signature
    --------------
    __call__(wavelength, omega_p) -> dimensionless transmission in [0, 1]

    wavelength : jnp.ndarray, shape (...,)
        Wavelength(s) in microns.
    omega_p : jnp.ndarray, shape (..., 2)
        Detector-plane positions in arcsec.
    """

    lambda_intercept: float
    lambda_slope: float
    width: float

    def central_wavelength(self, omega_p: jnp.ndarray) -> jnp.ndarray:
        """Return the central wavelength at detector position(s) ``omega_p``.

        Parameters
        ----------
        omega_p : jnp.ndarray, shape (..., 2)
            Detector-plane positions in arcsec.

        Returns
        -------
        jnp.ndarray, shape (...,)
            Central wavelength(s) in microns.
        """
        y_p = omega_p[..., 1]   # linear ramp is along y-axis
        return self.lambda_intercept + self.lambda_slope * y_p

    def __call__(
        self,
        wavelength: jnp.ndarray,
        omega_p: jnp.ndarray,
    ) -> jnp.ndarray:
        lam_c = self.central_wavelength(omega_p)            # (...)
        delta = (wavelength - lam_c) / self.width           # (...)
        return jnp.exp(-0.5 * delta**2)

    def quantile(
        self,
        q: jnp.ndarray,
        omega_p: jnp.ndarray,
    ) -> jnp.ndarray:
        """Inverse CDF of the normalised transmission profile.

        Since ``T(lambda | omega_p)`` is (up to normalisation) a Gaussian
        density in ``lambda`` with mean ``central_wavelength(omega_p)``
        and standard deviation ``width``, its inverse CDF has the usual
        closed form in terms of the inverse error function.

        Parameters
        ----------
        q : jnp.ndarray, shape (...,)
            Quantile(s) in ``(0, 1)``.
        omega_p : jnp.ndarray, shape (..., 2)
            Detector-plane position(s) in arcsec.  Broadcasts against
            ``q``.

        Returns
        -------
        jnp.ndarray
            Wavelength(s) in microns such that a fraction ``q`` of the
            transmission profile's integral lies below it.
        """
        lam_c = self.central_wavelength(omega_p)
        return lam_c + self.width * jnp.sqrt(2.0) * jss.erfinv(2.0 * q - 1.0)

    def total_transmission(self, omega_p: jnp.ndarray) -> jnp.ndarray:
        """Total integral of the transmission profile over wavelength.

        ``integral T(lambda | omega_p) d(lambda) = sqrt(2 pi) * width``
        for a Gaussian profile.  Returned with the same broadcast shape
        as ``central_wavelength(omega_p)`` (i.e. ``omega_p``'s leading
        shape), even though ``width`` is currently a scalar constant,
        for consistency with a possible future spatially-varying width.

        Parameters
        ----------
        omega_p : jnp.ndarray, shape (..., 2)
            Detector-plane position(s) in arcsec.

        Returns
        -------
        jnp.ndarray
            The normalising constant of the transmission profile.
        """
        lam_c = self.central_wavelength(omega_p)
        return jnp.sqrt(2.0 * jnp.pi) * self.width * jnp.ones_like(lam_c)

