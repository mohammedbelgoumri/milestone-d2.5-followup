"""
This module contains synthetic loss functions that are easy to compute,
but difficult to optimize.
These functions are used in our experiments to compare the rolling ball optimizer
to standard optimizers.
"""

from typing import Sequence

import jax.numpy as jnp
import numpy as np
from chex import Array
from jax import vmap


def weierstrass_loss_fn(
    theta: Array, a: float = 0.5, b: int = 3, n: int = 100
) -> Array:
    """The (nth parial sum of the) Weierstrass function in arbitrary dimensions

    Args:
        theta (Array): A 2D array of shape (n_points, n_dims)
        a (float, optional): The a parameter of the Weierstrass function\
            controls how rapidly higher harmonics decay.\
            Must be between 0 and 1, use larger values for rougher functions.\
                Defaults to 0.5.
        b (int, optional): The b parameter of the Weierstrass function,\
            controls how quickly the frequencies increase. Must be an odd integer.\
            Use larger values for rougher functions.  Defaults to 3.
        n (int, optional): The number of harmonics to sum.\
            Larger values result in rougher functions. Defaults to 100.

    Returns:
        Array: An array of shape (n_points,)\
            containing the values of the Weierstrass function at each point in x
    """
    # x of shape (n_points, n_dims)
    if theta.ndim == 1:
        theta = theta[:, None]

    # Exponenents of shape (n + 1, 1, 1) (repeated for each dimension)
    ks = jnp.arange(n + 1).reshape(-1, 1, 1)
    aks = a**ks  # Amplitude coefficients of shape (n + 1, 1, 1)
    bks = b**ks  # Frequency coefficients of shape (n + 1, 1, 1)
    angles = bks * jnp.pi * theta  # Angles of shape (n + 1, n_points, n_dims)
    terms = aks * jnp.cos(angles)  # Terms of shape (n + 1, n_points, n_dims)
    subsums = jnp.sum(terms, axis=0)  # Subsums of shape (n_points, n_dims)
    return jnp.sum(subsums, axis=1)  # Sum of shape (n_points,)


def weierstrass_like_loss_fn(theta: Array, n: int = 100) -> Array:
    """The (nth partial sum of the) Weierstrass-like function defined as
    ```math
    f_n(\\theta) = \\sum_{k=0}^{n} \frac{1}{k^2} \\sin{\\left(k^2 x\\right)}
    ```

    Args:
        theta (Array): A 2D array of shape (n_points, n_dims)
        n (int, optional): The number of harmonics to sum. Defaults to 100.

    Returns:
        Array: Output of shape (n_points,)
    """
    if theta.ndim == 1:
        theta = theta[:, None]

    ks = jnp.arange(1, n + 1).reshape(-1, 1, 1)
    aks = 1 / ks**2
    angles = ks**2 * theta
    terms = aks * jnp.sin(angles)
    subsums = jnp.sum(terms, axis=0)
    return jnp.sum(subsums, axis=1)


def whitley_loss_fn(theta: Array) -> Array:
    """Whitley function defined as
    ```math
    f(\\theta) = \\sum_{i=1}^{n} \\sum_{j=1}^{n}\
    \\left(100 * (i^2 - j)^2 + (1 - j)^2\\right)^2 / 4000 - \
    \\cos\\left(\\sqrt{\\left(100 * (i^2 - j)^2 + (1 - j)^2\\right)} + 1\\right)
    ```

    Args:
        theta (Array): The parameter of shape (n_points, n_dims)

    Returns:
        Array: Output of shape (n_points,)
    """
    if theta.ndim == 1:
        theta = theta[:, None]  # shape (n_points, 1)

    def whitley_single(thi):
        thi = thi.reshape(-1)  # shape (n_dims,)
        i = thi[None, :]  # shape (1, n_dims)
        j = thi[:, None]  # shape (n_dims, 1)
        term = 100 * (i**2 - j) ** 2 + (1 - j) ** 2
        return jnp.sum((term**2 / 4000 - jnp.cos(term + 1)).reshape(-1))

    return vmap(whitley_single)(theta)  # shape (n_points,)


def sine_mse_loss_fn(
    theta: tuple[float, float], x: Sequence[float], y: Sequence[float]
) -> float:
    """The MSE loss function for the 2-D sinusoidal model.
    This function uses the model `y = theta_2 * sin(theta_1 * x)`
    to generate predictions and then returns the mean squared error.

    Args:
    theta: tuple[float, float], the parameters of the model
    x: Sequence[float], training features
    y: Sequence[float], training labels
    Returns:
        float: The loss
    """
    theta1, theta2 = theta
    ŷ = theta2 * jnp.sin(theta1 * x)
    return jnp.mean((ŷ - y) ** 2)


np.random.seed(0)
minima = np.random.uniform(-3, 3, (1000, 2))


def gaussian_mse_loss_fn(
    theta: Array, minima: Array = minima, sigma: float = 0.1
) -> Array:
    """Gaussian quadratic loss function
    This function is parameterized by a set of local minima and a width parameter.
    Given an input theta, it will return a sum of quadratics,
    minimized at each of the local minima.

    Args:
        theta (Array): Parameter.
        minima (Array, optional): Specified local minima.
        sigma (Array, optional): The width of each local minimum's basin.

    Returns:
        Array: The loss
    """
    diff = theta - minima
    r2 = jnp.sum(diff**2, axis=-1)
    return -jnp.sum(jnp.exp(-r2 / (2 * sigma**2))) + jnp.linalg.norm(theta) ** 2
