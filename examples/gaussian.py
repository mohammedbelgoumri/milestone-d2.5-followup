import jax.numpy as jnp
from chex import Array


def gaussian_mse_loss_fn(theta: Array, minima: Array, sigma: float = 0.1) -> Array:
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
