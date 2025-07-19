import jax.numpy as jnp
import numpy as np
from jax import jit, tree, value_and_grad

from utilities.spatial import project_on_graph, tree_norm


def ball_trajectory(
    loss_fn,
    params0,
    radius=0.5,
    n_steps=100,
    learning_rate=0.1,
    projection_step_size=0.01,
    n_projection_steps=10,
):
    @jit
    def update(params):
        value_and_grad_fn = value_and_grad(loss_fn)
        fx, grads = value_and_grad_fn(params)
        g2 = tree_norm(grads) ** 2
        normal = (tree.map(lambda g: -g, grads), 1.0)
        normal = tree.map(lambda n: n / jnp.sqrt(g2 + 1), normal)
        tangent = (grads, g2)

        center = tree.map(lambda p, n: p + radius * n, (params, fx), normal)
        candidate = tree.map(lambda u, v: u - learning_rate * v, center, tangent)
        footpoint = project_on_graph(
            value_and_grad_fn,
            candidate,
            x0=params,
            step_size=projection_step_size,
            n_steps=n_projection_steps,
        )
        return footpoint

    trajectory = [params0]
    for _ in range(n_steps):
        params = trajectory[-1]
        params = update(params)
        trajectory.append(params)
    return trajectory


def rolling_ball_trajectory(
    value_and_grad_fn, lr, radius, n_epochs, n_projections, projection_step, initial=2.0
):
    """
    This function retruns RBO trajectory for a given (1-D) loss function.
    """
    params = [initial]
    centers = []
    for epoch in range(n_epochs):
        param = params[-1]
        value, grad = value_and_grad_fn(param)
        point = np.array([param, value])
        normal = np.array([1, -grad])
        normal /= np.linalg.norm(normal)
        tangent = np.array([grad, np.linalg.norm(grad) ** 2])
        center = point - radius * normal
        centers.append(center)
        candidate_center = center.copy()
        candidate_center -= lr * tangent
        # Project the candidate center onto the loss function
        footpoint = center[0]
        for p_step in range(n_projections):
            ff, grad = value_and_grad_fn(footpoint)
            distance_gradient = (footpoint - candidate_center[0]) + grad * (
                ff - candidate_center[1]
            )
            footpoint -= projection_step * distance_gradient
        params.append(footpoint)
    params = np.array(params)
    centers = np.array(centers)
    return params, centers
