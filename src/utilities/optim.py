import numpy as np


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
