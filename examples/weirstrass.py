import numpy as np

from utilities.optim import rolling_ball_trajectory


def loss_fn(x, n=100):
    """
    Weirstrass-like loss function.
    See https://github.com/mohammedbelgoumri/milestone-d-2.5 for details.
    """
    return np.sum([np.sin(i**2 * x) / i**2 for i in range(1, n + 1)], axis=0)


def grad_fn(x, n=100):
    """
    Hand computed gradient of the Weirstrass-like loss function.
    """
    return np.sum([np.cos(i**2 * x) / i**2 for i in range(1, n + 1)], axis=0)


def main():
    import matplotlib.pyplot as plt

    lr = 0.001
    n_epochs = 2000
    n_projections = 20
    projection_step = 0.05
    initial = 2.0

    n_radii = 5
    configs = {
        "lr": [lr] * n_radii,
        "radius": np.logspace(-2, 0, n_radii),
        "n_epochs": [n_epochs] * n_radii,
        "n_projections": [n_projections] * n_radii,
        "projection_step": [projection_step] * n_radii,
        "initial": [initial] * n_radii,
    }

    x = np.linspace(0, 2.1, 1000)
    y = loss_fn(x)

    fig, ax = plt.subplots(ncols=n_radii + 1, figsize=(20, 5), sharey=True, sharex=True)

    for i, config in enumerate(zip(*configs.values())):
        lr, radius, n_epochs, n_projections, projection_step, initial = config
        params, centers = rolling_ball_trajectory(
            value_and_grad_fn=lambda x: (loss_fn(x), grad_fn(x)),
            lr=lr,
            radius=radius,
            n_epochs=n_epochs,
            n_projections=n_projections,
            projection_step=projection_step,
            initial=initial,
        )
        ax[i + 1].set_title(f"Radius: {radius:.4f}")
        ax[i + 1].plot(centers[:, 0], centers[:, 1])
        ax[i + 1].set_xlabel("Parameter")
        ax[i + 1].set_ylabel("Loss")
        ax[i + 1].set_aspect("equal")
    ax[0].plot(x, y)
    ax[0].set_title("Original Loss Landscape (Radius = 0)")
    ax[0].set_xlabel("Parameter")
    ax[0].set_ylabel("Loss")
    ax[0].set_aspect("equal")
    plt.tight_layout()
    plt.savefig("weirstrass_trajectory.pdf")
    plt.show()


if __name__ == "__main__":
    main()
