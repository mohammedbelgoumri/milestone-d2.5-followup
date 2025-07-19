import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns

n = 100


def loss_fn(x):
    return np.sum([np.sin(i**2 * x) / i**2 for i in range(1, n + 1)], axis=0)
    # return x**2


def grad_fn(x):
    return np.sum([np.cos(i**2 * x) / i**2 for i in range(1, n + 1)], axis=0)
    # return 2 * x


def get_points(lr, radius, n_epochs, n_projections, projection_step):
    xs = [2.0]
    centers = []
    for epoch in range(n_epochs):
        current_x = xs[-1]
        current_y = loss_fn(current_x)
        current_point = np.array([current_x, current_y])
        grad = grad_fn(current_x)
        normal = np.array([1, -grad])
        normal /= np.linalg.norm(normal)
        tangent = np.array([grad, np.linalg.norm(grad) ** 2])
        center = current_point - radius * normal
        centers.append(center)
        candidate_center = center.copy()
        candidate_center -= lr * tangent
        # Project the candidate center onto the loss function
        footpoint = center[0]
        for projection in range(n_projections):
            ff = loss_fn(footpoint)
            grad = grad_fn(footpoint)
            dist_grad = (footpoint - candidate_center[0]) + grad * (
                ff - candidate_center[1]
            )
            footpoint -= projection_step * dist_grad
        xs.append(footpoint)
    xs = np.array(xs)
    centers = np.array(centers)
    return xs, centers


def main():
    lr = 0.001
    radius = 0.3
    n_epochs = 2000
    n_projections = 20
    projection_step = 0.05
    n_radii = 5
    configs = {
        "lr": [lr] * n_radii,
        "radius": np.logspace(-2, 0, n_radii),
        "n_epochs": [n_epochs] * n_radii,
        "n_projections": [n_projections] * n_radii,
        "projection_step": [projection_step] * n_radii,
    }

    configs = [{k: v[i] for k, v in configs.items()} for i in range(n_radii)]

    # sns.set_theme("paper")
    plt.rcParams.update({"mathtext.fontset": "cm"})
    x = np.linspace(0, 2.1, 1000)
    y = loss_fn(x)
    # fig, ax = plt.subplots(ncols=n_radii + 1)
    # plt.axis("equal")

    for i, config in enumerate(configs):
        fig = plt.figure()
        ax = fig.add_subplot()
        xs, centers = get_points(**config)
        ax.plot(centers[:, 0], centers[:, 1])
        # ax[i + 1].set_xlabel(r"$\theta$")
        # ax[i + 1].set_ylabel(r"$f(\theta)$")
        # ax[(i + 1) // 2, (i + 1) % 2].legend()
        ax.set_title(f"Radius = {config['radius']:.4f}", fontsize=30)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(20)
        plt.tight_layout()
        # ax.set_aspect("equal")
        plt.savefig(f"smoothingt{i}.pdf")

    fig = plt.figure()
    ax = fig.add_subplot()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(20)
    ax.plot(x, y)

    # ax[0].set_xlabel(r"$\theta$", fontsize=16)
    # ax[0].set_ylabel(r"$f(\theta)$", fontsize=16)
    # ax[0, 0].legend()
    ax.set_title("Original landscape", fontsize=30)
    plt.tight_layout()
    ax.set_aspect("equal")
    # plt.title("Loss function")
    # plt.tight_layout()
    plt.savefig("original.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
