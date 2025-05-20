import matplotlib.pyplot as plt
import numpy as np
from data import retrieve_dataset
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_points(
    data: np.ndarray | list[np.ndarray],
    title: str = "3D Curve",
    line: bool = False,
    ax: Axes3D | None = None,
):
    """Plot 3D points.

    Args:
        data (np.ndarray): The input array of shape (x_i, y_i, z_i), for i = 1, ..., n.
        title (str): The title of the plot.
    """
    given_ax = ax is not None
    if not given_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if isinstance(data, np.ndarray):
        data = [data]

    func = ax.plot if line else ax.scatter

    for d in data:
        func(d[:, 0], d[:, 1], d[:, 2])

    ax.set_title(title)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_zlabel("$Z$")

    if given_ax:
        return ax

    plt.show()


if __name__ == "__main__":
    data = [retrieve_dataset(i) for i in range(1, 10)]
    plot_3d_points(data, title="3D Curve of the heart")
