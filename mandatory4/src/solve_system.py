import matplotlib.pyplot as plt
import numpy as np
from data import chord_length_parametrization, retrieve_dataset
from mpl_toolkits.mplot3d import Axes3D
from plot import plot_3d_points
from spline_eval import eval_bspline_at_point


def setup_A(u_arr: np.ndarray, knot_percentage: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < knot_percentage <= 1:
        raise ValueError("knot_percentage must be in (0, 1])")

    m = u_arr.shape[0]
    n = int(m * knot_percentage)
    d = 3

    u_min = u_arr[0]
    u_max = u_arr[-1]

    knot_start = np.repeat(u_min, d)
    knot_mid = np.linspace(u_min, u_max, n - d + 1)
    knot_end = np.repeat(u_max, d)
    knot_vector = np.concatenate((knot_start, knot_mid, knot_end))

    A = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            A[i, j] = eval_bspline_at_point(j, d, knot_vector, u_arr[i])

    return A, knot_vector


def compute_coeffs(
    file_nr: int, knot_percentage: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the coefficients of the B-spline basis functions.

    Args:
        file_nr (int): The number corresponding to the file "hj{file_nr}.dat".
        knot_percentage (float): The percentage of knots to use.

    Returns:
        np.ndarray: The coefficients of the B-spline basis functions.
    """
    data = retrieve_dataset(file_nr)
    u_arr = chord_length_parametrization(data)
    A, knot_vector = setup_A(u_arr, knot_percentage)

    # ATA_inv = np.linalg.inv(A.T @ A)
    # coeffs = ATA_inv @ A.T @ data

    coeffs = np.linalg.lstsq(A, data)[0]

    return coeffs, knot_vector


def solve_and_plot(knot_percentage: float, ax: Axes3D) -> None:
    """Solve the B-spline system.

    Args:
        knot_percentage (float): The percentage of knots to use.
        ax (Axes3D): The Axes3D object to plot on. Defaults to None.
    """
    layers = []
    degree = 3
    for file_nr in range(1, 10):
        coeffs, knot_vector = compute_coeffs(file_nr, knot_percentage)
        n = len(knot_vector) - degree - 1

        u_vals = np.linspace(knot_vector[0], knot_vector[-1], 200)
        b_spline_vals = np.array(
            [
                eval_bspline_at_point(i, degree, knot_vector, u)
                for u in u_vals
                for i in range(n)
            ]
        ).reshape(len(u_vals), -1)

        results = b_spline_vals @ coeffs
        layers.append(results)

    plot_3d_points(layers, title=f"With ${knot_percentage * 100}$%", line=True, ax=ax)


if __name__ == "__main__":
    import time

    knot_percentages = [0.05, 0.1, 0.2]
    fig = plt.figure()
    axs = fig.subplots(1, len(knot_percentages), subplot_kw={"projection": "3d"})

    start = time.perf_counter()

    for i, knot_percentage in enumerate(knot_percentages):
        ax = axs[i]
        solve_and_plot(knot_percentage, ax=ax)

    end = time.perf_counter()
    print(f"Time taken: {end - start:.2f} seconds")

    fig.suptitle("B-spline with different knot percentages")
    fig.tight_layout()

    plt.show()
