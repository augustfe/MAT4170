import numpy as np
from data import chord_length_parametrization, retrieve_dataset
from spline_eval import eval_bspline_at_point, eval_spline


def setup_A(u_arr: np.ndarray, knot_percentage: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < knot_percentage <= 1:
        raise ValueError("knot_percentage must be in (0, 1])")

    # data = retrieve_dataset(file_nr)
    m = u_arr.shape[0]
    n = int(m * knot_percentage)
    d = 3
    # if knot_percentage == 1:
    #     n = m - d - 1  # First and last knot are the same

    # u_arr = chord_length_parametrization(data)
    u_min = u_arr[0]
    u_max = u_arr[-1]
    # knot_vector = np.linspace(u_min, u_max, n + d + 1)
    knot_start = np.repeat(u_min, d)
    knot_mid = np.linspace(u_min, u_max, n - d + 1)
    knot_end = np.repeat(u_max, d)
    knot_vector = np.concatenate((knot_start, knot_mid, knot_end))

    """
    (n + d + 1) - 2d
    = n - d + 1
    """

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

    ATA_inv = np.linalg.pinv(A.T @ A)
    coeffs = ATA_inv @ A.T @ data
    return coeffs, knot_vector


if __name__ == "__main__":
    layers = []
    d = 3
    # file_nr = 2
    knot_percentage = 0.2

    for file_nr in range(1, 10):
        coeffs, knot_vector = compute_coeffs(file_nr, knot_percentage)
        n = len(knot_vector) - d - 1

        u_vals = np.linspace(knot_vector[0], knot_vector[-1], 200)
        print(knot_vector)
        b_spline_vals = np.array(
            [
                eval_bspline_at_point(i, 3, knot_vector, u)
                for u in u_vals
                for i in range(n)
            ]
        ).reshape(len(u_vals), -1)

        results = b_spline_vals @ coeffs  # [10:-10]
        layers.append(results)

    from plot import plot_3d_points

    plot_3d_points(layers, title="3D Curve of the heart", line=True)

    # print(b_spline_vals[0:10])
    # print(b_spline_vals[-10:])
    # print(coeffs[0])
    # print(b_spline_vals[0])

    # print(results[:, 2])
    # import matplotlib.pyplot as plt

    # plt.plot(results[:, 0], results[:, 1], label="B-spline curve")
    # plt.scatter(coeffs[:, 0], coeffs[:, 1], color="red", label="Control Points", s=10)
    # data = retrieve_dataset(file_nr)
    # plt.scatter(data[:, 0], data[:, 1], color="green", label="Data Points", s=10)
    # plt.legend()
    # plt.title("B-spline curve")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.show()
    # results = np.array([eval_spline(coeffs, u_arr, u) for u in u_vals])
    # plt.plot(results[:, 0], results[:, 1])
    # plt.show()
