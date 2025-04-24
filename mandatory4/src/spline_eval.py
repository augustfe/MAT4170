import numpy as np


def eval_bspline_at_point(i: int, d: int, t: np.ndarray, x: float):
    """Evaluate the B-spline basis function of degree d at a given knot vector t.

    This function assumes that the knots are distinct.

    Args:
        i (int): The index of the B-spline basis function.
        d (int): The degree of the B-spline basis function.
        t (np.ndarray): The knot vector.
        x (float): The point at which to evaluate the B-spline basis function.

    Returns:
        float: The value of the B-spline basis function at x.
    """
    if not (t[i] <= x <= t[i + d + 1]):
        return 0.0

    if d == 0:
        return 1.0 if t[i] <= x <= t[i + 1] else 0.0

    left = (x - t[i]) / (t[i + d] - t[i]) if t[i] != t[i + d] else 0.0
    right = (
        (t[i + d + 1] - x) / (t[i + d + 1] - t[i + 1])
        if t[i + 1] != t[i + d + 1]
        else 0.0
    )

    B_left = eval_bspline_at_point(i, d - 1, t, x)
    B_right = eval_bspline_at_point(i + 1, d - 1, t, x)
    return left * B_left + right * B_right


def eval_spline(coeffs: np.ndarray, t: np.ndarray, x: float) -> np.ndarray:
    """Evaluate the B-spline at a given point x.

    Args:
        coeffs (np.ndarray): The coefficients of the B-spline basis functions.
        t (np.ndarray): The knot vector.
        x (float): The point at which to evaluate the B-spline.

    Returns:
        np.ndarray: The value of the B-spline at x.
    """
    d = coeffs.shape[0] - 1
    n = len(t) - d - 1
    result = 0.0

    for i in range(n):
        result += coeffs[i] * eval_bspline_at_point(i, d, t, x)

    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = np.array([0, 1.1, 2.4, 3, 4, 5.2, 6.0, 7.2, 8])
    d = 3
    n = len(t) - d - 1

    x_vals = np.linspace(0, 8, 100)

    for i in range(n):
        y_vals = [eval_bspline_at_point(i, d, t, x) for x in x_vals]
        plt.plot(x_vals, y_vals, label=f"$B_{i}$")

    plt.title("B-spline basis functions")
    plt.xlabel("x")
    plt.ylabel("B-spline value")
    plt.legend()
    plt.grid()
    plt.show()
