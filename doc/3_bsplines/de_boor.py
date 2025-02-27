import numpy as np


def foo(
    x: float,
    mu: int,
    t: np.ndarray,
    coeffs: np.ndarray,
    d: int,
) -> np.ndarray:
    mu = np.searchsorted(t, x, side="right") - 1
    for r in range(1, d + 1):
        new_coeffs = np.zeros_like(coeffs)
        for i in range(mu - d + r, mu + 1):
            left = 0.0
            if i - 1 >= 0 and i + d - r + 1 != i:
                factor = (t[i + d - r + 1] - x) / (t[i + d - r + 1] - t[i])
                left = factor * coeffs[i - 1]

            right = 0.0
            if i + d - r + 1 != i:
                factor = (x - t[i]) / (t[i + d - r + 1] - t[i])
                right = factor * coeffs[i]

            new_coeffs[i] = left + right
        coeffs = new_coeffs
    return coeffs


if __name__ == "__main__":
    n = 5
    d = 3
    t_org = np.array([0.0, 1.1, 2.4, 3, 4, 5.2, 6.0, 7.2, 8.0])
    t = np.pad(t_org, (0, d), mode="edge")
    print(len(t), n + d + 1)
    coeffs = np.zeros_like(t)

    x = np.linspace(t.min(), t.max(), 200, endpoint=False)

    values = np.zeros((n, len(x)))
    for i, x_i in enumerate(x):
        mu = np.searchsorted(t, x_i, side="right") - 1
        for idx in range(n):
            coeffs = np.zeros_like(t)
            coeffs[idx] = 1.0
            res = foo(x_i, mu, t, coeffs, d)
            values[idx, i] = res[mu]

    import matplotlib.pyplot as plt

    plt.plot(x, values.T, "k")
    plt.ylim(0, 1)
    plt.xlim(t.min(), t.max())
    plt.yticks([0, 0.5, 1])
    plt.xticks(t_org)

    plt.savefig("de_boor.pdf")
    plt.show()
