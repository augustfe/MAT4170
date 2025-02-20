from functools import partial

import jax.numpy as jnp
from find_mu import find_mu
from jax import Array, jit, lax, vmap


def one_iteration(knots: Array, curr_B: Array, x: float, r: int, mu: int) -> Array:
    """Perform one iteration of the recursion formula.

    Args:
        knots: The array of knots.
        curr_B: The current B-spline basis functions.
        x: The value to evaluate the B-spline basis functions at.
        r: The current iteration of the recursion formula.
        mu: The index of the knot interval that contains x.

    Returns:
        The updated B-spline basis functions.
    """
    idxs = jnp.arange(mu - r, mu)
    print(r, idxs)
    print(curr_B)
    left = (x - knots[idxs]) / (knots[idxs + r] - knots[idxs]) * curr_B[:r]
    idxs = idxs + 1
    right = (knots[idxs + r] - x) / (knots[idxs + r] - knots[idxs]) * curr_B[1 : r + 1]
    left = jnp.concatenate([jnp.array([0.0]), left])
    right = jnp.concatenate([right, jnp.array([0.0])])
    return left + right


def bspline_recursion(knots: Array, x: float, d: int) -> float:
    """Evaluate the B-spline basis functions recursively.

    Evaluate the B-spline basis functions recursively using the Cox-de Boor
    recursion formula.

    Args:
        knots: The array of knots.
        x: The value to evaluate the B-spline basis functions at.
        d: The degree of the B-spline basis functions.

    Returns:
        The value of the B-spline basis functions at x.
    """
    mu = find_mu(knots, x)
    initial_B = jnp.zeros(d + 2).at[0].set(1.0)
    # initial_B = jnp.ones(1)
    print(mu)

    one_iter = lambda r, curr_B: one_iteration(knots, curr_B, x, r, mu)(r, curr_B)

    # final_B = lax.fori_loop(1, d + 1, one_iter, initial_B)
    curr_B = initial_B
    for i in range(1, d + 1):
        curr_B = one_iteration(knots, curr_B, x, i, mu)
    return curr_B[-1]


if __name__ == "__main__":
    t = jnp.array([0.0, 1.1, 2.4, 3.4, 5.2, 6.0, 7.2, 8])
    d = 3
    x = jnp.linspace(0, 8, 100)
    # v_recursion = vmap(lambda x: bspline_recursion(t, x, d))
    B = [bspline_recursion(t, x_i, d) for x_i in x]

    # B = v_recursion(x)

    import matplotlib.pyplot as plt

    for i in range(d + 1):
        plt.plot(x, B[:, i], label=f"B_{i}")
    plt.legend()
    plt.show()
