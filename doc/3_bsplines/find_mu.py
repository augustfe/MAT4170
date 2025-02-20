import jax.numpy as jnp
import pytest
from jax import Array


def find_mu(knots: Array, x: float) -> Array:
    """Find the index of the knot interval that contains x.

    Simply performs a binary search to find the index of the knot interval that
    contains x.

    Args:
        knots: The array of knots.
        x: The value to find the interval for.

    Returns:
        The index of the knot interval that contains x.
    """
    mu = jnp.searchsorted(knots, x, side="right") - 1
    return mu


@pytest.mark.parametrize(
    "knots, xs",
    [
        (jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]), jnp.array([0.5, 1.5, 2.5, 3.5])),
        (jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]), jnp.array([0.0, 1.0, 2.0, 3.0])),
    ],
)
def test_find_mu(knots: Array, xs: Array) -> None:
    for x in xs:
        mu = find_mu(knots, x)
        assert 0 <= mu < len(knots) - 1
        assert knots[mu] <= x < knots[mu + 1]


if __name__ == "__main__":
    knots = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x = 1.5
    mu = find_mu(knots, x)
    print(mu)
    # Output: 1
