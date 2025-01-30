import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array, jit, vmap


@jit
def de_casteljau(cs: Array, x: float) -> Array:
    left = cs[:-1]
    right = cs[1:]
    return (1 - x) * left + x * right


def eval_de_casteljau(cs: Array, x: float) -> float:
    for _ in range(len(cs) - 1):
        cs = de_casteljau(cs, x)

    return cs


def plot_control_points(cs: Array, x_coords: Array) -> None:
    plt.scatter(x_coords, cs, c="r", clip_on=False, zorder=2)
    plt.plot(x_coords, cs, "b", zorder=1)


if __name__ == "__main__":
    from pathlib import Path

    cs = jnp.array([0.2, 0.4, -0.1, 0.5])

    plt.figure(figsize=(6, 4))
    plt.tick_params(direction="in", top=True, right=True)
    plt.xlim(0, 1)
    plt.ylim(cs.min(), cs.max())

    x = jnp.linspace(0, 1, 100)
    eval_vmap = vmap(eval_de_casteljau, in_axes=(None, 0))
    y = eval_vmap(cs, x)

    plt.plot(x, y, "k", zorder=0)

    x_point = 0.6
    x_coords = jnp.linspace(0, 1, len(cs))

    while len(cs) > 0:
        plot_control_points(cs, x_coords)
        cs = de_casteljau(cs, x_point)
        x_coords = de_casteljau(x_coords, x_point)

    plt.savefig(Path(__file__).parent / "de_casteljau.pdf", bbox_inches="tight")
    plt.show()
