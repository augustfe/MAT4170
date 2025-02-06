import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array, jit, vmap


@jit
def de_casteljau(cs: Array, lmbd: float) -> Array:
    left = cs[:-1]
    right = cs[1:]
    return (1 - lmbd) * left + lmbd * right


@jit
def eval_de_casteljau(cs: Array, lmbd: float) -> Array:
    for _ in range(cs.shape[0] - 1):
        cs = de_casteljau(cs, lmbd)

    return cs.flatten()


def plot_control_points(cs: Array) -> None:
    plt.scatter(cs[:, 0], cs[:, 1], c="r", clip_on=False, zorder=2)
    plt.plot(cs[:, 0], cs[:, 1], "b", zorder=1)


if __name__ == "__main__":
    from pathlib import Path

    cs = jnp.array([[-1, 1], [-1, 0], [0, 0]])
    ds = jnp.array([[0, 0], [1, 0], [2, 1]])

    t_values = jnp.linspace(0, 1, 100)
    eval_vmap = vmap(eval_de_casteljau, in_axes=(None, 0))
    p_values = eval_vmap(cs, t_values)
    q_values = eval_vmap(ds, t_values)

    plt.figure(figsize=(6, 4))
    plt.tick_params(direction="in", top=True, right=True)
    plot_control_points(cs)
    plot_control_points(ds)

    plt.plot(p_values[:, 0], p_values[:, 1], "k", zorder=0)
    plt.plot(q_values[:, 0], q_values[:, 1], "k", zorder=0)

    plt.savefig(Path(__file__).parent / "bezier_casteljau.pdf", bbox_inches="tight")
    plt.show()
