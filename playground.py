import jax
import jax.numpy as jnp
import haiku as hk
import oryx

from typing import Sequence, Callable


class HookableMLP(hk.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: Callable = jax.nn.relu,
    ):
        super().__init__(name="model")

        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation

    def __call__(self, x, *args):
        x = oryx.core.sow(x, name="a_0", tag="activations")

        for i, hidden_dim in enumerate(self.hidden_dims):
            x = hk.Linear(hidden_dim)(x) + args[i]
            x = oryx.core.sow(self.activation(x), tag="activations", name=f"a_{i + 1}")

        x = (
            oryx.core.sow(
                hk.Linear(self.output_dim)(x),
                tag="activations",
                name=f"a_{len(self.hidden_dims) + 1}",
            )
            + args[-1]
        )

        return x


model = hk.without_apply_rng(
    hk.transform(
        lambda x, b1, b2, b3: HookableMLP(hidden_dims=[10, 10], output_dim=1)(
            x, b1, b2, b3
        )
    )
)
params = model.init(
    jax.random.PRNGKey(0),
    jnp.ones([10]),
    jnp.zeros([10]),
    jnp.zeros([10]),
    jnp.zeros([1]),
)


x = jnp.ones([10])
b1 = jnp.zeros([10])
b2 = jnp.zeros([10])
b3 = jnp.zeros([1])


def loss_fn(params, x, b1, b2, b3, y):
    pred = model.apply(params, x, b1, b2, b3)
    return jnp.mean((pred - y) ** 2)


grad_fn = jax.grad(loss_fn, argnums=[0, 1, 2, 3, 4])

activations = oryx.core.reap(model.apply, tag="activations")(params, x, b1, b2, b3)

grads, x_g, b1_g, b2_g, b3_g = grad_fn(params, x, b1, b2, b3, 0.0)

print("Einsum computation")
print(jnp.einsum("i,j->ij", activations["a_0"], b1_g))
print("Direct computation")
print(grads["model/linear"]["w"])
