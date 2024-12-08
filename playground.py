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
        x = oryx.core.sow(x, name="a_0", tag="activations") + args[0]

        for i, hidden_dim in enumerate(self.hidden_dims):
            x = (
                oryx.core.sow(
                    hk.Linear(hidden_dim)(x), tag="activations", name=f"a_{i + 1}"
                )
                + args[i + 1]
            )
            x = self.activation(x)

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
        lambda x, b1, b2, b3, b4: HookableMLP(hidden_dims=[10, 10], output_dim=1)(
            x, b1, b2, b3, b4
        )
    )
)
params = model.init(
    jax.random.PRNGKey(0),
    jnp.ones([10]),
    jnp.zeros([10]),
    jnp.zeros([10]),
    jnp.zeros([10]),
    jnp.zeros([1]),
)


x = jnp.ones([10])
b1 = jnp.zeros([10])
b2 = jnp.zeros([10])
b3 = jnp.zeros([10])
b4 = jnp.zeros([1])
print(oryx.core.trace(model.apply)(params, x, b1, b2, b3, b4))
