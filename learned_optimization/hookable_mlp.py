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
