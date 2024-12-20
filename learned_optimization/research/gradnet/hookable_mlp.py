import jax
import haiku as hk
import oryx

from typing import Sequence, Callable


class HookableMLP(hk.Module):
    """
    An MLP that can be hooked to get activations and tangents. Activations are
    sown with the tag "activations" and tangents can be computed by taking
    gradients with respect to auxiliary arguments, termed bs.
    """

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

    def __call__(self, x, *bs):
        x = oryx.core.sow(x, name="a_0", tag="activations")

        for i, hidden_dim in enumerate(self.hidden_dims):
            x = hk.Linear(hidden_dim)(x) + bs[i]
            x = oryx.core.sow(self.activation(x), tag="activations", name=f"a_{i + 1}")

        x = hk.Linear(self.output_dim)(x) + bs[-1]

        return x
