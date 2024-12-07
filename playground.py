import jax
import jax.numpy as jnp
import haiku as hk
import oryx


def record_vjp(f, name):
    @jax.custom_vjp
    def custom_f(*args):
        return f(*args)

    def custom_fwd(*args, **kwargs):
        return f(*args, **kwargs), args

    def custom_bwd(res, dy):
        _, vjp = jax.vjp(f, *res)
        return oryx.core.sow(vjp(dy), tag="vjp", name=name)

    custom_f.defvjp(custom_fwd, custom_bwd)
    return custom_f


class Model(hk.Module):
    def __init__(self):
        super().__init__(name="model")

    def __call__(self, x):
        x = oryx.core.sow(
            record_vjp(hk.Linear(10))(x, name="g_0"), tag="activations", name="a_0"
        )
        x = jax.nn.relu(x)
        x = oryx.core.sow(
            record_vjp(hk.Linear(10))(x, name="g_1"), tag="activations", name="a_1"
        )
        x = jax.nn.relu(x)
        x = oryx.core.sow(
            record_vjp(hk.Linear(1))(x, name="g_2"), tag="activations", name="a_2"
        )
        return x


model = hk.without_apply_rng(hk.transform(lambda x, b1, b2: Model()(x, b1, b2)))
params = model.init(
    jax.random.PRNGKey(0), jnp.ones([10]), jnp.zeros([10]), jnp.zeros([10])
)


x = jnp.ones([10])
b1 = jnp.zeros([10])
b2 = jnp.zeros([10])
print(model.apply(params, x, b1, b2))
