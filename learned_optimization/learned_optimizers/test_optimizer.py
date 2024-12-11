import wandb
import oryx
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
import tqdm

from learned_optimization.tasks.fixed import image_mlp

from learned_optimization.learned_optimizers import gradnet_lopt
from learned_optimization.learned_optimizers import mlp_lopt


task = image_mlp.HookableImageMLP_FashionMnist8_Relu32()
lopt = gradnet_lopt.GradNetLOpt()
# lopt = mlp_lopt.MLPLOpt()

theta = pickle.load(open("lopt_gradnet_ds_damping2_200pes_50000.pkl", "rb"))
optimizer = lopt.opt_fn(theta)

train_dataset = task.datasets.train
test_dataset = task.datasets.test


@jax.jit
def optstep(key, opt_state, batch):
    params = optimizer.get_params(opt_state)

    loss, g = jax.value_and_grad(task.loss)(params, key, batch)

    # Get activations
    ais = oryx.core.reap(task.loss, tag="activations")(params, key, batch)
    activations = [ais[f"a_{i}"] for i in range(len(ais))]

    # Get tangents
    bs = [jnp.zeros([ais["a_0"].shape[0], size]) for size in task.sizes]
    tangents = jax.grad(task.loss_with_bs, argnums=list(range(3, len(bs) + 3)))(
        params, subkey, batch, *bs
    )

    opt_state = optimizer.update(opt_state, g, activations, tangents, loss)
    return loss, opt_state

for seed in range(30):
    wandb.init(
        settings=wandb.Settings(start_method="thread"),
        project="gradient-networks",
        name=f"lopt-test-mlp-50000-optimizer-seed-{seed}",
        config={"AAA": "g-ds-damping-200pes"}
    )
    key = jax.random.PRNGKey(0)
    params = task.init(key)

    opt_state = optimizer.init(params)


    for batch_idx, batch in enumerate(tqdm.tqdm(train_dataset)):
        if batch_idx == 2000:
            break

        key, subkey = jax.random.split(key)
        loss, opt_state = optstep(subkey, opt_state, batch)

        wandb.log({"loss": loss})
    wandb.finish()
