import wandb
import oryx
import jax
import jax.numpy as jnp
import haiku as hk
import pickle

from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.outer_trainers import full_es
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.outer_trainers import lopt_truncated_step

from learned_optimization.tasks import quadratics
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks import base as tasks_base

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import gradnet_lopt
from learned_optimization.optimizers import base as opt_base

from learned_optimization import optimizers
from learned_optimization import training
from learned_optimization import eval_training


task = image_mlp.HookedImageMLP()
lopt = gradnet_lopt.GradNetLOpt()

theta = pickle.load(open("lopt_gradnet.pkl", "rb"))
optimizer = lopt.opt_fn(theta)

train_dataset = task.datasets.train
test_dataset = task.datasets.test

key = jax.random.PRNGKey(0)
params = task.init_params(key)

opt_state = optimizer.init(params)

if True:
    wandb.init(
        settings=wandb.Settings(start_method="thread"),
        project="gradient-networks",
        name="lopt-test-gradnet-optimizer",
    )
for epoch in range(10):
    for batch in train_dataset:
        key, subkey = jax.random.split(key)
        params = optimizer.get_params(opt_state)

        loss, g = jax.value_and_grad(task.loss)(params, subkey, batch)

        # Get activations
        ais = oryx.core.reap(task.loss, tag="activations")(params, subkey, batch)
        activations = [ais[f"a_{i}"] for i in range(len(ais))]

        # Get tangents
        bs = [jnp.zeros([ais["a_0"].shape[0], size]) for size in task.sizes]
        tangents = jax.grad(task.loss_with_bs, argnums=list(range(3, len(bs) + 3)))(
            params, subkey, batch, *bs
        )

        opt_state = optimizer.update(opt_state, g, activations, tangents, loss)

        wandb.log({"loss": loss})
