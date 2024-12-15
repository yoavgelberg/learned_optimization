import pickle
import wandb
import flax
import flax.linen as nn
import numpy as np
import jax.numpy as jnp
import jax
from matplotlib import pylab as plt

from typing import Any, Callable, Sequence

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
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.optimizers import base as opt_base

from learned_optimization import optimizers
from learned_optimization import training
from learned_optimization import eval_training

import tqdm

MetaParams = Any

if __name__ == "__main__":
    theta_opt = opt_base.Adam(1e-3)

    # lopt = mlp_lopt.MLPLOpt(hidden_layers=3)
    lopt = gradnet_lopt.GradNetLOpt()

    max_length = 2000
    trunc_sched = truncation_schedule.LogUniformLengthSchedule(
        min_length=2000, max_length=max_length)


    def grad_est_fn(task_family):
      truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
          task_family,
          lopt,
          trunc_sched,
          num_tasks=16,
          random_initial_iteration_offset=max_length)
      return truncated_pes.TruncatedPES(
          truncated_step=truncated_step, trunc_length=50)


    mlp_task_family = tasks_base.single_task_to_family(
        image_mlp.HookableImageMLP_FashionMnist8_Relu32())

    gradient_estimators = [
        grad_est_fn(mlp_task_family),
    ]

    outer_trainer = gradient_learner.SingleMachineGradientLearner(
        lopt, gradient_estimators, theta_opt)

    key = jax.random.PRNGKey(0)
    outer_trainer_state = outer_trainer.init(key)
    jax.tree_util.tree_map(lambda x: jnp.asarray(x).shape, outer_trainer_state)

    losses = []
    import tqdm

    outer_train_steps = 50000

    if True:
       wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project="gradient-networks",
            name="lopt-gradnet++-abg",
        )

    for i in tqdm.trange(outer_train_steps):
      outer_trainer_state, loss, metrics = outer_trainer.update(
          outer_trainer_state, key, with_metrics=False)
      losses.append(loss)
      theta = outer_trainer.get_meta_params(outer_trainer_state)
      wandb.log({"meta training loss": loss, 
          "unscaled-alpha": theta["alpha"], 
          "unscaled-beta": theta["beta"], 
          "unscaled-gamma": theta["gamma"], 
          "alpha": mlp_lopt._scaled_lr.inverse(theta["alpha"]), 
          "beta": mlp_lopt._scaled_lr.inverse(theta["beta"]), 
          "gamma": mlp_lopt._scaled_lr.inverse(theta["gamma"])})

    pickle.dump(outer_trainer.get_meta_params(outer_trainer_state), open("lopt_gradnet++_abg.pkl", "wb"))
