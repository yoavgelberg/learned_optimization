# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple learned optimizer training example using gradient estimator APIs."""
from typing import Optional, Sequence

import os
from absl import app
from absl import flags
import pickle
import jax
from learned_optimization import filesystem
from learned_optimization import summary
from learned_optimization import eval_training
from learned_optimization.learned_optimizers import base as lopt_base  # pylint: disable=unused-import
from learned_optimization.learned_optimizers import mlp_lopt
from learned_optimization.learned_optimizers import adafac_nominal
from learned_optimization.research.univ_nfn.learned_opt import learned_opts
from learned_optimization.research.gradnet.tasks import hookable_image_mlp
from learned_optimization.research.gradnet.learned_opts import ResidualOptNFNWithGradnet
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks.fixed import conv
from learned_optimization.tasks.fixed import rnn_lm
from learned_optimization.tasks.fixed import transformer_lm
import numpy as np
import tqdm

FLAGS = flags.FLAGS


def train(train_log_dir: str,
          outer_iterations: int = 100_000,
          task: Optional[tasks_base.Task] = None):
  """Train a learned optimizer!"""

  if not task:
    if FLAGS.task == "cnn":
      task = conv.Conv_Cifar10_8_16x32()
    elif FLAGS.task == "mlp":
      task = image_mlp.ImageMLP_FashionMnist8_Relu32()
    elif FLAGS.task == "hookable_mlp":
      task = hookable_image_mlp.HookableImageMLP_FashionMnist8_Relu32()
    elif FLAGS.task == "rnn":
      # task = rnn_lm.RNNLM_lm1bbytes_Patch32_IRNN128_Embed64()
      task = rnn_lm.RNNLM_lm1bbytes_Patch16_IRNN64_Embed32()
      # task = rnn_lm.RNNLM_lm1bbytes_Patch32_VanillaRNN128_Embed64()
    elif FLAGS.task == "transformer":
      task = transformer_lm.TransformerLM_LM1B_MultiRuntime_2()
    else:
      raise ValueError(f"Unknown task {FLAGS.task}.")

  #### Hparams
  # learning rate used to train the learned optimizer
  outer_learning_rate = 1e-4
  # max length of inner training unrolls
  max_length = 2_000
  # number of tasks to train in parallel
  num_tasks = 16
  # length of truncations for PES
  trunc_length = 100

  key = jax.random.PRNGKey(int(np.random.randint(0, int(2**30))))

  filesystem.make_dirs(train_log_dir)
  summary_writer = summary.MultiWriter(
      summary.JaxboardWriter(train_log_dir), summary.PrintWriter())

  theta_opt = opt_base.Adam(outer_learning_rate)

  step_mult = FLAGS.step_mult
  out_mult = FLAGS.out_mult
  print(FLAGS.lopt)
  if FLAGS.lopt == "mlp":
    lopt = learned_opts.ResidualOptMLP(
      task, step_mult=step_mult, out_mult=out_mult, learnable_hp=FLAGS.learnable_hp
    )
  elif FLAGS.lopt == "nfn":
    lopt = learned_opts.ResidualOptNFN(
      task, step_mult=step_mult, out_mult=out_mult, ptwise_init=FLAGS.pointwise,
      nfn_type="hybrid", learnable_hp=FLAGS.learnable_hp
    )
  elif FLAGS.lopt == "gradnet_nfn":
    assert FLAGS.task == "hookable_mlp", "Only hookable_mlp is supported for gradnet_nfn"
    lopt = learned_opts.ResidualOptNFNWithGradnet(
      task, step_mult=step_mult, out_mult=out_mult, ptwise_init=FLAGS.pointwise,
      nfn_type="hybrid", learnable_hp=FLAGS.learnable_hp
    )
  elif FLAGS.lopt == "nfn_het":
    lopt = learned_opts.ResidualOptNFN(
      task, step_mult=step_mult, out_mult=out_mult, ptwise_init=FLAGS.pointwise, nfn_type="het",
      learnable_hp=FLAGS.learnable_hp
    )
  elif FLAGS.lopt == "nfn_baseline":
    lopt = learned_opts.ResidualOptNFNCNN(
      task, step_mult=step_mult, out_mult=out_mult, ptwise_init=FLAGS.pointwise,
      learnable_hp=FLAGS.learnable_hp
    )
  elif FLAGS.lopt == "sgdm":
    # Optax uses trace instead of EMA, so for momentum=0.9 the LR should be divided by 10.
    lopt = lopt_base.LearnableSGDM(initial_lr=step_mult / 10)
  elif FLAGS.lopt == "adam":
    lopt = lopt_base.LearnableAdam(initial_lr=step_mult)
  elif FLAGS.lopt == "mlp_adam":
    lopt = adafac_nominal.MLPNomLOpt(
      task,
      nominal_grad_estimator="Adam",
      # Matching the other LOpt convention of step_mult * (nominal + out_mult * f( ))
      # Adafac nominal uses nominal_stepsize * nominal + step_mult * f( )
      nominal_stepsize=step_mult,
      step_mult=step_mult * out_mult,
      learnable_hp=FLAGS.learnable_hp,
      method="mlp")
  elif FLAGS.lopt == "nfn_adam":
    assert FLAGS.pointwise, "Only pointwise is supported."
    lopt = adafac_nominal.MLPNomLOpt(
      task,
      nominal_grad_estimator="Adam",
      nominal_stepsize=step_mult,
      step_mult=step_mult * out_mult,
      learnable_hp=FLAGS.learnable_hp,
      method="nfn")
  elif FLAGS.lopt == "nfnhybrid_adam":
    assert FLAGS.pointwise, "Only pointwise is supported."
    lopt = adafac_nominal.MLPNomLOpt(
      task,
      nominal_grad_estimator="Adam",
      nominal_stepsize=step_mult,
      step_mult=step_mult * out_mult,
      learnable_hp=FLAGS.learnable_hp,
      method="nfn_hybrid",)

  # trunc_sched = truncation_schedule.LogUniformLengthSchedule(
  #     min_length=100, max_length=max_length)
  trunc_sched = truncation_schedule.ConstantTruncationSchedule(
    total_length=max_length)

  task_family = tasks_base.single_task_to_family(task)

  truncated_step = lopt_truncated_step.VectorizedLOptTruncatedStep(
      task_family=task_family,
      learned_opt=lopt,
      trunc_sched=trunc_sched,
      num_tasks=num_tasks,
      random_initial_iteration_offset=max_length)

  grad_est = truncated_pes.TruncatedPES(
      truncated_step=truncated_step, trunc_length=trunc_length)

  gradient_estimators = [grad_est]

  outer_trainer = gradient_learner.SingleMachineGradientLearner(
      lopt, gradient_estimators, theta_opt)

  outer_trainer_state = outer_trainer.init(key)

  eval_key = jax.random.PRNGKey(int(np.random.randint(0, int(2**30))))
  theta0 = outer_trainer.get_meta_params(outer_trainer_state)
  print("Theta param count", sum([x.size for x in jax.tree_util.tree_leaves(theta0)]))
  init_opt = lopt.opt_fn(theta0)
  initial_results = eval_training.single_task_training_curves(
      task, init_opt, max_length, eval_key)
  for i, val in enumerate(initial_results['train/loss']):
    summary_writer.scalar("single_task/init", val, step=i)

  losses = []
  for i in tqdm.trange(outer_iterations):
    if i % 10_000 == 0:
      theta_i = outer_trainer.get_meta_params(outer_trainer_state)
      with open(os.path.join(train_log_dir, f"theta{i}.pkl"), 'wb') as f:
        pickle.dump(theta_i, f)
    with_m = True if i % 10 == 0 else False
    key1, key = jax.random.split(key)
    outer_trainer_state, loss, metrics = outer_trainer.update(
        outer_trainer_state, key1, with_metrics=with_m)
    losses.append(loss)

    # log out summaries to tensorboard
    if with_m:
      summary_writer.scalar("average_meta_loss", np.mean(losses), step=i)
      losses = []
      for k, v in metrics.items():
        agg_type, metric_name = k.split("||")
        if agg_type == "collect":
          summary_writer.histogram(metric_name, v, step=i)
        else:
          summary_writer.scalar(metric_name, v, step=i)
      summary_writer.flush()

  thetaT = outer_trainer.get_meta_params(outer_trainer_state)
  with open(os.path.join(train_log_dir, "thetaT.pkl"), 'wb') as f:
    pickle.dump(thetaT, f)
  final_opt = lopt.opt_fn(thetaT)
  final_results = eval_training.single_task_training_curves(
      task, final_opt, max_length, eval_key)
  for i, val in enumerate(final_results['train/loss']):
    summary_writer.scalar("single_task/final", val, step=i)
  summary_writer.flush()

def main(unused_argv: Sequence[str]) -> None:
  train(FLAGS.train_log_dir)


if __name__ == "__main__":
  flags.DEFINE_string("lopt", None, "")
  flags.DEFINE_string("task", None, "")
  flags.DEFINE_string("train_log_dir", None, "")
  flags.DEFINE_float("step_mult", 0.1, "")
  flags.DEFINE_float("out_mult", 1e-3, "")
  flags.DEFINE_boolean("pointwise", False, "")
  flags.DEFINE_boolean("learnable_hp", False, "")
  flags.mark_flag_as_required("lopt")
  flags.mark_flag_as_required("task")
  flags.mark_flag_as_required("train_log_dir")
  app.run(main)
