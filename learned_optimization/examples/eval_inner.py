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
from learned_optimization.research.univ_nfn.learned_opt import learned_opts
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization.outer_trainers import lopt_truncated_step
from learned_optimization.outer_trainers import truncated_pes
from learned_optimization.outer_trainers import truncation_schedule
from learned_optimization.tasks import base as tasks_base
from learned_optimization.tasks.fixed import image_mlp
from learned_optimization.tasks.fixed import conv
from learned_optimization.tasks.fixed import rnn_lm
import numpy as np
import tqdm

FLAGS = flags.FLAGS


def evaluate(ckpt_path: str, out_path, task: Optional[tasks_base.Task] = None):
  """Train a learned optimizer!"""

  if not task:
    if FLAGS.task == "cnn":
      task = conv.Conv_Cifar10_8_16x32()
    elif FLAGS.task == "mlp":
      task = image_mlp.ImageMLP_FashionMnist8_Relu32()
    elif FLAGS.task == "rnn":
      task = rnn_lm.RNNLM_lm1bbytes_Patch16_IRNN64_Embed32()
    else:
      raise ValueError(f"Unknown task {FLAGS.task}.")

  filesystem.make_dirs(out_path)

  print(FLAGS.lopt)
  if FLAGS.lopt == "mlp":
    lopt = learned_opts.ResidualOptMLP(task, learnable_hp=FLAGS.learnable_hp)
  elif FLAGS.lopt == "nfn":
    lopt = learned_opts.ResidualOptNFN(task, ptwise_init=FLAGS.pointwise, nfn_type="hybrid", learnable_hp=FLAGS.learnable_hp)
  elif FLAGS.lopt == "nfn_het":
    lopt = learned_opts.ResidualOptNFN(task, ptwise_init=FLAGS.pointwise, nfn_type="het", learnable_hp=FLAGS.learnable_hp)
  elif FLAGS.lopt == "nfn_baseline":
    lopt = learned_opts.ResidualOptNFNCNN(task, ptwise_init=FLAGS.pointwise, learnable_hp=FLAGS.learnable_hp)
  elif FLAGS.lopt == "sgdm":
    # Optax uses trace instead of EMA, so for momentum=0.9 the LR should be divided by 10.
    lopt = lopt_base.LearnableSGDM()

  max_length = 4_000
  with open(ckpt_path, 'rb') as f:
    theta = pickle.load(f)
  print("Theta param count", sum([x.size for x in jax.tree_util.tree_leaves(theta)]))
  all_results = []
  for _ in range(FLAGS.num_evals):
    key = jax.random.PRNGKey(int(np.random.randint(0, int(2**30))))
    opt = lopt.opt_fn(theta)
    results = eval_training.single_task_training_curves(task, opt, max_length, key)
    all_results.append(results)
  with open(os.path.join(out_path, "results.pkl"), 'wb') as f:
    pickle.dump(all_results, f)


def main(unused_argv: Sequence[str]) -> None:
  evaluate(FLAGS.ckpt_path, FLAGS.out_path)


if __name__ == "__main__":
  flags.DEFINE_string("ckpt_path", None, "")
  flags.DEFINE_integer("num_evals", None, "")
  flags.DEFINE_string("out_path", None, "")
  flags.DEFINE_string("lopt", None, "")
  flags.DEFINE_string("task", None, "")
  flags.DEFINE_boolean("pointwise", False, "")
  flags.DEFINE_boolean("learnable_hp", False, "")
  flags.mark_flag_as_required("ckpt_path")
  flags.mark_flag_as_required("num_evals")
  flags.mark_flag_as_required("out_path")
  flags.mark_flag_as_required("lopt")
  flags.mark_flag_as_required("task")
  app.run(main)
