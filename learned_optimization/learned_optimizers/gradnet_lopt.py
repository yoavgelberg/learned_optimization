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

"""Learned optimizer which applies a per parameter MLP.

This is the same model in "Understanding and correcting pathologies in the
training of learned optimizers
(https://arxiv.org/abs/1810.10180).
"""

from typing import Any, Optional

import flax
import gin
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base

PRNGKey = jnp.ndarray


def _second_moment_normalizer(x, axis, eps=1e-5):
  return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def _tanh_embedding(iterations):
  f32 = jnp.float32

  def one_freq(timescale):
    return jnp.tanh(iterations / (f32(timescale)) - 1.0)

  timescales = jnp.asarray(
      [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
      dtype=jnp.float32)
  return jax.vmap(one_freq)(timescales)


@flax.struct.dataclass
class GradNetLOptState:
  params: Any
  rolling_features: common.MomAccumulator
  gmom: common.MomAccumulator
  iteration: jnp.ndarray
  state: Any


@gin.configurable
class GradNetLOpt(lopt_base.LearnedOptimizer):
  """
  Learned optimizer leveraging a per parameter MLP. and a gradient net.
  """

  def __init__(self,
               exp_mult=0.001,
               step_mult=0.001,
               hidden_size=32,
               hidden_layers=2,
               compute_summary=True):

    super().__init__()
    self._step_mult = step_mult
    self._exp_mult = exp_mult
    self._compute_summary = compute_summary

    def ff_mod(inp1, inp2, a):
      o1 = hk.nets.MLP([hidden_size] * hidden_layers + [2])(inp1)
      o2 = hk.nets.MLP([hidden_size] * (hidden_layers + 1) + [1])(inp2)
      return o1 if a else o2

    self._mod = hk.without_apply_rng(hk.transform(ff_mod))

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    theta =  self._mod.init(key, jnp.zeros([0, 21]), jnp.zeros([0, 1]), False)
    theta["alpha"] = jnp.array([0.1])
    theta["beta"] = jnp.array([0.001])
    theta["gamma"] = jnp.array([0.9])
    return theta

  def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: bool = False) -> opt_base.Optimizer:
    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])

    mod = self._mod
    exp_mult = self._exp_mult
    step_mult = self._step_mult
    compute_summary = self._compute_summary

    class _Opt(opt_base.Optimizer):
      """Optimizer instance which has captured the meta-params (theta)."""

      def init(self,
               params: lopt_base.Params,
               model_state: Any = None,
               num_steps: Optional[int] = None,
               key: Optional[PRNGKey] = None) -> GradNetLOptState:
        """Initialize inner opt state."""

        return GradNetLOptState(
            params=params,
            state=model_state,
            rolling_features=common.vec_rolling_mom(decays).init(params),
            gmom=common.vec_rolling_mom(jnp.asarray([0.9])).init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32))

      def update(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: GradNetLOptState,
          grad: Any,
          activations: Any,
          tangents: Any,
          loss: float,
          model_state: Any = None,
          is_valid: bool = False,
          key: Optional[PRNGKey] = None,
      ) -> GradNetLOptState:
        alpha = theta["alpha"][0]
        beta = theta["beta"][0]
        gamma = theta["gamma"][0]

        activations = [jnp.sum(jnp.mean(mod.apply(theta, jnp.zeros([0, 21]), jnp.expand_dims(a, axis=-1), False), axis=-1), axis=0) for a in activations]
        tangents = [jnp.sum(jnp.mean(mod.apply(theta, jnp.zeros([0, 21]), jnp.expand_dims(t, axis=-1), False), axis=-1), axis=0) for t in tangents]

        ff = []
        for a, t in zip(activations, tangents):
          a = jnp.repeat(a[:, jnp.newaxis], t.shape[0], axis=1)
          a = jnp.expand_dims(a, axis=-1)
          t = jnp.repeat(t[jnp.newaxis, :], a.shape[0], axis=0)
          t = jnp.expand_dims(t, axis=-1)
          ff.append(jnp.concatenate([a,  t], axis=-1))

        fish_feat = {"model/linear": {'w': ff[0], 'b': jnp.stack([activations[1], tangents[0]], -1)}, "model/linear_1": {'w': ff[1], 'b': jnp.stack([activations[2], tangents[1]], axis=-1)}}

        next_rolling_features = common.vec_rolling_mom(decays).update(
            opt_state.rolling_features, grad)

        gmom = common.vec_rolling_mom(jnp.asarray([gamma])).update(opt_state.gmom, grad)

        training_step_feature = _tanh_embedding(opt_state.iteration)

        def _update_tensor(p, g, m, f, gm):
          # this doesn't work with scalar parameters, so let's reshape.
          if not p.shape:
            p = jnp.expand_dims(p, 0)
            g = jnp.expand_dims(g, 0)
            m = jnp.expand_dims(m, 0)
            did_reshape = True
          else:
            did_reshape = False

          inps = []

          # feature consisting of raw gradient values
          batch_g = jnp.expand_dims(g, axis=-1)
          inps.append(batch_g)

          # feature consisting of raw parameter values
          batch_p = jnp.expand_dims(p, axis=-1)
          inps.append(batch_p)

          # feature consisting of all momentum values
          inps.append(m)

          # fish feat
          inps.append(f)

          inp_stack = jnp.concatenate(inps, axis=-1)
          axis = list(range(len(p.shape)))

          inp_stack = _second_moment_normalizer(inp_stack, axis=axis)

          # if len(inp_stack.shape) == 2:
            # inp_stack.at[:, -1].multiply(opt_state.iteration + 1)
            # inp_stack.at[:, -2].multiply(opt_state.iteration + 1)
          # else:
            # inp_stack.at[:, :, -1].multiply(opt_state.iteration + 1)
            # inp_stack.at[:, :, -2].multiply(opt_state.iteration + 1)

          # once normalized, add features that are constant across tensor.
          # namly the training step embedding.
          stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                                list(training_step_feature.shape[-1:]))
          stacked = jnp.tile(stacked, list(p.shape) + [1])

          inp = jnp.concatenate([inp_stack, stacked], axis=-1)

          # apply the per parameter MLP.
          output = mod.apply(theta, inp, jnp.zeros([0, 1]), True)

          # split the 2 outputs up into a direction and a magnitude
          direction = output[..., 0]
          magnitude = output[..., 1]

          gm = jnp.reshape(gm, direction.shape)

          # compute the step
          step = alpha * (gm + beta * direction * jnp.exp(magnitude * exp_mult) * step_mult)
          step = step.reshape(p.shape)
          new_p = p - step
          if did_reshape:
            new_p = jnp.squeeze(new_p, 0)

          if compute_summary:
            for fi, f in enumerate(inp):
              summary.summary(f"mlp_lopt/inp{fi}/mean_abs",
                              jnp.mean(jnp.abs(f)))

            avg_step_size = jnp.mean(jnp.abs(step))
            summary.summary("mlp_lopt/avg_step_size", avg_step_size)

            summary.summary(
                "mlp_lopt/avg_step_size_hist",
                avg_step_size,
                aggregation="collect")

            summary.summary("mlp_lopt/direction/mean_abs",
                            jnp.mean(jnp.abs(direction)))
            summary.summary("mlp_lopt/magnitude/mean_abs",
                            jnp.mean(jnp.abs(magnitude)))
            summary.summary("mlp_lopt/magnitude/mean", jnp.mean(magnitude))

            summary.summary("mlp_lopt/grad/mean_abs", jnp.mean(jnp.abs(g)))

          return new_p

        next_params = jax.tree_util.tree_map(_update_tensor, opt_state.params,
                                             grad, next_rolling_features.m, fish_feat, gmom.m)
        next_opt_state = GradNetLOptState(
            params=tree_utils.match_type(next_params, opt_state.params),
            rolling_features=tree_utils.match_type(next_rolling_features,
                                                   opt_state.rolling_features),
            gmom=tree_utils.match_type(gmom,opt_state.gmom),
            iteration=opt_state.iteration + 1,
            state=model_state)
        return next_opt_state

    return _Opt()
