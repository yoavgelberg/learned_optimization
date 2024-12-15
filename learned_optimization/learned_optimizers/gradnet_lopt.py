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

from typing import Any, Optional, Callable
import functools

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

class _Invertable:
  """Base class to help manage hparam transformations."""

  def __init__(self, forward: Callable[[jnp.ndarray], jnp.ndarray],
               inverse: Callable[[jnp.ndarray], jnp.ndarray]):
    self.forward = jax.jit(forward)
    self.inverse = jax.jit(inverse)

  @functools.partial(jax.jit, static_argnums=0)
  def tree_inverse_forward(self, val):
    f = lambda v: self.forward(self.inverse(v))
    return jax.tree_util.tree_map(f, val)

_scaled_lr = _Invertable(
    forward=lambda x: 0.1 * jnp.log(x),
    inverse=lambda x: jnp.clip(jnp.exp(10. * x), 1e-8, 1e3))

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
               ds_hidden_size=32,
               ds_hidden_layers=3,
               gradnet_hidden_size=32,
               gradnet_hidden_layers=3,
               grandet_features=8,
               compute_summary=True):

    super().__init__()
    self._step_mult = step_mult
    self._exp_mult = exp_mult
    self._compute_summary = compute_summary
    self._gradnet_features = grandet_features

    def ff_gradnet(inp):
      return hk.nets.MLP([gradnet_hidden_size] * gradnet_hidden_layers + [grandet_features])(inp)

    def ff_ds(inp):
      return hk.nets.MLP([ds_hidden_size] * ds_hidden_layers + [2])(inp)

    self._gradnet_mod = hk.without_apply_rng(hk.transform(ff_gradnet))
    self._ds_mod = hk.without_apply_rng(hk.transform(ff_ds))

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)

    return flax.core.FrozenDict({
      # Init GradNet
      "gradnet": self._gradnet_mod.init(subkey1, jnp.zeros([0, 2])),
      # Init DeepSet
      "ds": self._ds_mod.init(subkey2, jnp.zeros([0, 19 + 2 * self._gradnet_features]))
    })

  def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: bool = False) -> opt_base.Optimizer:
    decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])

    ds_mod = self._ds_mod
    gradnet_mod = self._gradnet_mod
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
        # Clip grads
        # grad = jax.tree_util.tree_map(lambda x: jnp.clip(x, -1., 1.), grad)
        # print([a.shape for a in activations])
        # print([t.shape for t in tangents])

        # List num_layers tensors of shape (B, N_i, 2), where B is the input
        # batch size, N_i is the number of neurons in layer i. The first element of
        # the pair is the activation and the second is the pre-activation gradient.
        # For the first layer the activation is repeated, for the last layer the
        # gradient is repeated.
        neuron_features = [jnp.stack([activations[0], activations[0]], axis=-1)]
        for i in range(1, len(activations)-1):
          neuron_features.append(
            jnp.stack([activations[i], tangents[i - 1]], axis=-1)
          )
        neuron_features.append(jnp.stack([tangents[-1], tangents[-1]], axis=-1))

        # Apply GradNet
        neuron_features = [jnp.mean(gradnet_mod.apply(theta["gradnet"], nf), axis=0) for nf in neuron_features]

        _fisher_features = []
        for i in range(len(neuron_features)-1):
          low = neuron_features[i]
          high = neuron_features[i  + 1]

          low_neurons = low.shape[0]
          high_neurons = high.shape[0]

          low_expanded = jnp.repeat(low[:, jnp.newaxis, :], high_neurons, axis=1)
          high_expanded = jnp.repeat(high[jnp.newaxis, :, :], low_neurons, axis=0)

          _fisher_features.append(jnp.concatenate([low_expanded,  high_expanded], axis=-1))

        fish_feat = {
          "model/linear": {
            'w': _fisher_features[0],
            'b': jnp.concatenate([neuron_features[1], neuron_features[1]], axis=-1)
          }, 
          "model/linear_1": {
            'w': _fisher_features[1],
            'b': jnp.concatenate([neuron_features[2], neuron_features[2]], axis=-1)
            }
          }

        next_rolling_features = common.vec_rolling_mom(decays).update(
            opt_state.rolling_features, grad)

        training_step_feature = _tanh_embedding(opt_state.iteration)

        def _update_tensor(p, g, m, f):
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
          output = ds_mod.apply(theta["ds"], inp)

          # split the 2 outputs up into a direction and a magnitude
          direction = output[..., 0]
          magnitude = output[..., 1]

          # compute the step
          step = direction * jnp.exp(magnitude * exp_mult) * step_mult
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
                                             grad, next_rolling_features.m, fish_feat)
        next_opt_state = GradNetLOptState(
            params=tree_utils.match_type(next_params, opt_state.params),
            rolling_features=tree_utils.match_type(next_rolling_features,
                                                   opt_state.rolling_features),
            iteration=opt_state.iteration + 1,
            state=model_state)
        return next_opt_state

    return _Opt()
