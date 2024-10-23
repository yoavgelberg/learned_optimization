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

"""MLP learned optimizer with adafactor features and nominal AggMo term."""
import functools
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
from learned_optimization.research.univ_nfn.learned_opt import learned_opts as nfn_lopts
from haiku.data_structures import to_mutable_dict
import numpy as onp

PRNGKey = jnp.ndarray


def second_moment_normalizer(x, axis, eps=1e-5):
  return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def tanh_embedding(x):
  f32 = jnp.float32

  def one_freq(timescale):
    return jnp.tanh(x / (f32(timescale)) - 1.0)

  timescales = jnp.asarray(
      [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
      dtype=jnp.float32)
  return jax.vmap(one_freq)(timescales)


@flax.struct.dataclass
class AdafacMLPLOptState:
  params: Any
  state: Any
  mom_rolling: common.MomAccumulator
  rms_rolling: common.RMSAccumulator
  fac_rolling_features: common.FactoredAccum
  num_steps: jnp.ndarray
  iteration: jnp.ndarray


def decay_to_param(x):
  return jnp.log(1 - x) / 10.


def param_to_decay(x):
  return 1 - jnp.exp(x * 10.)


@gin.configurable
class MLPNomLOpt(lopt_base.LearnedOptimizer):
  """MLP based learned optimizer with adafactor style accumulators."""

  def __init__(self,
               task,
               exp_mult=(0.001, 0.001, 0.001),
               step_mult=0.001,
               initial_momentum_decays=(0.9, 0.99, 0.999),
               initial_rms_decays=(0.999,),
               initial_adafactor_decays=(0.9, 0.99, 0.999),
               nominal_stepsize=0.,
               weight_decay=0.,
               nominal_controller=False,
               regularization_controller=False,
               nominal_grad_estimator="AdamAggMo",
               aggregate_magnitude=False,
               aggregate_nom_magnitude=False,
               aggregate_reg_magnitude=False,
               normalize_blackbox=False,
               selfnormalize_blackbox=False,
               learnable_hp=False,
               # Match hidden_channels and num_layers defaults in original code.
               hidden_channels=4,
               num_layers=2,
               method="nfn"):
    super().__init__()
    self._exp_mult = onp.array(exp_mult)
    self._step_mult = step_mult
    self._initial_momentum_decays = initial_momentum_decays
    self._initial_rms_decays = initial_rms_decays
    self._initial_adafactor_decays = initial_adafactor_decays
    self._stepsize = nominal_stepsize
    self._nom_controller = nominal_controller
    assert not nominal_controller
    self._reg_controller = regularization_controller
    self._weight_decay = weight_decay
    self._nominal_grad_estimator = nominal_grad_estimator
    self._aggregate_magnitude = aggregate_magnitude
    self._aggregate_nom_magnitude = aggregate_nom_magnitude
    self._aggregate_reg_magnitude = aggregate_reg_magnitude
    self._normalize_blackbox = normalize_blackbox
    self._selfnormalize_blackbox = selfnormalize_blackbox
    self._learnable_hp = learnable_hp

    self._example_params = to_mutable_dict(task.init(jax.random.PRNGKey(0)))
    if method == "nfn":
      perm_spec = nfn_lopts.make_hk_perm_spec(self._example_params)
      self._network = nfn_lopts.UnivNFNForOpt(
          in_channels=39,
          hidden_channels=hidden_channels,
          out_channels=4,
          num_layers=num_layers,
          perm_spec=perm_spec,
          ptwise_init=True,
      )
    elif method == "nfn_hybrid":
      perm_spec = nfn_lopts.make_hk_perm_spec(self._example_params)
      self._network = nfn_lopts.HybridMLPNFN(
          in_channels=39,
          hidden_channels=hidden_channels,
          out_channels=4,
          num_layers=num_layers,
          perm_spec=perm_spec,
          ptwise_init=True,
      )
    else:
      self._network = nfn_lopts.MLPForOpt(
          hidden_channels=hidden_channels,
          out_channels=4,
          num_layers=num_layers+1,
          pos_emb=False
      )

  def _inp(self, global_feat, p, g, m, rms, fac_g, fac_vec_col, fac_vec_row,
           fac_vec_v):
    # this doesn't work with scalar parameters, so instead lets just reshape.
    if not p.shape:
      p = jnp.expand_dims(p, 0)
      g = jnp.expand_dims(g, 0)
      m = jnp.expand_dims(m, 0)
      rms = jnp.expand_dims(rms, 0)
      fac_g = jnp.expand_dims(fac_g, 0)
      fac_vec_v = jnp.expand_dims(fac_vec_v, 0)
      fac_vec_col = jnp.expand_dims(fac_vec_col, 0)
      fac_vec_row = jnp.expand_dims(fac_vec_row, 0)
    inps = []

    inps.append(jnp.expand_dims(g, axis=-1))
    inps.append(jnp.expand_dims(p, axis=-1))
    inps.append(m)
    inps.append(rms)
    rsqrt = lax.rsqrt(rms + 1e-6)
    adam_feats = m * rsqrt
    inps.append(adam_feats)
    inps.append(rsqrt)
    inps.append(fac_g)

    factored_dims = common.factored_dims(g.shape)
    if factored_dims is not None:
      # Construct features for
      d1, d0 = factored_dims

      # add 2 dims: 1 for batch of decay, one because low rank
      to_tile = [1] * (1 + len(g.shape))
      to_tile[d0] = g.shape[d0]

      row_feat = jnp.tile(jnp.expand_dims(fac_vec_row, axis=d0), to_tile)

      to_tile = [1] * (1 + len(g.shape))
      to_tile[d1] = g.shape[d1]
      col_feat = jnp.tile(jnp.expand_dims(fac_vec_col, axis=d1), to_tile)

      # 3 possible kinds of adafactor style features.
      # Raw values
      inps.append(row_feat)
      inps.append(col_feat)

      # 1/sqrt
      inps.append(lax.rsqrt(row_feat + 1e-8))
      inps.append(lax.rsqrt(col_feat + 1e-8))

      # multiplied by momentum
      reduced_d1 = d1 - 1 if d1 > d0 else d1
      row_col_mean = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)

      row_factor = common.safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
      col_factor = common.safe_rsqrt(fac_vec_col)
      fac_mom_mult = (
          m * jnp.expand_dims(row_factor, axis=d0) *
          jnp.expand_dims(col_factor, axis=d1))
      inps.append(fac_mom_mult)
    else:
      # In the non-factored case, match what RMSProp does.
      inps.append(fac_vec_v)
      inps.append(fac_vec_v)

      inps.append(lax.rsqrt(fac_vec_v + 1e-8))
      inps.append(lax.rsqrt(fac_vec_v + 1e-8))

      fac_mom_mult = m * (fac_vec_v + 1e-6)**-0.5
      inps.append(fac_mom_mult)

    # concat the inputs, normalize
    inp_stack = jnp.concatenate(inps, axis=-1)
    axis = list(range(len(p.shape)))
    inp_stack = second_moment_normalizer(inp_stack, axis=axis)

    # add features that should not be normalized
    training_step_feature = global_feat["training_step_feature"]
    stacked = jnp.reshape(training_step_feature, [1] * len(axis) +
                          list(training_step_feature.shape[-1:]))
    stacked = jnp.tile(stacked, list(p.shape) + [1])
    inp_stack = jnp.concatenate([inp_stack, stacked], axis=-1)
    return inp_stack

  def _produce_update(self, p, m, g, rms, out, step_mult, exp_mult, stepsize, mom_decay, rms_decay):
    m_corrected = m / (1 - mom_decay)
    rms_corrected = rms / (1 - rms_decay)
    # rsqrt = lax.rsqrt(rms + 1e-6)
    rsqrt = 1 / (jnp.sqrt(rms_corrected) + 1e-8)  # matches optax eps=1e-8, eps_root=0
    adam_feats = m_corrected * rsqrt
    direction = out[..., 0]
    magnitude = out[..., 1]
    if self._nom_controller:
      nom_magnitude = out[..., 2]
    else:
      nom_magnitude = jnp.ones_like(out[..., 0])

    if self._reg_controller:
      reg_magnitude = out[..., 3]
    else:
      reg_magnitude = jnp.zeros_like(out[..., 0])

    if self._aggregate_magnitude:
      magnitude = jnp.mean(magnitude)
    if self._aggregate_nom_magnitude:
      nom_magnitude = jnp.mean(nom_magnitude)
    if self._aggregate_reg_magnitude:
      reg_magnitude = jnp.mean(reg_magnitude)

    step = direction * jnp.exp(magnitude * exp_mult[0])
    step *= step_mult

    if self._normalize_blackbox:
      step = step * rsqrt[..., 0]

    if self._selfnormalize_blackbox:
      step = step * jnp.mean(g)

    step = step.reshape(p.shape)

    reg = (1. - self._weight_decay * jnp.exp(reg_magnitude * exp_mult[2]))
    reg = reg.reshape(p.shape)

    # nominal grad estimator
    if self._nominal_grad_estimator == "SGD":
      g_est = g
    elif self._nominal_grad_estimator == "SGDM":
      g_est = m[..., -1]
    elif self._nominal_grad_estimator == "AggMo":
      g_est = jnp.mean(m, axis=-1)
    elif self._nominal_grad_estimator == "Adam":
      g_est = adam_feats[..., 0]
    elif self._nominal_grad_estimator == "AdamAggMo":
      g_est = jnp.mean(adam_feats, axis=-1)
    else:
      raise NotImplementedError

    nom_step = stepsize * jnp.exp(
        nom_magnitude * exp_mult[1]) * g_est

    # compute updated params
    new_p = reg * p
    new_p -= step
    new_p -= nom_step
    return new_p

  def init(self, key: PRNGKey) -> lopt_base.MetaParams:
    # We meta-learn:
    # * weights of the MLP
    # * decays of momentum, RMS, and adafactor style accumulators
    example_inp = jax.tree_util.tree_map(
      lambda x: jnp.repeat(x[..., None], 39, -1), self._example_params)
    params = self._network.init(key, example_inp)
    return {
        "momentum_decays": jnp.zeros([len(self._initial_momentum_decays)]),
        "rms_decays": jnp.zeros([len(self._initial_rms_decays)]),
        "adafactor_decays": jnp.zeros([len(self._initial_adafactor_decays)]),
        "nn": params,
        "step_mult": jnp.log(jnp.asarray(self._step_mult)),
        "exp_mult": jnp.log(jnp.asarray(self._exp_mult)),
        "stepsize": jnp.log(jnp.asarray(self._stepsize)),
    }

  def opt_fn(self,
             theta: lopt_base.MetaParams,
             is_training: Optional[bool] = False) -> opt_base.Optimizer:

    network_fn = self._network.apply
    parent = self

    class _Opt(opt_base.Optimizer):
      """Optimizer capturing the meta params."""

      def __init__(self, theta):
        self.theta = theta

      def _get_rolling(self):
        mom_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_momentum_decays)) +  # pylint: disable=protected-access
            self.theta["momentum_decays"])
        mom_roll = common.vec_rolling_mom(mom_decay)

        rms_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_rms_decays)) +  # pylint: disable=protected-access
            self.theta["rms_decays"])
        rms_roll = common.vec_rolling_rms(rms_decay)

        adafactor_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_adafactor_decays)) +  # pylint: disable=protected-access
            self.theta["adafactor_decays"])
        fac_vec_roll = common.vec_factored_rolling(adafactor_decay)
        return mom_roll, rms_roll, fac_vec_roll

      def init(
          self,
          params: opt_base.Params,
          model_state: Optional[opt_base.ModelState] = None,
          num_steps: Optional[int] = None,
          key: Optional[PRNGKey] = None,
      ) -> AdafacMLPLOptState:
        if num_steps is None:
          raise ValueError("Must specify number of steps for this lopt!")

        mom_roll, rms_roll, fac_vec_roll = self._get_rolling()

        return AdafacMLPLOptState(
            params=params,
            state=model_state,
            rms_rolling=rms_roll.init(params),
            mom_rolling=mom_roll.init(params),
            fac_rolling_features=fac_vec_roll.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            num_steps=jnp.asarray(num_steps))

      def update(
          self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
          opt_state: AdafacMLPLOptState,
          grad: opt_base.Gradient,
          loss: jnp.ndarray,
          model_state: Optional[opt_base.ModelState] = None,
          is_valid: bool = False,
          key: Optional[PRNGKey] = None,
      ) -> AdafacMLPLOptState:
        mom_roll, rms_roll, fac_vec_roll = self._get_rolling()
        next_mom_rolling = mom_roll.update(opt_state.mom_rolling, grad)
        next_rms_rolling = rms_roll.update(opt_state.rms_rolling, grad)
        next_fac_rolling_features, fac_g = fac_vec_roll.update(
            opt_state.fac_rolling_features, grad)

        # compute some global features
        training_step_feature = tanh_embedding(opt_state.iteration)

        global_features = {
            "iterations": opt_state.iteration,
            "num_steps": opt_state.num_steps,
            "training_step_feature": training_step_feature,
        }

        inp_fn = functools.partial(parent._inp, global_features)
        inps = jax.tree_util.tree_map(
          inp_fn, opt_state.params, grad, next_mom_rolling.m,
          next_rms_rolling.rms, fac_g, next_fac_rolling_features.v_col,
          next_fac_rolling_features.v_row, next_fac_rolling_features.v_diag)
        out = network_fn(self.theta["nn"], inps)
        if parent._learnable_hp:
          step_mult = jnp.exp(self.theta["step_mult"])
          exp_mult = jnp.exp(self.theta["exp_mult"])
          stepsize = jnp.exp(self.theta["stepsize"])
        else:
          step_mult = parent._step_mult
          exp_mult = parent._exp_mult
          stepsize = parent._stepsize
        mom_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_momentum_decays)) +  # pylint: disable=protected-access
            self.theta["momentum_decays"])
        rms_decay = param_to_decay(
            decay_to_param(jnp.asarray(parent._initial_rms_decays)) +  # pylint: disable=protected-access
            self.theta["rms_decays"])
        produce_update = functools.partial(
          parent._produce_update,
          step_mult=step_mult,
          exp_mult=exp_mult,
          stepsize=stepsize,
          mom_decay=mom_decay,
          rms_decay=rms_decay)
        next_params = jax.tree_util.tree_map(
          produce_update,
          opt_state.params,
          next_mom_rolling.m,
          grad, next_rms_rolling.rms, out)

        next_opt_state = AdafacMLPLOptState(
            params=next_params,
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            fac_rolling_features=next_fac_rolling_features,
            iteration=opt_state.iteration + 1,
            state=model_state,
            num_steps=opt_state.num_steps)

        return tree_utils.match_type(next_opt_state, opt_state)

    return _Opt(theta)
