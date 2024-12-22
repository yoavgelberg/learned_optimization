import functools

from typing import Any, Optional, List

import haiku as hk
import flax
import flax.linen as nn
import gin
import jax
from jax import lax
import jax.numpy as jnp
import jax.tree_util as jtu
from learned_optimization import summary
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.learned_optimizers import common
from learned_optimization.optimizers import base as opt_base
from learned_optimization.research.univ_nfn.nfn import universal_layers
from learned_optimization.research.univ_nfn.nfn import utils as nfu
from learned_optimization.research.univ_nfn.learned_opt import (
    SimpleOptState,
    HybridMLPNFN,
    UnivNFNForOpt,
    HetNFNForOpt,
    MLPForOpt,
)
from haiku._src import data_structures


MetaParams = Any
KeyArray = Any


def cat_tstep_feature(training_step_feature, x):
    """Concatenate training_step_feature along chan dim."""
    new_shape = x.shape[:-1] + training_step_feature.shape
    tstep = jnp.broadcast_to(training_step_feature, new_shape)
    return jnp.concatenate([x, tstep], -1)


def _tanh_embedding(iterations):
    f32 = jnp.float32

    def one_freq(timescale):
        return jnp.tanh(iterations / (f32(timescale)) - 1.0)

    timescales = jnp.asarray(
        [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
        dtype=jnp.float32,
    )
    return jax.vmap(one_freq)(timescales)


def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def make_hk_perm_spec(mlp_params):
    """Produces perm spec for a haiku mlp."""
    perm_spec = {}
    for i in range(len(mlp_params)):
        name = f"mlp/~/linear_{i}"
        perm_spec[name] = {"w": (i, i + 1), "b": (i + 1,)}
    return perm_spec


def make_hk_cnn_perm_spec(params, residual=False):
    """Produces perm spec for a haiku cnn."""
    perm_spec = {}
    num_convs = len([k for k in params if k.startswith("conv2_d")])
    neuron_idx = 0
    for i in range(num_convs):
        if i == 0:
            conv_name = "conv2_d"
            ln_name = "layer_norm"
        else:
            conv_name = f"conv2_d_{i}"
            ln_name = f"layer_norm_{i}"
        next_idx = neuron_idx + int((not residual) or (i == 0))
        perm_spec[conv_name] = {
            "w": (-i, -(len(params) + i), neuron_idx, next_idx),
            "b": (next_idx,),
        }
        if ln_name in params:  # layernorm is optional
            perm_spec[ln_name] = {"offset": (next_idx,), "scale": (next_idx,)}
        neuron_idx = next_idx
    perm_spec["linear"] = {
        "w": (neuron_idx, neuron_idx + 1),
        "b": (neuron_idx + 1,),
    }  # final linear layer
    print(perm_spec)
    return perm_spec


def make_hk_irnn_perm_spec(mlp_params):
    """Tested on RNNLM_lm1bbytes_Patch32_IRNN128_Embed64."""
    # -1: vocab, 0: embed, 1: hidden
    del mlp_params
    perm_spec = {
        "embed": {"embeddings": (-1, 0)},
        "irnn/linear": {"b": (1,), "w": (0, 1)},
        "irnn/linear_1": {"b": (1,), "w": (1, 1)},
        "linear": {"b": (-1,), "w": (1, -1)},
        "~": {"initial_state_0": (-2, 1)},
    }
    return perm_spec


def make_hk_transformer_perm_spec(mlp_params):
    """Make perm spec for a transformer_lm.

    Example:
      {'transformer/embed': {'embeddings': (32100, 32)},
      'transformer/h0_attn/key': {'b': (128,), 'w': (32, 128)},
      'transformer/h0_attn/linear': {'b': (32,), 'w': (128, 32)},
      'transformer/h0_attn/query': {'b': (128,), 'w': (32, 128)},
      'transformer/h0_attn/value': {'b': (128,), 'w': (32, 128)},
      'transformer/h0_ln_1': {'offset': (32,), 'scale': (32,)},
      'transformer/h0_ln_2': {'offset': (32,), 'scale': (32,)},
      'transformer/h0_mlp/linear': {'b': (128,), 'w': (32, 128)},
      'transformer/h0_mlp/linear_1': {'b': (32,), 'w': (128, 32)},
      'transformer/h1_attn/key': {'b': (128,), 'w': (32, 128)},
      'transformer/h1_attn/linear': {'b': (32,), 'w': (128, 32)},
      'transformer/h1_attn/query': {'b': (128,), 'w': (32, 128)},
      'transformer/h1_attn/value': {'b': (128,), 'w': (32, 128)},
      'transformer/h1_ln_1': {'offset': (32,), 'scale': (32,)},
      'transformer/h1_ln_2': {'offset': (32,), 'scale': (32,)},
      'transformer/h1_mlp/linear': {'b': (128,), 'w': (32, 128)},
      'transformer/h1_mlp/linear_1': {'b': (32,), 'w': (128, 32)},
      'transformer/h_f': {'offset': (32,), 'scale': (32,)},
      'transformer/linear': {'b': (32100,), 'w': (32, 32100)}}
    """
    # -1,-2: vocab, 0: embed, 1: hidden, 2: embed_2, 3: hidden_2, 4: embed_3
    del (mlp_params,)
    perm_spec = {
        "transformer/embed": {"embeddings": (-1, 0)},
        "transformer/h0_attn/key": {"b": (1,), "w": (0, 1)},
        "transformer/h0_attn/linear": {"b": (0,), "w": (1, 0)},
        "transformer/h0_attn/query": {"b": (1,), "w": (0, 1)},
        "transformer/h0_attn/value": {"b": (1,), "w": (0, 1)},
        "transformer/h0_ln_1": {"offset": (0,), "scale": (0,)},
        "transformer/h0_ln_2": {"offset": (0,), "scale": (0,)},
        "transformer/h0_mlp/linear": {"b": (1,), "w": (0, 1)},
        "transformer/h0_mlp/linear_1": {"b": (2,), "w": (1, 2)},
        "transformer/h1_attn/key": {"b": (3,), "w": (2, 3)},
        "transformer/h1_attn/linear": {"b": (2,), "w": (3, 2)},
        "transformer/h1_attn/query": {"b": (3,), "w": (2, 3)},
        "transformer/h1_attn/value": {"b": (3,), "w": (2, 3)},
        "transformer/h1_ln_1": {"offset": (2,), "scale": (2,)},
        "transformer/h1_ln_2": {"offset": (2,), "scale": (2,)},
        "transformer/h1_mlp/linear": {"b": (3,), "w": (2, 3)},
        "transformer/h1_mlp/linear_1": {"b": (4,), "w": (3, 4)},
        "transformer/h_f": {"offset": (4,), "scale": (4,)},
        "transformer/linear": {"b": (-2,), "w": (4, -2)},
    }
    return perm_spec


class ResidualOptWithGradnet(lopt_base.LearnedOptimizer):
    """NFN learning a modified version of SGD+momentum."""

    def __init__(
        self,
        network,
        example_params,
        gradnet_hidden_dims: List[int],
        gradnet_output_dim: int,
        out_mult=1e-4,
        step_mult=0.1,
        learnable_hp=False,
    ):
        self._network = network
        self._example_params = example_params
        self._out_mult = out_mult
        self._step_mult = step_mult
        self._learnable_hp = learnable_hp

        def gradnet_fn(inp):
            return hk.nets.MLP(gradnet_hidden_dims + [gradnet_output_dim])(inp)

        self._gradnet = hk.without_apply_rng(hk.transform(gradnet_fn))
        self._gradnet_output_dim = gradnet_output_dim

    def init(self, key: KeyArray) -> MetaParams:
        key, subkey1, subkey2 = jax.random.split(key, 3)

        fixed_params = jtu.tree_map(
            lambda x: jnp.repeat(x[..., None], 19 + self._gradnet_output_dim, -1),
            self._example_params,
        )
        gradnet_inp = jnp.zeros([0, 19 + self._gradnet_output_dim])
        return {
            "gradnet_params": self._gradnet.init(subkey1, gradnet_inp),
            "mod_params": self._network.init(subkey2, fixed_params),
            "step_mult": jnp.log(jnp.asarray(self._step_mult)),
            "out_mult": jnp.log(jnp.asarray(self._out_mult)),
            "one_minus_momentum": lopt_base.one_minus_log.forward(0.9),
        }

    def opt_fn(self, theta, is_training=False) -> opt_base.Optimizer:
        decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])
        network_fn = self._network.apply
        gradnet_fn = self._gradnet.apply
        if self._learnable_hp:
            step_mult, out_mult = jnp.exp(theta["step_mult"]), jnp.exp(
                theta["out_mult"]
            )
            mom_decay = lopt_base.one_minus_log.inverse(theta["one_minus_momentum"])
        else:
            step_mult, out_mult = self._step_mult, self._out_mult
            mom_decay = 0.9

        class _Opt(opt_base.Optimizer):
            """Optimizer instance which has captured the meta-params (theta)."""

            def init(
                self,
                params: lopt_base.Params,
                model_state: Any = None,
                num_steps: Optional[int] = None,
                key: Optional[KeyArray] = None,
            ) -> SimpleOptState:
                """Initialize inner opt state."""

                return SimpleOptState(
                    params=params,
                    state=model_state,
                    rolling_features=common.vec_rolling_mom(decays).init(params),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    momentum=common.rolling_mom(mom_decay).init(params),
                )

            def update(
                self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                opt_state: SimpleOptState,
                grad: Any,
                activations: Any,
                tangents: Any,
                loss: float,
                model_state: Any = None,
                is_valid: bool = False,
                key: Optional[KeyArray] = None,
            ) -> SimpleOptState:
                next_rolling_features = common.vec_rolling_mom(decays).update(
                    opt_state.rolling_features, grad
                )

                # List num_layers tensors of shape (B, N_i, 2), where B is the input
                # batch size, N_i is the number of neurons in layer i. The first element of
                # the pair is the activation and the second is the pre-activation gradient.
                # For the first layer the activation is repeated, for the last layer the
                # gradient is repeated.
                neuron_features = [jnp.stack([activations[0], activations[0]], axis=-1)]

                for i in range(1, len(activations) - 1):
                    neuron_features.append(
                        jnp.stack([activations[i], tangents[i - 1]], axis=-1)
                    )
                neuron_features.append(jnp.stack([tangents[-1], tangents[-1]], axis=-1))

                # Apply GradNet
                neuron_features = [
                    jnp.mean(gradnet_fn(theta["gradnet"], nf), axis=0)
                    for nf in neuron_features
                ]

                _fisher_features = []
                for i in range(len(neuron_features) - 1):
                    low = neuron_features[i]
                    high = neuron_features[i + 1]

                    low_neurons = low.shape[0]
                    high_neurons = high.shape[0]

                    low_expanded = jnp.repeat(
                        low[:, jnp.newaxis, :], high_neurons, axis=1
                    )
                    high_expanded = jnp.repeat(
                        high[jnp.newaxis, :, :], low_neurons, axis=0
                    )

                    _fisher_features.append(
                        jnp.concatenate([low_expanded, high_expanded], axis=-1)
                    )

                fish_feat = {
                    "model/linear": {
                        "w": _fisher_features[0],
                        "b": jnp.concatenate(
                            [neuron_features[1], neuron_features[1]], axis=-1
                        ),
                    },
                    "model/linear_1": {
                        "w": _fisher_features[1],
                        "b": jnp.concatenate(
                            [neuron_features[2], neuron_features[2]], axis=-1
                        ),
                    },
                }

                training_step_feature = _tanh_embedding(opt_state.iteration)  # (11,)

                # concatenate different input features
                inp_features = nfu.tree_concatenate(
                    [
                        fish_feat,
                        nfu.tree_expand_dims(opt_state.params, -1),
                        nfu.tree_expand_dims(grad, -1),
                        next_rolling_features.m,
                    ],
                    -1,
                )
                summary.summary("nfn_lopt/inp_rms_raw", nfu.tree_mean_rms(inp_features))
                inp_features = jtu.tree_map(lambda x: jnp.clip(x, -1, 1), inp_features)
                summary.summary(
                    "nfn_lopt/inp_rms_clipped", nfu.tree_mean_rms(inp_features)
                )

                def norm_second_moment(p):
                    norm_axis = list(range(len(p.shape)))
                    return _second_moment_normalizer(p, axis=norm_axis)

                inp_features = jtu.tree_map(norm_second_moment, inp_features)
                inp_features = jtu.tree_map(
                    functools.partial(cat_tstep_feature, training_step_feature),
                    inp_features,
                )
                out = nfu.tree_squeeze(
                    network_fn(theta["mod_params"], inp_features), -1
                )
                summary.summary("nfn_lopt/out_magn", nfu.tree_mean_magn(out))
                momentum = common.rolling_mom(mom_decay).update(
                    opt_state.momentum, grad
                )
                summary.summary("nfn_lopt/momentum_magn", nfu.tree_mean_magn(momentum))
                next_params = jtu.tree_map(
                    lambda p, o, m: p - step_mult * (out_mult * o + m),
                    opt_state.params,
                    out,
                    momentum.m,
                )
                summary.summary("nfn_lopt/mean_abs_mom", nfu.tree_mean_magn(momentum))
                next_opt_state = SimpleOptState(
                    params=tree_utils.match_type(next_params, opt_state.params),
                    rolling_features=tree_utils.match_type(
                        next_rolling_features, opt_state.rolling_features
                    ),
                    iteration=opt_state.iteration + 1,
                    state=model_state,
                    momentum=tree_utils.match_type(momentum, opt_state.momentum),
                )
                return next_opt_state

        return _Opt()


@gin.configurable
class ResidualOptNFNWithGradnet(ResidualOptWithGradnet):
    def __init__(
        self,
        task,
        step_mult=0.1,
        out_mult=1e-4,
        gradnet_hidden_dims: List[int] = [32, 32],
        gradnet_output_dim: int = 1,
        ptwise_init=False,
        pos_emb=False,
        nfn_type="default",
        conv_is_residual=False,
        learnable_hp=False,
    ):
        example_params = task.init(jax.random.PRNGKey(0))
        if "conv2_d" in example_params:
            perm_spec = make_hk_cnn_perm_spec(example_params, conv_is_residual)
        elif "irnn/linear" in example_params:
            perm_spec = make_hk_irnn_perm_spec(example_params)
        elif "transformer/embed" in example_params:
            perm_spec = make_hk_transformer_perm_spec(example_params)
        else:
            perm_spec = make_hk_perm_spec(example_params)
        if nfn_type == "hybrid":
            assert not pos_emb
            network = HybridMLPNFN(
                in_channels=19,
                hidden_channels=32,
                out_channels=1,
                num_layers=4,
                perm_spec=perm_spec,
                ptwise_init=ptwise_init,
            )
        elif nfn_type == "default":
            network = UnivNFNForOpt(
                in_channels=19,
                hidden_channels=32,
                out_channels=1,
                num_layers=4,
                perm_spec=perm_spec,
                ptwise_init=ptwise_init,
                pos_emb=pos_emb,
            )
        elif nfn_type == "het":
            assert not pos_emb
            assert not ptwise_init
            network = HetNFNForOpt(
                in_channels=19,
                hidden_channels=32,
                out_channels=1,
                num_layers=4,
                perm_spec=perm_spec,
            )
        else:
            raise NotImplementedError

        super().__init__(
            network,
            example_params,
            gradnet_hidden_dims,
            gradnet_output_dim,
            step_mult=step_mult,
            out_mult=out_mult,
            learnable_hp=learnable_hp,
        )


@gin.configurable
class ResidualOptMLPWithGradnet(ResidualOptWithGradnet):
    def __init__(
        self,
        task,
        step_mult=0.1,
        out_mult=1e-4,
        gradnet_hidden_dims: List[int] = [32, 32],
        gradnet_output_dim: int = 1,
        pos_emb=False,
        learnable_hp=False,
    ):
        example_params = task.init(jax.random.PRNGKey(0))
        network = MLPForOpt(
            hidden_channels=32, out_channels=1, num_layers=4, pos_emb=pos_emb
        )
        super().__init__(
            network,
            example_params,
            gradnet_hidden_dims,
            gradnet_output_dim,
            step_mult=step_mult,
            out_mult=out_mult,
            learnable_hp=learnable_hp,
        )
