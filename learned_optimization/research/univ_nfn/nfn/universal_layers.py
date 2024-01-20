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

# pylint: disable=invalid-name
"""NF-Layers for any architecture's weight space."""

import collections
import functools
import itertools
import math
from typing import List
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from learned_optimization.research.univ_nfn.nfn import utils as nf_utils

LeafTuple = nf_utils.LeafTuple


def group_indices_by_perms(indices, perms):
  """Group indices by corresponding values in perms."""
  groups = collections.defaultdict(list)
  for index, perm in zip(indices, perms):
    groups[perm].append(index)
  return groups


def generate_partitions(items):
  """Generate all non-empty partitions of items."""
  if len(items) == 1:
    yield [items]
    return
  first = items[0]
  for subpart in generate_partitions(items[1:]):
    for n, subset in enumerate(subpart):
      yield subpart[:n] + [[first] + subset] + subpart[n + 1 :]
    yield [[first]] + subpart


def valid_partitions(indices, perms):
  """Only partitions of indices such that subsets share value in perms."""
  # Create a mapping from perm to indices
  groups_map = group_indices_by_perms(indices, perms)
  groups = list(groups_map.values())

  # Generate all possible products of non-empty subsets of the groups
  partitions = [
      itertools.chain(*p)
      for p in itertools.product(*[generate_partitions(g) for g in groups])
  ]

  for partition in partitions:
    yield [p for p in partition]


def get_repeated_idcs(lst):
  """Get indices of list for values that repeat."""
  index_map = {}
  for index, item in enumerate(lst):
    if item in index_map:
      index_map[item].append(index)
    else:
      index_map[item] = [index]
  rep_idcs = []
  for idcs in index_map.values():
    if len(idcs) > 1:
      rep_idcs.append(idcs)
  return rep_idcs


def diagonal_slice(X, matching_dims):
  """Take X's diagonal along some subset of dimensions."""
  # len(matching_dims) must be at least 2
  out = jnp.diagonal(X, axis1=matching_dims[0], axis2=matching_dims[1])
  for dim in matching_dims[2:]:
    # The diagonal dim is always kept at the end.
    out = jnp.diagonal(out, axis1=dim, axis2=-1)
  return out


def move_dims_to_end(X, dims):
  """Move the specified dims of X to the end."""
  new_in_order = remove_idcs(list(range(X.ndim)), dims) + dims
  return jnp.transpose(X, new_in_order)


def remove_idcs(lst, idcs):
  return [l for idx, l in enumerate(lst) if idx not in idcs]


def find_missing_indices(lstA, lstB):
  """Find the indices for elements of A not in B."""
  setB = set(lstB)
  missing = []
  for i, el in enumerate(lstA):
    if el not in setB:
      missing.append(i)
  return missing


def is_broadcastable(shp1, shp2):
  """Check two array shapes are broadcastable."""
  for a, b in zip(shp1[::-1], shp2[::-1]):
    if a == 1 or b == 1 or a == b:
      pass
    else:
      return False
  return True


def match_dim_order(X, X_idcs, out_idcs):
  """Given X and idcs naming each dimension, reorder them to match out_idcs."""
  X_idcs_copy = X_idcs[:]
  perm = []
  for idx in out_idcs:
    if idx in X_idcs_copy:
      i = X_idcs_copy.index(idx)
      perm.append(i)
      X_idcs_copy[i] = None  # "use up" this index.
  X = jnp.transpose(X, perm)
  X_idcs = [X_idcs[i] for i in perm]
  return X, X_idcs


def gen_terms(out_spec, in_spec, out_shape, inp):
  """Generate the basis terms of an equivariant map from in_spec to out_spec."""
  terms = []
  k1, k2 = len(out_spec), len(in_spec)  # number of output/input indices
  for partition in valid_partitions(list(range(k1 + k2)), out_spec + in_spec):
    indices = [-1] * (k1 + k2)
    for subset_idx, subset in enumerate(partition):
      for dim in subset:
        indices[dim] = subset_idx
    out_indices, in_indices = indices[:k1], indices[k1:]
    term = inp
    # For repeated input idcs, take the corresponding diagonals
    while rep_idcs := get_repeated_idcs(in_indices):
      term = diagonal_slice(term, rep_idcs[0])
      # Diagonal is a new dim at the end
      idx = in_indices[rep_idcs[0][0]]
      in_indices = remove_idcs(in_indices, rep_idcs[0])
      in_indices.append(idx)
    # Aggregate over in_indices that aren't in out_indices
    agg_idcs = find_missing_indices(in_indices, out_indices)
    term = jnp.mean(term, axis=agg_idcs)
    in_indices = remove_idcs(in_indices, agg_idcs)
    # If we are outputting onto a diagonal, construct a larger tensor and then
    # embed term into the appropriate diagonal.
    for diag_set in get_repeated_idcs(out_indices):
      dim_size, out_idx = out_shape[diag_set[0]], out_indices[diag_set[0]]
      matching_in_dims = [
          i for i, idx in enumerate(in_indices) if idx == out_idx
      ]
      num_new_dims = len(diag_set) - len(matching_in_dims)
      # move matching in dimensions to end
      term = move_dims_to_end(term, matching_in_dims)
      new_term = jnp.zeros(term.shape + (dim_size,) * num_new_dims)
      diag_slice = (jnp.s_[:],) * (new_term.ndim - len(diag_set)) + (
          jnp.arange(dim_size),
      ) * len(diag_set)
      if not matching_in_dims:
        term = jnp.expand_dims(term, axis=-1)
      term = new_term.at[diag_slice].add(term)
      in_indices = remove_idcs(in_indices, matching_in_dims)
      in_indices.extend([out_idx] * len(diag_set))
    # Permute term dimensions to match out_indices.
    term, in_indices = match_dim_order(term, in_indices, out_indices)
    # Broadcast
    term = jnp.expand_dims(
        term, axis=find_missing_indices(out_indices, in_indices)
    )
    terms.append(term)
  return terms


def build_init_fn(scale, shape):
  return lambda rng, _shape: scale * jrandom.normal(rng, shape)


class NFLinear(nn.Module):
  """Linear equivariant layer for an arbitrary weight space.

  Uses fixed parameters, so the weight space cannot be changed once layer is
  initialized.
  """

  c_out: int
  c_in: int
  w_init: str = "he"  # he or lecun or zeros

  @nn.compact
  def __call__(self, params, perm_spec):
    params_and_spec, tree_def = jtu.tree_flatten(
        jtu.tree_map(LeafTuple, params, perm_spec)
    )
    flat_params = [x[0] for x in params_and_spec]
    flat_spec = [x[1] for x in params_and_spec]
    L = len(flat_params)

    outs = []
    for i in range(L):  # output
      terms_i = []
      for j in range(L):  # input
        in_param, out_param = flat_params[j], flat_params[i]
        out_spec, in_spec = flat_spec[i], flat_spec[j]
        term_gen_fn = jax.vmap(
            functools.partial(gen_terms, out_spec, in_spec, out_param.shape),
            in_axes=-1,
            out_axes=-1,
        )
        terms_i.extend(term_gen_fn(in_param))
      fan_in = self.c_in * len(terms_i)
      if self.w_init == "he":
        scale = math.sqrt(2 / fan_in)
      elif self.w_init == "lecun":  # lecun
        scale = math.sqrt(1 / fan_in)
      else:  # zeros
        scale = 0.0
      shape = (len(terms_i), self.c_in, self.c_out)
      theta_i = self.param(f"theta_{i}", build_init_fn(scale, shape), shape)
      out = 0
      for j, term in enumerate(terms_i):
        out += term @ theta_i[j]
      bias_i = self.param(
          f"bias_{i}", nn.initializers.zeros_init(), (self.c_out,)
      )
      out += bias_i
      outs.append(out)

    return jtu.tree_unflatten(tree_def, outs)


class PointwiseInitNFLinear(nn.Module):
  """NFLinear initialized to zeros + a linear layer applied pointwise."""
  c_out: int
  c_in: int

  @nn.compact
  def __call__(self, params, perm_spec):
    nf_part = NFLinear(self.c_out, self.c_in, w_init="zeros")(params, perm_spec)
    ptwise_mlp = nn.Dense(self.c_out, use_bias=False)
    ptwise_part = jtu.tree_map(ptwise_mlp, params)
    return jtu.tree_map(lambda x, y: x + y, nf_part, ptwise_part)


class NFPool(nn.Module):
  """Universal neural functional pooling op, *without* any parameters."""
  @nn.compact
  def __call__(self, params, perm_spec):
    params_and_spec, tree_def = jtu.tree_flatten(
        jtu.tree_map(LeafTuple, params, perm_spec)
    )
    flat_params = [x[0] for x in params_and_spec]
    flat_spec = [x[1] for x in params_and_spec]
    L = len(flat_params)

    outs = []
    for i in range(L):  # output
      terms_i = []
      for j in range(L):  # input
        in_param, out_param = flat_params[j], flat_params[i]
        out_spec, in_spec = flat_spec[i], flat_spec[j]
        term_gen_fn = jax.vmap(
            functools.partial(gen_terms, out_spec, in_spec, out_param.shape),
            in_axes=-1,
            out_axes=-1,
        )
        terms_i.extend(term_gen_fn(in_param))
      out = 0
      for term in terms_i:  # mean pooling
        out += term / len(terms_i)
      outs.append(out)
    return jtu.tree_unflatten(tree_def, outs)


class PointwiseDense(nn.Module):
  c_out: int
  @nn.compact
  def __call__(self, params, perm_spec):
    return jtu.tree_map(nn.Dense(self.c_out), params)


class NFDropout(nn.Module):
  dropout: float

  @nn.compact
  def __call__(self, params, train):
    drop = nn.Dropout(self.dropout)
    return jtu.tree_map(lambda x: drop(x, deterministic=not train), params)


def nf_relu(params):
  """Apply relu to each ws feature."""
  return jtu.tree_map(nn.relu, params)


def nf_pool(params):
  """Pool permutable dims in params and then concatenate."""

  def _arr_pool(x):
    axes = list(range(x.ndim - 1))
    return jnp.mean(x, axis=axes)

  return jnp.concatenate(
      jtu.tree_flatten(jtu.tree_map(_arr_pool, params))[0], -1
  )


# Define batched layers
BatchNFLinear = nn.vmap(
    NFLinear,
    in_axes=(0, None),
    out_axes=0,
    variable_axes={"params": None},
    split_rngs={"params": False},
)
batch_nf_pool = jax.vmap(nf_pool, in_axes=0, out_axes=0)


class NFLinearCNN(nn.Module):
  """Layer from Zhou et al 2023, for CNNs."""
  c_out: int
  c_in: int
  w_init = "lecun"

  @nn.compact
  def __call__(self, params, spec):
    # We assume the spec is of the form {"conv2_d", "conv2_d_1", ..., "linear"}
    num_layers = len(params)
    if self.w_init == "zeros":
      w_scale = 0
      b_scale = 0
    else:
      w_scale = math.sqrt(2 / (self.c_in * (2 * num_layers + 7)))
      b_scale = math.sqrt(2 / (self.c_in * (2 * num_layers + 3)))
    out = {}
    lnames, ws, bs = [], [], []
    allsums_w, rsums_w, csums_w, sums_b = [], [], [], []
    w_c_ins, w_c_outs, spatial_dims = [], [], []
    for i in range(num_layers):
      if i == 0:
        lname = "conv2_d"
      elif i < num_layers - 1:
        lname = f"conv2_d_{i}"
      else:
        lname = "linear"
      lnames.append(lname)
      # (num_in, num_out, c_in)
      in_w = params[lname]["w"]
      if lname.startswith("conv"):
        # (k, k, num_in, num_out, c_in) -> (num_in, num_out, c_in * k * k)
        w_c_ins.append(self.c_in * in_w.shape[0] * in_w.shape[1])
        w_c_outs.append(self.c_out * in_w.shape[0] * in_w.shape[1])
        spatial_dims.append((in_w.shape[0], in_w.shape[1]))
        in_w = jnp.transpose(in_w, (2, 3, 4, 0, 1))
        array_shapes = in_w.shape[:2]
        in_w = jnp.reshape(in_w, array_shapes + (-1,))
      else:
        w_c_ins.append(self.c_in)
        w_c_outs.append(self.c_out)
        spatial_dims.append(())
      # (bs, num_out, c_in)
      in_b = params[lname]["b"]
      ws.append(in_w)
      bs.append(in_b)
      rsum, csum = jnp.mean(in_w, axis=0), jnp.mean(in_w, axis=1)
      rsums_w.append(rsum)
      csums_w.append(csum)
      allsums_w.append(jnp.mean(rsum, axis=0))
      sums_b.append(jnp.mean(in_b, axis=0))

    sums_w = jnp.concatenate(allsums_w, -1)  # (c_in * L)
    sums_b = jnp.concatenate(sums_b, -1)  # (c_in * L)
    sums_wb = jnp.concatenate([sums_w, sums_b], -1)  # (c_in * L * 2)

    for i, (lname, w_c_in, w_c_out) in enumerate(zip(lnames, w_c_ins, w_c_outs)):
      # Compute out_w
      theta_w_w = self.param(
        f"theta_w{i}_w{i}",
        lambda rng, _shape: w_scale * jrandom.normal(rng, (w_c_in, w_c_out)),
        (w_c_in, w_c_out))
      out_w = jnp.einsum("jkc,cd->jkd", ws[i], theta_w_w)
      theta_w_allwb = self.param(
        f"theta_w{i}_allwb",
        lambda rng, _shape: w_scale * jrandom.normal(rng, (sums_wb.shape[-1], w_c_out)),
        (sums_wb.shape[-1], w_c_out))
      out_w += jnp.expand_dims(sums_wb[None] @ theta_w_allwb, axis=0)
      theta_w_wcol = self.param(
        f"theta_w{i}_wcol",
        lambda rng, _shape: w_scale * jrandom.normal(rng, (w_c_in, w_c_out)),
        (w_c_in, w_c_out))
      out_w += jnp.expand_dims(
          jnp.einsum("jc,cd->jd", csums_w[i], theta_w_wcol), axis=1)
      theta_w_wrow = self.param(
        f"theta_w{i}_wrow",
        lambda rng, _shape: w_scale * jrandom.normal(rng, (w_c_in, w_c_out)),
        (w_c_in, w_c_out))
      out_w += jnp.expand_dims(
          jnp.einsum("kc,cd->kd", rsums_w[i], theta_w_wrow), axis=0)
      theta_w_b = self.param(
        f"theta_w{i}_b",
        lambda rng, _shape: w_scale * jrandom.normal(rng, (self.c_in, w_c_out)),
        (self.c_in, self.c_out))
      out_w += jnp.expand_dims(
          jnp.einsum("kc,cd->kd", bs[i], theta_w_b), axis=0)
      if i > 0:
        theta_w_wm1 = self.param(
          f"theta_w{i}_wm1",
          lambda rng, _shape: w_scale * jrandom.normal(rng, (w_c_ins[i-1], w_c_out)),
          (w_c_ins[i-1], w_c_out))
        out_w += jnp.expand_dims(
            jnp.einsum("jc,cd->jd", rsums_w[i - 1], theta_w_wm1,), axis=1)
        theta_w_bm1 = self.param(
          f"theta_w{i}_bm1",
          lambda rng, _shape: w_scale * jrandom.normal(rng, (self.c_in, w_c_out)),
          (self.c_in, self.c_out))
        out_w += jnp.expand_dims(
            jnp.einsum("jc,cd->jd", bs[i - 1], theta_w_bm1), axis=1)
      if i < num_layers - 1:
        theta_w_wp1 = self.param(
          f"theta_w{i}_wp1",
          lambda rng, _shape: w_scale * jrandom.normal(rng, (w_c_ins[i+1], w_c_out)),
          (w_c_ins[i+1], w_c_out))
        out_w += jnp.expand_dims(
            jnp.einsum("kc,cd->kd", csums_w[i + 1], theta_w_wp1), axis=0)
      # Compute out_b
      theta_bb = self.param(
        f"theta_bb{i}",
        lambda rng, _shape: b_scale * jrandom.normal(rng, (self.c_in, self.c_out)),
        (self.c_in, self.c_out))
      out_b = jnp.einsum("jc,cd->jd", bs[i], theta_bb)
      theta_b_wcol = self.param(
        f"theta_b{i}_wcol",
        lambda rng, _shape: b_scale * jrandom.normal(rng, (w_c_in, self.c_out)),
        (w_c_in, self.c_out))
      out_b += jnp.einsum("kc,cd->kd", rsums_w[i], theta_b_wcol)
      if i < num_layers - 1:
        theta_b_wp1 = self.param(
          f"theta_b{i}_wp1",
          lambda rng, _shape: b_scale * jrandom.normal(rng, (w_c_ins[i+1], self.c_out)),
          (w_c_ins[i+1], self.c_out))
        out_b += jnp.einsum("kc,cd->kd", csums_w[i + 1], theta_b_wp1)
      theta_b_allwb = self.param(
        f"theta_b{i}_allwb",
        lambda rng, _shape: b_scale * jrandom.normal(rng, (sums_wb.shape[-1], self.c_out)),
        (sums_wb.shape[-1], self.c_out))
      out_b += sums_wb[None] @ theta_b_allwb
      if lname.startswith("conv"):
        #  (num_in, num_out, c_in * k * k) -> (k, k, num_in, num_out, c_in)
        kw, kh = spatial_dims[i]
        out_w = jnp.reshape(out_w, out_w.shape[:2] + (self.c_out, kw, kh))
        out_w = jnp.transpose(out_w, (3, 4, 0, 1, 2))
      out[lname] = {"w": out_w, "b": out_b}
    return out


class PointwiseInitNFLinearCNN(nn.Module):
  c_out: int
  c_in: int

  @nn.compact
  def __call__(self, params, perm_spec):
    nf_part = NFLinearCNN(self.c_out, self.c_in, w_init="zeros")(params, perm_spec)
    ptwise_mlp = nn.Dense(self.c_out, use_bias=False)
    ptwise_part = jtu.tree_map(ptwise_mlp, params)
    return jtu.tree_map(lambda x, y: x + y, nf_part, ptwise_part)


class UniversalSequential(nn.Module):
  layers: List  # pylint: disable=g-bare-generic

  @nn.compact
  def __call__(self, params, spec):
    out = params
    for layer in self.layers:
      if isinstance(layer, (NFLinear, PointwiseInitNFLinear, NFLinearCNN, PointwiseInitNFLinearCNN, NFPool, PointwiseDense)):
        out = layer(out, spec)
      else:
        out = layer(out)
    return out
