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

"""Tasks based on MLP."""
# pylint: disable=invalid-name

from typing import Any, Mapping, Tuple, List

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from learned_optimization.tasks import base
from learned_optimization.tasks.datasets import image
import numpy as onp

from learned_optimization.hookable_mlp import HookableMLP

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray

class _HookableMLPImageTask(base.Task):
  """Hookable MLP based image task."""

  def __init__(self,
               datasets,
               hidden_sizes,
               act_fn=jax.nn.relu):
    super().__init__()
    num_classes = datasets.extra_info["num_classes"]
    sizes = list(hidden_sizes) + [num_classes]
    self.sizes = sizes
    self.datasets = datasets

    def _forward_with_bs(inp, bs):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return HookableMLP(hidden_sizes, output_dim=num_classes, activation=act_fn)(inp, *bs)

    def _forward(inp):
      return _forward_with_bs(inp, [jax.zeros([size]) for size in self.sizes])

    self._mod = hk.transform(_forward)
    self._mod_with_bs = hk.transform(_forward_with_bs)

  def init_with_bs(self, key: PRNGKey) -> Any:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._mod_with_bs.init(key, batch["image"], [jax.zeros([size]) for size in self.sizes])

  def init(self, key: PRNGKey) -> Any:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._mod.init(key, batch["image"])

  def loss_with_bs(self, params: Params, key: PRNGKey, data: Any, *bs) -> jnp.ndarray:  # pytype: disable=signature-mismatch  # jax-ndarray
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod_with_bs.apply(params, data["image"], *bs)
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss)

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:  # pytype: disable=signature-mismatch  # jax-ndarray
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod.apply(params, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss)

  def loss_with_bs_and_aux(
      self, params: Params, key: PRNGKey, data: Any, *bs) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    loss = self.loss_with_bs(params, key, data, *bs)
    return loss, {}  # pytype: disable=bad-return-type  # jax-ndarray

  def loss_with_bs_state_and_aux(self, params: Params, state: ModelState, key: PRNGKey, data: Any, *bs) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
    if state is not None:
      raise ValueError("Define a custom loss_with_state_and_aux when using a"
                       " state!")
    loss, aux = self.loss_with_bs_and_aux(params, key, data, *bs)
    return loss, None, aux

  def normalizer(self, loss):
    num_classes = self.datasets.extra_info["num_classes"]
    maxval = 1.5 * onp.log(num_classes)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)


@gin.configurable
def ImageMLP_Cifar10BW8_Relu32():
  """A 1 hidden layer, 32 unit MLP for 8x8 black and white cifar10."""
  datasets = image.cifar10_datasets(
      batch_size=128, image_size=(8, 8), convert_to_black_and_white=True)
  return _MLPImageTask(datasets, [32])

@gin.configurable
def HookableImageMLP_FashionMnist8_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
  datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(8, 8))
  return _HookableMLPImageTask(datasets, [32])

class _MLPImageTask(base.Task):
  """MLP based image task."""

  def __init__(self,
               datasets,
               hidden_sizes,
               act_fn=jax.nn.relu,
               dropout_rate=0.0):
    super().__init__()
    num_classes = datasets.extra_info["num_classes"]
    sizes = list(hidden_sizes) + [num_classes]
    self.datasets = datasets

    def _forward(inp):
      inp = jnp.reshape(inp, [inp.shape[0], -1])
      return hk.nets.MLP(
          sizes, activation=act_fn)(
              inp, dropout_rate=dropout_rate, rng=hk.next_rng_key())

    self._mod = hk.transform(_forward)

  def init(self, key: PRNGKey) -> Any:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    return self._mod.init(key, batch["image"])

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:  # pytype: disable=signature-mismatch  # jax-ndarray
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod.apply(params, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss)

  def normalizer(self, loss):
    num_classes = self.datasets.extra_info["num_classes"]
    maxval = 1.5 * onp.log(num_classes)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)


@gin.configurable
def ImageMLP_Cifar10BW8_Relu32():
  """A 1 hidden layer, 32 unit MLP for 8x8 black and white cifar10."""
  datasets = image.cifar10_datasets(
      batch_size=128, image_size=(8, 8), convert_to_black_and_white=True)
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_FashionMnist_Relu128x128():
  """A 2 hidden layer, 128 hidden unit MLP designed for fashion mnist."""
  datasets = image.fashion_mnist_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128])


@gin.configurable
def ImageMLP_FashionMnist8_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
  datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(8, 8))
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_FashionMnist16_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
  datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(16, 16))
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_FashionMnist32_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
  datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(32, 32))
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_Cifar10_8_Relu32():
  """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 cifar10."""
  datasets = image.cifar10_datasets(batch_size=128, image_size=(8, 8))
  return _MLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_Imagenet16_Relu256x256x256():
  """A 3 hidden layer MLP trained on 16x16 resized imagenet."""
  datasets = image.imagenet16_datasets(batch_size=128)
  return _MLPImageTask(datasets, [256, 256, 256])


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def ImageMLP_Cifar100_128x128x128_Relu():
  datasets = image.cifar100_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Tanh_bs64():
  datasets = image.cifar10_datasets(batch_size=64)
  return _MLPImageTask(datasets, [128, 128, 128], act_fn=jnp.tanh)


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Tanh_bs128():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128, 128], act_fn=jnp.tanh)


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Tanh_bs256():
  datasets = image.cifar10_datasets(batch_size=256)
  return _MLPImageTask(datasets, [128, 128, 128], act_fn=jnp.tanh)


@gin.configurable
def ImageMLP_Mnist_128x128x128_Relu():
  datasets = image.mnist_datasets(batch_size=128)
  return _MLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def ImageMLP_Cifar10_256x256_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [256, 256])


@gin.configurable
def ImageMLP_Cifar10_256x256_Relu_BS32():
  datasets = image.cifar10_datasets(batch_size=32)
  return _MLPImageTask(datasets, [256, 256])


@gin.configurable
def ImageMLP_Cifar10_1024x1024_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [1024, 1024])


@gin.configurable
def ImageMLP_Cifar10_4096x4096_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [4096, 4096])


@gin.configurable
def ImageMLP_Cifar10_8192x8192_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [8192, 8192])


@gin.configurable
def ImageMLP_Cifar10_16384x16384_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTask(datasets, [16384, 16384])


class _MLPImageTaskMSE(_MLPImageTask):
  """Image model with a Mean squared error loss."""

  def loss(self, params: Params, key: PRNGKey, data: Any) -> jnp.ndarray:
    num_classes = self.datasets.extra_info["num_classes"]
    logits = self._mod.apply(params, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    return jnp.mean(jnp.square(logits - labels))

  def normalizer(self, loss):
    maxval = 1.0
    loss = jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)
    return jnp.minimum(loss, 1.0) * 10


@gin.configurable
def ImageMLP_Cifar10_128x128x128_Relu_MSE():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskMSE(datasets, [128, 128, 128])


@gin.configurable
def ImageMLP_Cifar10_128x128_Dropout05_Relu_MSE():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskMSE(datasets, [128, 128], dropout_rate=0.5)


@gin.configurable
def ImageMLP_Cifar10_128x128_Dropout08_Relu_MSE():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskMSE(datasets, [128, 128], dropout_rate=0.8)


@gin.configurable
def ImageMLP_Cifar10_128x128_Dropout02_Relu_MSE():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskMSE(datasets, [128, 128], dropout_rate=0.2)


class _MLPImageTaskNorm(base.Task):
  """MLP based image task with layer norm."""

  def __init__(
      self,
      datasets,  # pylint: disable=super-init-not-called
      hidden_sizes,
      norm_type,
      act_fn=jax.nn.relu):
    self.datasets = datasets
    num_classes = datasets.extra_info["num_classes"]
    sizes = list(hidden_sizes) + [num_classes]

    def _forward(inp):
      net = jnp.reshape(inp, [inp.shape[0], -1])

      for i, h in enumerate(sizes):
        net = hk.Linear(h)(net)
        if i != (len(sizes) - 1):
          if norm_type == "layer_norm":
            net = hk.LayerNorm(
                axis=1, create_scale=True, create_offset=True)(
                    net)
          elif norm_type == "batch_norm":
            net = hk.BatchNorm(
                create_scale=True, create_offset=True, decay_rate=0.9)(
                    net, is_training=True)
          else:
            raise ValueError(f"No norm {norm_type} implemented!")
          net = act_fn(net)
      return net

    # Batchnorm has state -- though we don't use it here
    # (we are using training mode only for this loss.)
    self._mod = hk.transform_with_state(_forward)

  def init_with_state(self, key: PRNGKey) -> Any:
    batch = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype),
                                   self.datasets.abstract_batch)
    params, state = self._mod.init(key, batch["image"])
    return params, state

  def loss_with_state(self, params: Params, state: ModelState, key: PRNGKey,
                      data: Any) -> Tuple[jnp.ndarray, ModelState]:
    num_classes = self.datasets.extra_info["num_classes"]
    logits, state = self._mod.apply(params, state, key, data["image"])
    labels = jax.nn.one_hot(data["label"], num_classes)
    vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(vec_loss), state

  def loss_with_state_and_aux(
      self, params: Params, state: ModelState, key: PRNGKey,
      data: Any) -> Tuple[jnp.ndarray, ModelState, Mapping[str, jnp.ndarray]]:
    loss, state = self.loss_with_state(params, state, key, data)
    return loss, state, {}

  def normalizer(self, loss):
    num_classes = self.datasets.extra_info["num_classes"]
    maxval = 1.5 * onp.log(num_classes)
    loss = jnp.clip(loss, 0, maxval)
    return jnp.nan_to_num(loss, nan=maxval, posinf=maxval, neginf=maxval)


@gin.configurable
def ImageMLP_Cifar10_128x128x128_LayerNorm_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskNorm(datasets, [128, 128, 128], norm_type="layer_norm")


@gin.configurable
def ImageMLP_Cifar10_128x128x128_LayerNorm_Tanh():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskNorm(
      datasets, [128, 128, 128], norm_type="layer_norm", act_fn=jnp.tanh)


@gin.configurable
def ImageMLP_Cifar10_128x128x128_BatchNorm_Relu():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskNorm(datasets, [128, 128, 128], norm_type="batch_norm")


@gin.configurable
def ImageMLP_Cifar10_128x128x128_BatchNorm_Tanh():
  datasets = image.cifar10_datasets(batch_size=128)
  return _MLPImageTaskNorm(
      datasets, [128, 128, 128], norm_type="batch_norm", act_fn=jnp.tanh)
