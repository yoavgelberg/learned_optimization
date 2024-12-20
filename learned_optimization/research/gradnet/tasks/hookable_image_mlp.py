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
from learned_optimization.research.gradnet.hookable_mlp import HookableMLP
import numpy as onp

Params = Any
ModelState = Any
PRNGKey = jnp.ndarray


class _HookableMLPImageTask(base.Task):
    """Hookable MLP based image task."""

    def __init__(self, datasets, hidden_sizes, act_fn=jax.nn.relu):
        super().__init__()
        num_classes = datasets.extra_info["num_classes"]
        sizes = list(hidden_sizes) + [num_classes]
        self.sizes = sizes
        self.datasets = datasets

        def _forward_with_bs(inp, *bs):
            inp = jnp.reshape(inp, [inp.shape[0], -1])
            return HookableMLP(hidden_sizes, output_dim=num_classes, activation=act_fn)(
                inp, *bs
            )

        def _forward(inp):
            return _forward_with_bs(inp, *[jnp.zeros([size]) for size in self.sizes])

        self._mod = hk.transform(_forward)
        self._mod_with_bs = hk.transform(_forward_with_bs)

    def init(self, key: PRNGKey) -> Any:
        batch = jax.tree_util.tree_map(
            lambda x: jnp.ones(x.shape, x.dtype), self.datasets.abstract_batch
        )
        return self._mod.init(key, batch["image"])

    def loss_with_bs(
        self, params: Params, key: PRNGKey, data: Any, *bs: List[jnp.ndarray]
    ) -> jnp.ndarray:  # pytype: disable=signature-mismatch  # jax-ndarray
        # Get number of classes
        num_classes = self.datasets.extra_info["num_classes"]
        # Compute logits
        logits = self._mod_with_bs.apply(params, key, data["image"], *bs)
        # Compute labels
        labels = jax.nn.one_hot(data["label"], num_classes)
        # Compute loss
        vec_loss = base.softmax_cross_entropy(logits=logits, labels=labels)
        return jnp.mean(vec_loss)

    def loss(
        self, params: Params, key: PRNGKey, data: Any
    ) -> jnp.ndarray:  # pytype: disable=signature-mismatch  # jax-ndarray
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
def HookableImageMLP_Cifar10BW8_Relu32():
    """A 1 hidden layer, 32 unit MLP for 8x8 black and white cifar10."""
    datasets = image.cifar10_datasets(
        batch_size=128, image_size=(8, 8), convert_to_black_and_white=True
    )
    return _HookableMLPImageTask(datasets, [32])


@gin.configurable
def HookableImageMLP_FashionMnist_Relu128x128():
    """A 2 hidden layer, 128 hidden unit MLP designed for fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [128, 128])


@gin.configurable
def HookableImageMLP_FashionMnist8_Relu32():
    """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(8, 8))
    return _HookableMLPImageTask(datasets, [32])


@gin.configurable
def HookableImageMLP_FashionMnist16_Relu32():
    """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(16, 16))
    return _HookableMLPImageTask(datasets, [32])


@gin.configurable
def ImageMLP_FashionMnist32_Relu32():
    """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 fashion mnist."""
    datasets = image.fashion_mnist_datasets(batch_size=128, image_size=(32, 32))
    return _HookableMLPImageTask(datasets, [32])


@gin.configurable
def HookableImageMLP_Cifar10_8_Relu32():
    """A 1 hidden layer, 32 hidden unit MLP designed for 8x8 cifar10."""
    datasets = image.cifar10_datasets(batch_size=128, image_size=(8, 8))
    return _HookableMLPImageTask(datasets, [32])


@gin.configurable
def HookableImageMLP_Imagenet16_Relu256x256x256():
    """A 3 hidden layer MLP trained on 16x16 resized imagenet."""
    datasets = image.imagenet16_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [256, 256, 256])


@gin.configurable
def HookableImageMLP_Cifar10_128x128x128_Relu():
    datasets = image.cifar10_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def HookableImageMLP_Cifar100_128x128x128_Relu():
    datasets = image.cifar100_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def HookableImageMLP_Cifar10_128x128x128_Tanh_bs64():
    datasets = image.cifar10_datasets(batch_size=64)
    return _HookableMLPImageTask(datasets, [128, 128, 128], act_fn=jnp.tanh)


@gin.configurable
def HookableImageMLP_Cifar10_128x128x128_Tanh_bs128():
    datasets = image.cifar10_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [128, 128, 128], act_fn=jnp.tanh)


@gin.configurable
def HookableImageMLP_Cifar10_128x128x128_Tanh_bs256():
    datasets = image.cifar10_datasets(batch_size=256)
    return _HookableMLPImageTask(datasets, [128, 128, 128], act_fn=jnp.tanh)


@gin.configurable
def HookableImageMLP_Mnist_128x128x128_Relu():
    datasets = image.mnist_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [128, 128, 128])


@gin.configurable
def HookableImageMLP_Cifar10_256x256_Relu():
    datasets = image.cifar10_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [256, 256])


@gin.configurable
def HookableImageMLP_Cifar10_256x256_Relu_BS32():
    datasets = image.cifar10_datasets(batch_size=32)
    return _HookableMLPImageTask(datasets, [256, 256])


@gin.configurable
def HookableImageMLP_Cifar10_1024x1024_Relu():
    datasets = image.cifar10_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [1024, 1024])


@gin.configurable
def HookableImageMLP_Cifar10_4096x4096_Relu():
    datasets = image.cifar10_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [4096, 4096])


@gin.configurable
def HookableImageMLP_Cifar10_8192x8192_Relu():
    datasets = image.cifar10_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [8192, 8192])


@gin.configurable
def HookableImageMLP_Cifar10_16384x16384_Relu():
    datasets = image.cifar10_datasets(batch_size=128)
    return _HookableMLPImageTask(datasets, [16384, 16384])
