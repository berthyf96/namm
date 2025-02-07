# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
from typing import Any, Optional, Tuple
import os

import jax
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset_builder_and_resize_op(config: ml_collections.ConfigDict, shuffle_seed = None, is_mdm: bool = False) -> Tuple[Any, Any]:
  """Create dataset builder and image resizing function for dataset."""
  if config.data.dataset in ['RIAF', 'GRMHD', 'Periodic', 'Burgers', 'Galaxies']:
    if config.data.dataset == 'RIAF':
      image_dim = 100 * 100
    elif config.data.dataset == 'GRMHD':
      image_dim = 64 * 64 if is_mdm else 400 * 400
    elif config.data.dataset == 'Periodic':
      image_dim = config.data.height * config.data.width
    elif config.data.dataset == 'Burgers':
      image_dim = config.data.height * config.data.width
    elif config.data.dataset == 'Galaxies':
      assert config.data.height == config.data.width
      image_dim = config.data.height * config.data.width

    features_dict = {
      'image': tf.io.FixedLenFeature([image_dim], tf.float32),
      'shape': tf.io.FixedLenFeature([2], tf.int64),
    }
    if config.data.dataset == 'Periodic':
      features_dict['periods'] = tf.io.FixedLenFeature([1], tf.int64)
    if config.data.dataset == 'Galaxies':
      features_dict['nsrc'] = tf.io.FixedLenFeature([1], tf.int64)
    if is_mdm:
      features_dict['shape'] = tf.io.FixedLenFeature([3], tf.int64)

    def parse_example(serialized_example):
      parsed_example = tf.io.parse_single_example(serialized_example, features=features_dict)
      shape = parsed_example['shape']
      parsed_example['image'] = tf.reshape(parsed_example['image'], (shape[0], shape[1], 1))
      return parsed_example
    
    def ds_from_tfrecords(tfrecords_pattern):
      shard_files = tf.io.matching_files(tfrecords_pattern)
      shard_files = tf.random.shuffle(shard_files, seed=shuffle_seed)
      shards = tf.data.Dataset.from_tensor_slices(shard_files)
      ds = shards.interleave(tf.data.TFRecordDataset)
      ds = ds.map(
        map_func=parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return ds

    dataset_name = config.data.dataset.lower()
    dataset_builder = {
      'train': ds_from_tfrecords(
        os.path.join(config.data.tfds_dir,
                      f'{dataset_name}/{dataset_name}-train.tfrecord-*')),
      'val': ds_from_tfrecords(
        os.path.join(config.data.tfds_dir,
                      f'{dataset_name}/{dataset_name}-val.tfrecord-*')),
      'test': ds_from_tfrecords(
        os.path.join(config.data.tfds_dir,
                      f'{dataset_name}/{dataset_name}-test.tfrecord-*')),
    }

    def resize_op(img):
      if config.data.num_channels == 3:
        img = tf.image.grayscale_to_rgb(img)
      return tf.image.resize(
          img, [config.data.height, config.data.width],
          antialias=config.data.antialias)
      
  else:
    raise ValueError(
        f'Dataset {config.data.dataset} not supported.')
  return dataset_builder, resize_op


def get_preprocess_fn(config: ml_collections.ConfigDict,
                      resize_op: Any,
                      uniform_dequantization: bool = False,
                      evaluation: bool = False) -> Any:
  """Create preprocessing function for dataset."""

  if config.data.dataset == 'RIAF' or config.data.dataset == 'GRMHD':
    # These datasets' images can be randomly rotated and zoomed in/out.
    @tf.autograph.experimental.do_not_convert
    def preprocess_fn(d):
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if config.data.random_rotation and not evaluation:
        img = tf.keras.layers.RandomRotation(
          factor=(-1., 1.), fill_mode='constant', fill_value=0.)(img)
      if config.data.random_zoom and not evaluation:
        print('USING RANDOM ZOOM')
        # Assume that GRMHD and RIAF images have ring diameter 40 uas and
        # FOV 128 uas. We want ring diameters between 35 and 48 uas.
        img = tf.keras.layers.RandomZoom(
          height_factor=(-0.167, 0.145), fill_mode='constant', fill_value=0.)(img)
      if config.data.constant_flux:
        img *= config.data.total_flux / tf.reduce_sum(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=d.get('label', None))
  else:
    @tf.autograph.experimental.do_not_convert
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if config.data.constant_flux:
        img *= config.data.total_flux / tf.reduce_sum(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=d.get('label', None))

  return preprocess_fn


def get_dataset(
    config: ml_collections.ConfigDict,
    additional_dim: Optional[int] = None,
    uniform_dequantization: bool = False,
    evaluation: bool = False,
    shuffle_seed: Optional[int] = None,
    device_batch: bool = True,
    dataset_name: Optional[str] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
  """Create data loaders for training, validation, and testing.
  Most of the logic from `score_flow/datasets.py` is kept.
  NOTE: This code assumes that if `dataset_name` is not `None`, then the
  requested dataset is of MDM samples.

  Args:
    config: The config.
    additional_dim: If not `None`, add an additional dimension
      to the output data for jitted steps.
    uniform_dequantization: If `True`, uniformly dequantize the images.
      This is usually only used when evaluating log-likelihood [bits/dim]
      of the data.
    evaluation: If `True`, fix number of epochs to 1.
    shuffle_seed: Optional seed for shuffling dataset.
    device_batch: If `True`, divide batch size into device batch and
      local batch.
    dataset_name: Directly specified dataset name.
      If `None`, dataset name is determined from `config.data.dataset`.
      Otherwise, TFRecords have the filepath f'{tfds_dir}/{dataset_name}/{dataset_name}-{split}...'.
  Returns:
    train_ds, val_ds, test_ds.
  """
  is_mdm = dataset_name is not None

  # Compute batch size for this worker.
  batch_size = (
      config.training.batch_size if not evaluation else config.eval.batch_size)
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by '
                     f'the number of devices ({jax.device_count()})')

  per_device_batch_size = batch_size // jax.device_count()
  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  # Create additional data dimension when jitting multiple steps together
  if not device_batch:
    batch_dims = [batch_size]
  elif additional_dim is None:
    batch_dims = [jax.local_device_count(), per_device_batch_size]
  else:
    batch_dims = [
        jax.local_device_count(), additional_dim, per_device_batch_size
    ]
  
  if (config.data.dataset == 'Kolmogorov' or (is_mdm and 'kolmogorov' in dataset_name)):
    # Kolmogorov/Navier-Stokes datasets get special treatment.
    if config.data.height % 64 != 0 or config.data.width % 64 != 0:
      raise ValueError('Kolmogorov dataset requires height and width to be multiple of 64.')
    if config.data.kolmogorov_representation == 'volume' and config.data.num_channels != config.data.num_kolmogorov_states * 2:
      raise ValueError('For volume representation, num_channels must be 2 * num_kolmogorov_states.')
    if config.data.kolmogorov_representation == 'image' and config.data.num_channels != 2:
      raise ValueError('For image representation, num_channels must be 2.')
    
    if dataset_name is None and config.data.dataset == 'Kolmogorov':
      tf_data_dir = os.path.join(config.data.tfds_dir, 'kolmogorov')
    else:
      tf_data_dir = os.path.join(config.data.tfds_dir, dataset_name)
    # Data are stored as (nt, h, w, 2) volumes.
    nt = config.data.num_kolmogorov_states
    if dataset_name is None:
      image_dim = nt * 64 * 64 * 2
    else:
      image_dim = config.data.height * config.data.width * config.data.num_channels
    features_dict = {
        'image': tf.io.FixedLenFeature([image_dim], tf.float32),
        'shape': tf.io.FixedLenFeature([4], tf.int64)
    }
    def parse_example(serialized_example):
      parsed_example = tf.io.parse_single_example(serialized_example, features=features_dict)
      shape = parsed_example['shape']
      parsed_example['image'] = tf.reshape(parsed_example['image'], (shape[0], shape[1], shape[2], shape[3]))
      return parsed_example

    def trajectory_to_volume(trajectory):
      # Concatenate states in the channel dimension.
      return np.concatenate(tuple([state for state in trajectory]), axis=-1)
    
    n_per_row = config.data.num_kolmogorov_states_per_row
    n_rows = nt // n_per_row
    def trajectory_to_image(trajectory):
      # Stitch states together into one image of shape (h * n_rows, w * n_per_row, 2).
      # Assume trajectory is of shape (nt, h, w, 2).
      return np.concatenate(
        tuple(
          [np.concatenate(tuple([trajectory[ti] for ti in range(i * n_per_row, (i + 1) * n_per_row)]), axis=1) \
          for i in range(n_rows)]), axis=0)

    @tf.autograph.experimental.do_not_convert
    def preprocess_fn(d):
      img = d['image']
      # Reshape volume into 2D image.
      if config.data.kolmogorov_representation == 'volume':
        img = tf.numpy_function(trajectory_to_volume, [img], tf.float32)
      elif config.data.kolmogorov_representation == 'image':
        img = tf.numpy_function(trajectory_to_image, [img], tf.float32)
      return dict(image=img, label=d.get('label', None))
    
    shuffle_buffer_size = 1000
    def _get_ds(split):
      if dataset_name is None and config.data.dataset == 'Kolmogorov':
        pattern = os.path.join(tf_data_dir, f'kolmogorov-{split}.tfrecord-*')
      elif dataset_name is None and config.data.dataset == 'NavStokes':
        pattern = os.path.join(tf_data_dir, f'navstokes-{split}.tfrecord-*')
      else:
        pattern = os.path.join(tf_data_dir, f'{dataset_name}-{split}.tfrecord-*')
      shard_files = tf.io.matching_files(pattern)
      shard_files = tf.random.shuffle(shard_files)
      shards = tf.data.Dataset.from_tensor_slices(shard_files)
      ds = shards.interleave(tf.data.TFRecordDataset)
      # ds = ds.repeat(count=num_epochs)
      ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
      ds = ds.map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return ds
    train_ds = _get_ds('train')  # 10000
    test_ds = _get_ds('test')  # 100
    val_ds = _get_ds('val')  # 100
    return train_ds, val_ds, test_ds

  elif is_mdm:
    # MDM dataset.
    tf_data_dir = os.path.join(config.data.tfds_dir, dataset_name)
    image_dim = config.data.height * config.data.width * config.data.num_channels
    features_dict = {
        'image': tf.io.FixedLenFeature([image_dim], tf.float32),
        'shape': tf.io.FixedLenFeature([3], tf.int64)
    }
    def parse_example(serialized_example):
      parsed_example = tf.io.parse_single_example(serialized_example, features=features_dict)
      shape = parsed_example['shape']
      parsed_example['image'] = tf.reshape(parsed_example['image'], (shape[0], shape[1], shape[2]))
      return parsed_example

    @tf.autograph.experimental.do_not_convert
    def preprocess_fn(d):
      img = d['image']
      return dict(image=img, label=d.get('label', None))
    
    shuffle_buffer_size = 1000
    def _get_ds(split):
      pattern = os.path.join(tf_data_dir, f'{dataset_name}-{split}.tfrecord-*')
      shard_files = tf.io.matching_files(pattern)
      shard_files = tf.random.shuffle(shard_files)
      shards = tf.data.Dataset.from_tensor_slices(shard_files)
      ds = shards.interleave(tf.data.TFRecordDataset)
      # ds = ds.repeat(count=num_epochs)
      ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
      ds = ds.map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size, drop_remainder=True)
      ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return ds
    train_ds = _get_ds('train')
    test_ds = _get_ds('test')
    val_ds = _get_ds('val')
    return train_ds, val_ds, test_ds

  # Get dataset builder.
  dataset_builder, resize_op = get_dataset_builder_and_resize_op(config, shuffle_seed, is_mdm)

  # Get preprocessing function.
  preprocess_fn = get_preprocess_fn(
      config, resize_op, uniform_dequantization, evaluation)

  def create_dataset(dataset_builder: Any,
                     split: str,
                     take_val_from_train: bool = False,
                     train_split: float = 0.9):
    # Some datasets only include train and test sets, in which case we take
    # validation data from the training set.
    if split == 'test':
      take_val_from_train = False
    source_split = 'train' if take_val_from_train else split

    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.threading.private_threadpool_size = 48
    dataset_options.threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(
        options=dataset_options, shuffle_seed=shuffle_seed)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
          split=source_split, shuffle_files=True, read_config=read_config)
    elif config.data.dataset in [
        'RIAF', 'GRMHD', 'Periodic', 'Burgers', 'Galaxies'
    ]:
      ds = dataset_builder[source_split].with_options(dataset_options)
    else:
      ds = dataset_builder.with_options(dataset_options)

    if take_val_from_train:
      train_size = int(train_split * len(ds))
      # Take the first `train_split` pct. for training and the rest for val.
      ds = ds.take(train_size) if split == 'train' else ds.skip(train_size)

    # ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for batch_size in reversed(batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  # Set the correct split names.
  train_ds = create_dataset(dataset_builder, 'train')
  test_ds = create_dataset(dataset_builder, 'test')
  val_ds = create_dataset(dataset_builder, 'val')
  return train_ds, val_ds, test_ds
