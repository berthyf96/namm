"""Generate a dataset of MDM samples."""
import datetime
from functools import partial
import logging
import os
import pickle
import time

from absl import app
from absl import flags
import jax
from ml_collections.config_flags import config_flags
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')  # use CPU-only

from pdes import make_trajectory_from_image
from score_flow import datasets
from score_flow import losses
from score_flow import sampling
from score_flow import utils
from score_flow.models import utils as score_mutils
from score_flow.models import ddpm, ncsnpp, ncsnv2  # pylint: disable=unused-import, g-multiple-import 


_CONFIG = config_flags.DEFINE_config_file(
  'config', None, 'Score-model config.')
_WORKDIR = flags.DEFINE_string(
  'workdir', None, 'Main working directory.')
_MDM_CKPT = flags.DEFINE_integer(
  'mdm_ckpt', None,
  'Epoch of the MDM checkpoint to use (e.g., 100). If None, then take the latest checkpoint under WORKDIR/mdm/checkpoints.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'Sampling batch size. This will also be the number of samples per shard.')
_N_TRAIN = flags.DEFINE_integer('n_train', 12800, 'Number of training samples.')
_N_TEST = flags.DEFINE_integer('n_test', 0, 'Number of test samples.')
_N_VAL = flags.DEFINE_integer('n_valid', 512, 'Number of validation samples.')
_TFDS_DIR = flags.DEFINE_string('tfds_dir', './data', 'Directory to store TFDS datasets.')
_DATASET_NAME = flags.DEFINE_string('dataset_name', None, 'Name of the dataset.')
_CONSTRAINT = flags.DEFINE_string('constraint', None, 'Constraint.')


def get_tf_example_fn():
  if _CONSTRAINT.value in ['kolmogorov', 'incompress']:
    def tf_example(volume):
      shape = volume.shape
      features = tf.train.Features(
        feature={
          'image': tf.train.Feature(float_list=tf.train.FloatList(value=list(volume.reshape(-1)))),
          'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0], shape[1], shape[2], shape[3]])),
        }
      )
      example = tf.train.Example(features=features)
      return example
  
  else:
    def tf_example(image):
      shape = image.shape
      features = tf.train.Features(
        feature={
          'image': tf.train.Feature(float_list=tf.train.FloatList(value=list(image.reshape(-1)))),
          'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0], shape[1], shape[2]])),
        }
      )
      example = tf.train.Example(features=features)
      return example
  
  return tf_example


def get_image_shape():
  if _CONSTRAINT.value in ['kolmogorov', 'incompress']:
    return (128, 256, 2)
  elif _CONSTRAINT.value == 'count':
    return (128, 128, 1)
  elif _CONSTRAINT.value == 'flux':
    return (64, 64, 1)
  elif _CONSTRAINT.value == 'burgers':
    return (64, 64, 1)
  elif _CONSTRAINT.value == 'periodic':
    return (64, 64, 1)


def write_data(split, tf_data_dir, psampling_fn):
  dataset_name = _DATASET_NAME.value
  n_per_shard = _BATCH_SIZE.value
  tf_example = get_tf_example_fn()

  image_shape = get_image_shape()

  if split == 'train':
    n_images = _N_TRAIN.value
    rng = jax.random.PRNGKey(0)
  elif split == 'test':
    n_images = _N_TEST.value
    rng = jax.random.PRNGKey(1)
  elif split == 'val':
    n_images = _N_VAL.value
    rng = jax.random.PRNGKey(2)
  
  if n_images == 0:
    return 0, 0

  n_shards = int(n_images / n_per_shard) + (1 if n_images % n_per_shard != 0 else 0)
  logging.info('%s # total shards (%d per shard): %d', split, n_per_shard, n_shards)
  
  total_sampling_time = 0  # time spent just on sampling
  total_shard_time = 0  # time spent on sampling and writing to TFRecords
  for shard in range(n_shards):

    # Write to TFRecord.
    tfrecord_path = os.path.join(
      tf_data_dir,
      f'{dataset_name}-{split}.tfrecord-{shard:05d}-of-{n_shards:05d}')
    if os.path.exists(tfrecord_path):
      continue

    s = time.perf_counter()

    # Sample batch.
    rng, step_rngs = utils.psplit(rng)
    samples, _ = psampling_fn(step_rngs)
    samples = samples.reshape(-1, *image_shape)

    total_sampling_time += time.perf_counter() - s

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
      for samp in samples:
        if _CONSTRAINT.value in ['kolmogorov', 'incompress']:
          # Reshape from (n_rows * h, n_per_row * w, 2) to (nt, h, w, 2).
          vx = samp[:, :, :1]
          vy = samp[:, :, 1:2]
          vx = make_trajectory_from_image(vx, n_per_row=4)  # (h, w, nt)
          vy = make_trajectory_from_image(vy, n_per_row=4)
          vx = np.transpose(vx, (2, 0, 1))  # (nt, h, w)
          vy = np.transpose(vy, (2, 0, 1))

          volume = np.stack((vx, vy), axis=-1)  # (nt, h, w, 2)
          example = tf_example(np.ascontiguousarray(volume))
        else:
          example = tf_example(samp)
        writer.write(example.SerializeToString())

    shard_time = time.perf_counter() - s
    total_shard_time += shard_time

    logging.info('Wrote shard %d/%d in %.1f sec', shard + 1, n_shards, shard_time)

  return total_shard_time, total_sampling_time


def main(_):
  s = time.time()
  score_config = _CONFIG.value
  batch_size = _BATCH_SIZE.value
  n_train = _N_TRAIN.value
  n_test = _N_TEST.value
  n_val = _N_VAL.value
  assert n_train % batch_size == 0 and n_test % batch_size == 0 and n_val % batch_size == 0
  image_shape = get_image_shape()

  logging.info('image_shape: %s', image_shape)

  # Set score-model parameters.
  score_config.data.height = image_shape[0]
  score_config.data.width = image_shape[1]
  score_config.data.num_channels = image_shape[2]

  # Initialize score model and training state.
  rng = jax.random.PRNGKey(score_config.seed)
  rng, step_rng = jax.random.split(rng)
  score_model, score_init_model_state, score_init_params = score_mutils.init_model(
    step_rng, score_config)
  score_tx = losses.get_optimizer(score_config)
  score_opt_state = score_tx.init(score_init_params)
  score_state = score_mutils.State(
    step=0,
    epoch=0,
    model_state=score_init_model_state,
    opt_state=score_opt_state,
    ema_rate=score_config.model.ema_rate,
    params=score_init_params,
    params_ema=score_init_params,
    rng=rng)
  
  # Load checkpoint.
  ckpt_path = os.path.join(_WORKDIR.value, 'mdm/checkpoints')
  ckpt_mgr = ocp.CheckpointManager(ckpt_path)
  if _MDM_CKPT.value:
    mdm_epoch = _MDM_CKPT.value
  else:
    mdm_epoch = ckpt_mgr.latest_step()
  score_state = ckpt_mgr.restore(mdm_epoch, args=ocp.args.StandardRestore(score_state))
  logging.info('Using MDM epoch %d (step %d)', score_state.epoch, score_state.step)
  
  # Get SDE.
  sde, t0_eps = utils.get_sde(score_config)

  # Build sampling function.
  image_shape = (score_config.data.height, score_config.data.width, score_config.data.num_channels)
  input_shape = (batch_size // jax.local_device_count(), *image_shape)
  inverse_scaler = datasets.get_data_inverse_scaler(score_config)

  sampling_fn = sampling.get_sampling_fn_without_pmap(
    score_config, sde, score_model, input_shape, inverse_scaler, t0_eps)
  sampling_fn = partial(sampling_fn, state=score_state)
  psampling_fn = jax.pmap(sampling_fn)

  # Create dataset.
  tf_data_dir = os.path.join(_TFDS_DIR.value, _DATASET_NAME.value)
  os.makedirs(tf_data_dir, exist_ok=True)
  logging.info('Writing dataset at %s', tf_data_dir)
  test_shard_time, test_sampling_time = write_data('test', tf_data_dir, psampling_fn)
  val_shard_time, val_sampling_time = write_data('val', tf_data_dir, psampling_fn)
  train_shard_time, train_sampling_time = write_data('train', tf_data_dir, psampling_fn)

  # Write elapsed time.
  if utils.is_coordinator():
    os.makedirs(os.path.join(_WORKDIR.value, 'finetune'), exist_ok=True)
  timing_dict = {
    'train': train_shard_time,
    'train_sampling': train_sampling_time,
    'val': val_shard_time,
    'val_sampling': val_sampling_time,
    'test': test_shard_time,
    'test_sampling': test_sampling_time,
  }
  with open(os.path.join(_WORKDIR.value, 'finetune/gen_mdm_data_total_time.pkl'), 'wb') as f:
    pickle.dump(timing_dict, f)

  # Total elapsed time.
  elapsed_time_str = str(datetime.timedelta(seconds=time.time() - s))
  logging.info('Done! Elapsed time: %s', elapsed_time_str)


if __name__ == '__main__':
  app.run(main)