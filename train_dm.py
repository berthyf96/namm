"""Train a score model with `score_sde` library.
Please see https://github.com/yang-song/score_sde/blob/main/run_lib.py
for the official training implementation.
"""

import logging
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
import time

from absl import app
from absl import flags
import flax
import jax
from ml_collections.config_flags import config_flags
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')  # use CPU-only

import vis
from losses import get_constraint_losses_fn
from score_flow import datasets
from score_flow import losses
from score_flow import sampling
from score_flow import utils
from score_flow.models import utils as mutils
from score_flow.models import ddpm, ncsnpp, ncsnv2  # pylint: disable=unused-import, g-multiple-import 


_CONFIG = config_flags.DEFINE_config_file('config', None, 'Score-model config.')
_NAMM_CONFIG = config_flags.DEFINE_config_file(
  'namm_config', None,
  'NAMM config (only used to plot constraint satisfaction).')
_WORKDIR = flags.DEFINE_string(
    'workdir', 'score_checkpoints/', 'Base working directory.')


def get_datasets_and_scalers():
  """Get train and eval datasets and data scaler and inverse scaler."""
  config = _CONFIG.value
  train_ds, eval_ds, _ = datasets.get_dataset(
      config,
      additional_dim=None,
      uniform_dequantization=config.data.uniform_dequantization)
  # `scaler` assumes images are originally [0, 1] and scales to
  # [0, 1] or [-1, 1].
  scaler = datasets.get_data_scaler(config)
  # `inverse_scaler` rescales to images that are [0, 1].
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  return (train_ds, eval_ds), (scaler, inverse_scaler)


def initialize_training_state():
  config = _CONFIG.value

  # Initialize model.
  rng = jax.random.PRNGKey(config.seed)
  rng, step_rng = jax.random.split(rng)
  score_model, init_model_state, init_params = mutils.init_model(step_rng, config)

  # Initialize optimizer.
  tx = losses.get_optimizer(config)
  opt_state = tx.init(init_params)

  # Construct initial state.
  state = mutils.State(
      step=0,
      epoch=0,
      model_state=init_model_state,
      opt_state=opt_state,
      ema_rate=config.model.ema_rate,
      params=init_params,
      params_ema=init_params,
      rng=rng)
  return score_model, state, tx


def main(_):
  config = _CONFIG.value
  namm_config = _NAMM_CONFIG.value
  # Copy certain values from LMM config.
  config.constraint = namm_config.constraint
  config.data.num_kolmogorov_states = namm_config.data.num_kolmogorov_states
  config.data.num_kolmogorov_states_per_row = namm_config.data.num_kolmogorov_states_per_row
  namm_config.data.height = config.data.height
  namm_config.data.width = config.data.width
  namm_config.data.num_channels = config.data.num_channels

  # Create workdir for this experiment.
  workdir = os.path.join(
    _WORKDIR.value,
    f'{config.data.dataset}_{config.data.height}x{config.data.width}_{config.model.name}' +
    f'_nf={config.model.nf}_{config.training.sde}' +
    f'_betamin={config.model.beta_min}_betamax={config.model.beta_max}'
  )

  # Create working directory and its subdirectories.
  ckpt_dir = os.path.join(workdir, 'checkpoints')
  progress_dir = os.path.join(workdir, 'progress')
  tf.io.gfile.makedirs(ckpt_dir)
  tf.io.gfile.makedirs(progress_dir)

  if utils.is_coordinator():
    logging.info(
      '# devices: %d, # local devices: %d',
      jax.device_count(), jax.local_device_count())
    # Save config.
    with tf.io.gfile.GFile(os.path.join(workdir, 'config.txt'), 'w') as f:
      f.write(str(config))

  # Create checkpoint manager.
  ckpt_mgr = ocp.CheckpointManager(ckpt_dir)

  # Get data.
  (train_ds, eval_ds), (_, inverse_scaler) = get_datasets_and_scalers()

  # Initialize model and training state.
  score_model, state, tx = initialize_training_state()

  # Load checkpoint.
  latest_epoch = ckpt_mgr.latest_step()
  if latest_epoch is not None:
    state = ckpt_mgr.restore(latest_epoch, args=ocp.args.StandardRestore(state))
    logging.info('Loaded checkpoint from epoch %d', latest_epoch)
  logging.info('Starting training at epoch %d (step %d)', state.epoch, state.step)
  if os.path.exists(os.path.join(progress_dir, 'losses_score.npy')):
    epoch_times = list(np.load(os.path.join(progress_dir, 'epoch_times.npy')))
    losses_score = list(np.load(os.path.join(progress_dir, 'losses_score.npy')))
    losses_val = list(np.load(os.path.join(progress_dir, 'losses_score_val.npy')))
  else:
    epoch_times = []
    losses_score, losses_val = [], []

  # Get SDE.
  sde, t0_eps = utils.get_sde(config)

  # Build training and eval functions.
  optimize_fn = losses.optimization_manager(config)
  train_step_fn = losses.get_step_fn(
      sde,
      score_model,
      optimizer=tx,
      train=True,
      optimize_fn=optimize_fn,
      reduce_mean=config.training.reduce_mean,
      continuous=config.training.continuous,
      likelihood_weighting=config.training.likelihood_weighting)
  eval_step_fn = losses.get_step_fn(
      sde,
      score_model,
      optimizer=tx,
      train=False,
      optimize_fn=optimize_fn,
      reduce_mean=config.training.reduce_mean,
      continuous=config.training.continuous,
      likelihood_weighting=config.training.likelihood_weighting)

  # Build sampling function.
  sampling_shape = (
      int(config.training.batch_size // jax.device_count()),
      config.data.height, config.data.width,
      config.data.num_channels)
  sampling_fn = sampling.get_sampling_fn(
      config, sde, score_model, sampling_shape, inverse_scaler, t0_eps)

  # Pmap and JIT multiple training/eval steps together for faster running.
  p_train_step = jax.pmap(
    train_step_fn, axis_name='batch', donate_argnums=1)
  p_eval_step = jax.pmap(
    eval_step_fn, axis_name='batch', donate_argnums=1)

  # Replicate training state to run on multiple devices.
  pstate = flax.jax_utils.replicate(state)

  # Get function for plotting training progress.
  plotter = vis.get_dm_progress_plotter(config, namm_config)

  # Check data constraint.
  image_shape = (config.data.height, config.data.width, config.data.num_channels)
  batch = next(iter(train_ds))['image']._numpy().reshape(-1, *image_shape)
  constraint_losses_fn = get_constraint_losses_fn(namm_config)
  constraint_losses = constraint_losses_fn(batch)
  logging.info('Constraint losses: %s', constraint_losses)
  if namm_config.constraint.type != 'count':
    assert(np.allclose(constraint_losses, np.zeros_like(constraint_losses), atol=1e-2))

  # Create different random states for different processes in a
  # multi-host environment (e.g., TPU pods).
  rng = jax.random.fold_in(state.rng, jax.process_index())

  for epoch in range(state.epoch, config.training.n_epochs):
    # Training.
    epoch_losses = []
    epoch_time = 0
    for step, item in enumerate(train_ds):
      s = time.perf_counter()

      batch = item['image']._numpy()

      rng, step_rngs = utils.psplit(rng)
      (_, pstate), ploss = p_train_step((step_rngs, pstate), batch)

      loss = flax.jax_utils.unreplicate(ploss).mean()

      t = time.perf_counter() - s
      epoch_time += t

      epoch_losses.append(loss)

      if ((step + 1) % config.training.log_freq == 0) and utils.is_coordinator():
        logging.info('[epoch %03d, step %03d] %.3f sec; training loss: %.5e',
                     epoch, step + 1, t, loss)

    # Update training curve.
    epoch_times.append(epoch_time)
    losses_score.append(np.mean(epoch_losses))

    # Validataion.
    epoch_val_losses = []
    for step, item in enumerate(eval_ds):
      s = time.perf_counter()

      val_batch = item['image']._numpy()

      rng, next_rngs = utils.psplit(rng)
      (_, _), peval_loss = p_eval_step((next_rngs, pstate), val_batch)

      eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
      epoch_val_losses.append(eval_loss)

      if ((step + 1) % config.training.log_freq == 0) and utils.is_coordinator():
        t = time.perf_counter() - s
        logging.info('[epoch %03d, step %03d] %.3f sec; val loss: %.5e', epoch, step + 1, t, eval_loss)

    # Update validation curve.
    losses_val.append(np.mean(epoch_val_losses))

    # Save progress snapshot.
    if ((epoch + 1) % config.training.snapshot_epoch_freq == 0
        and utils.is_coordinator()):
      state = flax.jax_utils.unreplicate(pstate)
      np.save(os.path.join(progress_dir, 'epoch_times.npy'), epoch_times)
      np.save(os.path.join(progress_dir, 'losses_score.npy'), losses_score)
      np.save(os.path.join(progress_dir, 'losses_score_val.npy'), losses_val)
    
      # Get samples.
      rng, sample_rngs = utils.psplit(rng)
      samples, _ = sampling_fn(sample_rngs, pstate)
  
      # Save progress.
      fig = plotter(losses_score, losses_val, samples)
      fig.savefig(os.path.join(progress_dir, f'progress_{epoch + 1:03d}.png'))

    # Save checkpoint.
    if ((epoch + 1) % config.training.ckpt_epoch_freq == 0
        and utils.is_coordinator()):
      # Save model checkpoint.
      state = flax.jax_utils.unreplicate(pstate)
      state = state.replace(rng=rng, epoch=epoch + 1)
      ckpt_mgr.save(epoch + 1, args=ocp.args.StandardSave(state))

  ckpt_mgr.wait_until_finished()


if __name__ == '__main__':
  app.run(main)