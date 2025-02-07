"""Train score model in learned mirror space."""

import logging
import os
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

import losses
import model_utils as namm_mutils
import vis
from score_flow import datasets
from score_flow import losses as score_losses
from score_flow import sampling
from score_flow import utils
from score_flow.models import utils as score_mutils
from score_flow.models import ddpm, ncsnpp, ncsnv2  # pylint: disable=unused-import, g-multiple-import 


_SCORE_CONFIG = config_flags.DEFINE_config_file(
  'score_config', None, 'Score-model config.')
_NAMM_CONFIG = config_flags.DEFINE_config_file(
  'namm_config', None, 'NAMM config.')
_WORKDIR = flags.DEFINE_string(
  'workdir', None,
  'Base working directory that includes namm/ subdirectory.')
_NAMM_CKPT = flags.DEFINE_string(
  'namm_ckpt', None,
  'Base name of NAMM checkpoint path in namm/checkpoints. '
  'If None, the latest checkpoint is used.')


def main(_):
  score_config = _SCORE_CONFIG.value
  namm_config = _NAMM_CONFIG.value

  # Copy certain values from NAMM config to score-model config.
  score_config.constraint = namm_config.constraint
  score_config.data.num_kolmogorov_states = namm_config.data.num_kolmogorov_states
  score_config.data.num_kolmogorov_states_per_row = namm_config.data.num_kolmogorov_states_per_row

  # Copy certain values from score-model config to NAMM config.
  namm_config.data.height = score_config.data.height
  namm_config.data.width = score_config.data.width
  namm_config.data.num_channels = score_config.data.num_channels

  workdir = os.path.join(_WORKDIR.value, 'mdm')
  namm_ckpt_path = os.path.join(_WORKDIR.value, 'namm', 'checkpoints')
  if _NAMM_CKPT.value is not None:
    namm_ckpt_path = os.path.join(namm_ckpt_path, _NAMM_CKPT.value)

  logging.info('workdir: %s', workdir)

  # Create working directory and its subdirectories.
  ckpt_dir = os.path.join(workdir, 'checkpoints')
  progress_dir = os.path.join(workdir, 'progress')
  tf.io.gfile.makedirs(ckpt_dir)
  tf.io.gfile.makedirs(progress_dir)

  if utils.is_coordinator():
    # Save config.
    with tf.io.gfile.GFile(os.path.join(workdir, 'config.txt'), 'w') as f:
      f.write(str(score_config))

  # Create checkpoint manager.
  ckpt_mgr = ocp.CheckpointManager(ckpt_dir)

  # Get data.
  score_config.constraint = namm_config.constraint
  train_ds, eval_ds, _ = datasets.get_dataset(
    score_config, additional_dim=None,
    uniform_dequantization=score_config.data.uniform_dequantization)

  # Get NAMM model and checkpoint.
  namm_model = namm_mutils.get_model(namm_config)
  namm_state = namm_mutils.init_state(namm_config, namm_model)
  namm_state, _, _ = namm_mutils.init_optimizer(namm_config, namm_state)
  namm_ckpt_mgr = ocp.CheckpointManager(namm_ckpt_path)
  namm_epoch = namm_ckpt_mgr.latest_step()
  if namm_epoch is not None:
    namm_state = namm_ckpt_mgr.restore(namm_epoch, args=ocp.args.StandardRestore(namm_state))
    logging.info('Loaded NAMM checkpoint from epoch %d', namm_epoch)
  else:
    raise RuntimeError('Pretrained NAMM checkpoint not found')

  # Initialize score model and training state.
  rng = jax.random.PRNGKey(score_config.seed)
  rng, step_rng = jax.random.split(rng)
  score_model, score_init_model_state, score_init_params = score_mutils.init_model(
    step_rng, score_config)
  score_tx = score_losses.get_optimizer(score_config)
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

  # Get SDE.
  sde, t0_eps = utils.get_sde(score_config)

  # Build sampling function.
  image_shape = (score_config.data.height, score_config.data.width, score_config.data.num_channels)
  input_shape = (score_config.training.batch_size // jax.local_device_count(), *image_shape)
  inverse_scaler = datasets.get_data_inverse_scaler(score_config)
  sde_sampling_fn = sampling.get_sampling_fn_without_pmap(
      score_config, sde, score_model, input_shape, inverse_scaler, t0_eps)
  
  # Build training function.
  score_optimize_fn = score_losses.optimization_manager(score_config)
  score_step_fn = losses.get_score_step_fn(
    sde, score_model, score_tx, namm_model, namm_state,
    train=True,
    optimize_fn=score_optimize_fn,
    reduce_mean=score_config.training.reduce_mean,
    continuous=score_config.training.continuous,
    likelihood_weighting=score_config.training.likelihood_weighting)
  score_pstep_fn = jax.pmap(score_step_fn, axis_name='batch', donate_argnums=1)

  # Get eval function.
  score_eval_fn = losses.get_score_step_fn(
    sde, score_model, score_tx, namm_model, namm_state,
    train=False,
    optimize_fn=score_optimize_fn,
    reduce_mean=score_config.training.reduce_mean,
    continuous=score_config.training.continuous,
    likelihood_weighting=score_config.training.likelihood_weighting)
  score_peval_fn = jax.pmap(score_eval_fn, axis_name='batch', donate_argnums=1)

  # Get sampling function.
  mdm_sampling_fn = losses.get_mdm_sampling_fn(
    namm_model,
    sde_sampling_fn,
    apply_inverse=True)
  mdm_psampling_fn = jax.pmap(mdm_sampling_fn)

  # Load checkpoint.
  if len(os.listdir(ckpt_dir)) > 0:
    latest_epoch = ckpt_mgr.latest_step()
    score_state = ckpt_mgr.restore(
      latest_epoch, args=ocp.args.StandardRestore(score_state))
    logging.info('Loaded checkpoint from epoch %d', latest_epoch)

  logging.info('Starting training at epoch %d (step %d)', score_state.epoch, score_state.step)
  if os.path.exists(os.path.join(progress_dir, 'losses_score.npy')):
    epoch_times = list(np.load(os.path.join(progress_dir, 'epoch_times.npy')))
    losses_score = list(np.load(os.path.join(progress_dir, 'losses_score.npy')))
    losses_val = list(np.load(os.path.join(progress_dir, 'losses_score_val.npy')))
  else:
    epoch_times = []
    losses_score, losses_val = [], []

  # Replicate training state to run on multiple devices.
  score_pstate = flax.jax_utils.replicate(score_state)
  namm_pstate = flax.jax_utils.replicate(namm_state)

  # Get function for plotting training progress.
  plotter = vis.get_mdm_progress_plotter(score_config, namm_config)

  # Check data constraint.
  image_shape = (score_config.data.height, score_config.data.width, score_config.data.num_channels)
  batch = next(iter(train_ds))['image']._numpy().reshape(-1, *image_shape)
  constraint_losses_fn = losses.get_constraint_losses_fn(namm_config)
  constraint_losses = constraint_losses_fn(batch)
  logging.info('Constraint losses: %s', constraint_losses)
  if namm_config.constraint.type != 'count':
    assert(np.allclose(constraint_losses, np.zeros_like(constraint_losses), atol=1e-2))

  # Create different random states for different processes in a
  # multi-host environment (e.g., TPU pods).
  rng = jax.random.fold_in(score_state.rng, jax.process_index())

  for epoch in range(score_state.epoch, score_config.training.n_epochs):
    # Training.
    epoch_losses = []
    epoch_time = 0
    for step, item in enumerate(train_ds):
      s = time.perf_counter()

      batch = item['image']._numpy()

      rng, step_rngs = utils.psplit(rng)
      (_, score_pstate), ploss, x_fwd = score_pstep_fn(
        (step_rngs, score_pstate), batch)
      
      loss = flax.jax_utils.unreplicate(ploss).mean()

      t = time.perf_counter() - s
      epoch_time += t

      epoch_losses.append(loss)

      if ((step + 1) % score_config.training.log_freq == 0) and utils.is_coordinator():
        logging.info('[epoch %03d, step %03d] %.3f sec; training loss: %.5e', epoch, step + 1, t, loss)

    # Update training curve.
    epoch_times.append(epoch_time)
    losses_score.append(np.mean(epoch_losses))

    # Validataion.
    epoch_val_losses = []
    for step, item in enumerate(eval_ds):
      s = time.perf_counter()

      val_batch = item['image']._numpy()

      rng, eval_rngs = utils.psplit(rng)
      peval_loss = score_peval_fn((eval_rngs, score_pstate), val_batch)

      eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
      epoch_val_losses.append(eval_loss)

      if ((step + 1) % score_config.training.log_freq == 0) and utils.is_coordinator():
        t = time.perf_counter() - s
        logging.info('[epoch %03d, step %03d] %.3f sec; val loss: %.5e', epoch, step + 1, t, eval_loss)

    # Update validation curve.
    losses_val.append(np.mean(epoch_val_losses))

    # Save progress snapshot.
    if ((epoch + 1) % score_config.training.snapshot_epoch_freq == 0
        and utils.is_coordinator()):
      score_state = flax.jax_utils.unreplicate(score_pstate)
      # Save stats.
      np.save(os.path.join(progress_dir, 'epoch_times.npy'), epoch_times)
      np.save(os.path.join(progress_dir, 'losses_score.npy'), losses_score)
      np.save(os.path.join(progress_dir, 'losses_score_val.npy'), losses_val)

      # Get samples from mirror diffusion model.
      rng, eval_rngs = utils.psplit(rng)
      y, y_bwd = mdm_psampling_fn(eval_rngs, score_pstate, namm_pstate)

      # Save plot of training progress.
      fig = plotter(losses_score, losses_val, batch, x_fwd, y, y_bwd)
      fig.savefig(os.path.join(progress_dir, f'progress_{epoch + 1:03d}.png'))

    # Save checkpoint.
    if ((epoch + 1) % score_config.training.ckpt_epoch_freq == 0
        and utils.is_coordinator()):
      score_state = flax.jax_utils.unreplicate(score_pstate)
      score_state = score_state.replace(epoch=epoch + 1)
      ckpt_mgr.save(epoch + 1, args=ocp.args.StandardSave(score_state))

  ckpt_mgr.wait_until_finished()


if __name__ == '__main__':
  app.run(main)