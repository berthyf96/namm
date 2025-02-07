"""Train learned mirror map and its inverse."""
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
from score_flow import utils


_CONFIG = config_flags.DEFINE_config_file('config', None, 'NAMM config.')
_WORKDIR = flags.DEFINE_string(
  'workdir', './checkpoints', 'Base working directory.')
_FINETUNE = flags.DEFINE_bool('finetune', False, 'Whether to train with MDM finetuning.')
_MAX_SIGMA = flags.DEFINE_float('max_sigma', 0.1, 'Max sigma used for finetuning.')
_CONSTRAINT_WEIGHT = flags.DEFINE_float(
  'constraint_weight', None, 'Weight of the constraint loss during finetuning.')
_MDM_DATASET = flags.DEFINE_string(
  'mdm_dataset', None, "Name of the MDM dataset (e.g., 'mdm_kolmogorov').")


def get_workdir():
  config = _CONFIG.value

  # Descriptive directory name:
  basedir = (f"{config.optim.regularization.replace('_', '')}"
            f'_cyc={config.optim.cycle_weight}'
            f'_reg={config.optim.regularization_weight}'
            f'_constr={config.optim.constraint_weight}'
            f'_maxsig={config.optim.max_sigma}'
            f'_lr={config.optim.learning_rate:.0e}'
            f'_bs={config.training.batch_size}')
  if config.model.fwd_network ==  'icnn':
    basedir += f'_layers={config.model.fwd_icnn_n_layers}_fwdnfilt={config.model.fwd_icnn_n_filters}'
  elif config.model.fwd_network == 'resnet':
    basedir += f'_fwdnfilt={config.model.fwd_n_filters}'
  if config.model.bwd_network == 'resnet':
    basedir += f'_bwdnfilt={config.model.bwd_n_filters}'
  if config.optim.fixed_sigma:
    basedir += '_fixedsigma'
  if config.model.fwd_network == 'icnn' and not config.model.bwd_residual:
    basedir += '_nores'
  elif config.model.fwd_network == 'resnet' and not config.model.bwd_residual:
    basedir += '_bwdnores'
  elif config.model.fwd_network == 'resnet' and not config.model.fwd_residual:
    basedir += '_fwdnores'

  if _FINETUNE.value:
    expdir = basedir + '/finetune'
  else:
    expdir = basedir

  workdir = os.path.join(
    _WORKDIR.value, f'{config.data.dataset}_{config.constraint.type}',
    expdir, 'namm')

  orig_workdir = workdir.replace(expdir, basedir)

  return workdir, orig_workdir


def main(_):
  config = _CONFIG.value
  finetune = _FINETUNE.value

  # Create working directory and its subdirectories.
  workdir, orig_workdir = get_workdir()
  logging.info('Working directory: %s', workdir)
  ckpt_dir = os.path.join(workdir, 'checkpoints')
  progress_dir = os.path.join(workdir, 'progress')
  tf.io.gfile.makedirs(ckpt_dir)
  tf.io.gfile.makedirs(progress_dir)

  if utils.is_coordinator():
    # Save config.
    with tf.io.gfile.GFile(os.path.join(workdir, 'config.txt'), 'w') as f:
      f.write(str(config))

  # Create checkpoint manager.
  ckpt_mgr = ocp.CheckpointManager(ckpt_dir)

  # Get data.
  train_ds, val_ds, _ = datasets.get_dataset(
    config, additional_dim=None,
    uniform_dequantization=config.data.uniform_dequantization)

  if finetune:
    mdm_train_ds, mdm_val_ds, _ = datasets.get_dataset(
      config, additional_dim=None,
      uniform_dequantization=config.data.uniform_dequantization,
      dataset_name=_MDM_DATASET.value)
    train_ds = tf.data.Dataset.zip((train_ds, mdm_train_ds))
    val_ds = tf.data.Dataset.zip((val_ds, mdm_val_ds))

  if finetune:
    config.optim.max_sigma = _MAX_SIGMA.value
    config.optim.constraint_weight = _CONSTRAINT_WEIGHT.value

    model = namm_mutils.get_model(config)
    state = namm_mutils.init_state(config, model)
    state, fwd_tx, bwd_tx = namm_mutils.init_optimizer(config, state)

    # Load pretrained weights.
    pretrained_ckpt_mgr = ocp.CheckpointManager(os.path.join(orig_workdir, 'checkpoints'))
    latest_epoch = pretrained_ckpt_mgr.latest_step()
    if latest_epoch is not None:
      state = pretrained_ckpt_mgr.restore(latest_epoch, args=ocp.args.StandardRestore(state))
      logging.info('Loaded checkpoint from epoch %d', latest_epoch)
    else:
      raise RuntimeError('Pretrained NAMM checkpoint not found')
    state = state.replace(constraint_weight=_CONSTRAINT_WEIGHT.value)
  else:
    model = namm_mutils.get_model(config)
    state = namm_mutils.init_state(config, model)
    state, fwd_tx, bwd_tx = namm_mutils.init_optimizer(config, state)

  # Load checkpoint and loss curves. Loss curves keep track of loss per epoch.
  if len(os.listdir(ckpt_dir)) > 0:
    latest_epoch = ckpt_mgr.latest_step()
    if latest_epoch is not None:
      state = ckpt_mgr.restore(latest_epoch, args=ocp.args.StandardRestore(state))
      logging.info('Loaded checkpoint from epoch %d', latest_epoch)
  if os.path.exists(os.path.join(progress_dir, 'losses_total.npy')):
    epoch_times = list(np.load(os.path.join(progress_dir, 'epoch_times.npy')))
    losses_total = list(np.load(os.path.join(progress_dir, 'losses_total.npy')))
    losses_cycle = list(np.load(os.path.join(progress_dir, 'losses_cycle.npy')))
    losses_constraint = list(np.load(os.path.join(progress_dir, 'losses_constraint.npy')))
    losses_reg = list(np.load(os.path.join(progress_dir, 'losses_reg.npy')))
    val_losses_total = list(np.load(os.path.join(progress_dir, 'val_losses_total.npy')))
    val_losses_cycle = list(np.load(os.path.join(progress_dir, 'val_losses_cycle.npy')))
    val_losses_constraint = list(np.load(os.path.join(progress_dir, 'val_losses_constraint.npy')))
    val_losses_reg = list(np.load(os.path.join(progress_dir, 'val_losses_reg.npy')))
  else:
    epoch_times = []
    losses_total, losses_cycle, losses_constraint, losses_reg = [], [], [], []
    val_losses_total, val_losses_cycle, val_losses_constraint, val_losses_reg = [], [], [], []

  # Get training and eval functions.
  constraint_losses_fn = losses.get_constraint_losses_fn(config)
  step_fn = losses.get_namm_step_fn(
    model,
    fwd_tx,
    bwd_tx,
    constraint_losses_fn,
    regularization=config.optim.regularization,
    train=True,
    max_sigma=config.optim.max_sigma,
    fixed_sigma=config.optim.fixed_sigma,
    fwd_strong_convexity=config.model.fwd_strong_convexity,
    use_mdm_samples=finetune,
    update_fwd=not finetune)
  eval_fn = losses.get_namm_step_fn(
    model,
    fwd_tx,
    bwd_tx,
    constraint_losses_fn,
    regularization=config.optim.regularization,
    train=False,
    max_sigma=config.optim.max_sigma,
    fixed_sigma=config.optim.fixed_sigma,
    fwd_strong_convexity=config.model.fwd_strong_convexity,
    use_mdm_samples=finetune)
  pstep_fn = jax.pmap(jax.jit(step_fn), axis_name='batch', donate_argnums=(1, 2))
  peval_fn = jax.pmap(jax.jit(eval_fn), axis_name='batch', donate_argnums=(1, 2))

  # Get function for plotting training progress.
  plotter = vis.get_namm_progress_plotter(config)

  # Check data constraint.
  shape = (config.data.height, config.data.width, config.data.num_channels)
  if finetune:
    batch = next(iter(train_ds))[0]['image']._numpy().reshape(-1, *shape)
  else:
    batch = next(iter(train_ds))['image']._numpy().reshape(-1, *shape)
  constraint_losses = constraint_losses_fn(batch)
  logging.info('Constraint losses: %s', constraint_losses)
  if config.constraint.type != 'count':
    assert(np.allclose(constraint_losses, np.zeros_like(constraint_losses), atol=1e-2))

  # Replicate training state to run on multiple devices.
  pstate = flax.jax_utils.replicate(state)

  # Create different random states for different processes in a
  # multi-host environment (e.g., TPU pods).
  rng = jax.random.fold_in(state.rng, jax.process_index())

  if utils.is_coordinator():
    logging.info('Starting training at epoch %d (step %d)', state.epoch, state.step)

  for epoch in range(state.epoch, config.training.n_epochs):
    # Training.
    epoch_losses_total = []
    epoch_losses_cycle = []
    epoch_losses_constraint = []
    epoch_losses_reg = []
    step = 0

    epoch_time = 0
    for step, item in enumerate(train_ds):
      s = time.perf_counter()

      if finetune:
        batch = item[0]['image']._numpy()
        mdm_batch = item[1]['image']._numpy()
      else:
        batch = item['image']._numpy()
        mdm_batch = None

      rng, step_rngs = utils.psplit(rng)
      (pstate, ploss, ploss_cycle, ploss_constraint, ploss_reg,
      x_fwd, x_fwdbwd, y, y_bwd, stds) = pstep_fn((step_rngs, pstate), batch, mdm_batch)

      loss = flax.jax_utils.unreplicate(ploss).item()
      loss_cycle = flax.jax_utils.unreplicate(ploss_cycle).item()
      loss_constraint = flax.jax_utils.unreplicate(ploss_constraint).item()
      loss_reg = flax.jax_utils.unreplicate(ploss_reg).item()

      t = time.perf_counter() - s  # step time
      epoch_time += t

      epoch_losses_total.append(loss)
      epoch_losses_cycle.append(loss_cycle)
      epoch_losses_constraint.append(loss_constraint)
      epoch_losses_reg.append(loss_reg)

      if ((step + 1) % config.training.log_freq == 0) and utils.is_coordinator():
        logging.info(
          '[epoch %03d, step %03d] %.3f sec; training loss: %.5e',
          epoch, step + 1, t, loss)

    # Update training curves.
    epoch_times.append(epoch_time)
    losses_total.append(np.mean(epoch_losses_total))
    losses_cycle.append(np.mean(epoch_losses_cycle))
    losses_constraint.append(np.mean(epoch_losses_constraint))
    losses_reg.append(np.mean(epoch_losses_reg))

    # Validation.
    epoch_val_losses_total = []
    epoch_val_losses_cycle = []
    epoch_val_losses_constraint = []
    epoch_val_losses_reg = []
    for step, item in enumerate(val_ds):
      s = time.perf_counter()

      if finetune:
        val_batch = item[0]['image']._numpy()
        mdm_val_batch = item[1]['image']._numpy()
      else:
        val_batch = item['image']._numpy()
        mdm_val_batch = None

      rng, next_rngs = utils.psplit(rng)
      (_, val_ploss, val_ploss_cycle, val_ploss_constraint, val_ploss_reg,
       _, _, _, _, _) = peval_fn((next_rngs, pstate), val_batch, mdm_val_batch)
      
      val_loss = flax.jax_utils.unreplicate(val_ploss).item()
      val_loss_cycle = flax.jax_utils.unreplicate(val_ploss_cycle).item()
      val_loss_constraint = flax.jax_utils.unreplicate(val_ploss_constraint).item()
      val_loss_reg = flax.jax_utils.unreplicate(val_ploss_reg).item()
      epoch_val_losses_total.append(val_loss)
      epoch_val_losses_cycle.append(val_loss_cycle)
      epoch_val_losses_constraint.append(val_loss_constraint)
      epoch_val_losses_reg.append(val_loss_reg)

      if (step == 0 or (step + 1) % config.training.log_freq == 0) and utils.is_coordinator():
        t = time.perf_counter() - s
        logging.info('[epoch %03d, step %03d] %.3f sec; val loss: %.5e', epoch, step + 1, t, val_loss)

    # Update validation curves.
    val_losses_total.append(np.mean(epoch_val_losses_total))
    val_losses_cycle.append(np.mean(epoch_val_losses_cycle))
    val_losses_constraint.append(np.mean(epoch_val_losses_constraint))
    val_losses_reg.append(np.mean(epoch_val_losses_reg))

    # Save progress snapshot.
    if ((epoch + 1) % config.training.snapshot_epoch_freq == 0
        and utils.is_coordinator()):
      state = flax.jax_utils.unreplicate(pstate)
      # Save stats.
      np.save(os.path.join(progress_dir, 'epoch_times.npy'), epoch_times)
      np.save(os.path.join(progress_dir, 'losses_total.npy'), losses_total)
      np.save(os.path.join(progress_dir, 'losses_cycle.npy'), losses_cycle)
      np.save(os.path.join(progress_dir, 'losses_constraint.npy'), losses_constraint)
      np.save(os.path.join(progress_dir, 'losses_reg.npy'), losses_reg)
      np.save(os.path.join(progress_dir, 'val_losses_total.npy'), val_losses_total)
      np.save(os.path.join(progress_dir, 'val_losses_cycle.npy'), val_losses_cycle)
      np.save(os.path.join(progress_dir, 'val_losses_constraint.npy'), val_losses_constraint)
      np.save(os.path.join(progress_dir, 'val_losses_reg.npy'), val_losses_reg)
      
      # Save plot of training progress.
      fig = plotter(
        losses_total, losses_cycle, losses_constraint, losses_reg,
        val_losses_total, batch, x_fwd, x_fwdbwd, y, y_bwd, stds,
        state.cycle_weight.item(), state.constraint_weight.item(),
        state.regularization_weight.item())
      fig.savefig(os.path.join(progress_dir, f'progress_{epoch + 1:03d}.png'))

    # Save checkpoint.
    if ((epoch + 1) % config.training.ckpt_epoch_freq == 0
        and utils.is_coordinator()):
      state = flax.jax_utils.unreplicate(pstate)
      state = state.replace(epoch=epoch + 1)
      ckpt_mgr.save(epoch + 1, args=ocp.args.StandardSave(state))
  
  ckpt_mgr.wait_until_finished()


if __name__ == '__main__':
  app.run(main)
