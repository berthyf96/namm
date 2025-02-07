"""Helper functions for visualizing data or training progress."""

from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import losses
import pdes


def get_dm_progress_plotter(score_config, namm_config):
  """Returns a function that plots regular score model training progress."""
  image_shape = (score_config.data.height, score_config.data.width, score_config.data.num_channels)
  constraint_losses_fn = losses.get_constraint_losses_fn(namm_config)
  cmap = 'viridis'
  if namm_config.constraint.type in ['kolmogorov', 'incompress', 'kol+incompress']:
    if namm_config.data.kolmogorov_representation == 'volume':
      vorticity_fn = pdes.get_vorticity_fn(vmap=False)
      image_fn = partial(pdes.make_image_from_trajectory,
                      n_per_row=namm_config.data.num_kolmogorov_states_per_row)
    elif namm_config.data.kolmogorov_representation == 'image':
      vorticity_fn = pdes.get_vorticity_image_fn(vmap=False)
      image_fn = lambda inp: inp
    cmap = sns.cm.icefire

  if namm_config.constraint.type == 'kol+incompress':
    namm_config.constraint.type = 'kolmogorov'
    kol_losses_fn = losses.get_constraint_losses_fn(namm_config)
    namm_config.constraint.type = 'incompress'
    div_losses_fn = losses.get_constraint_losses_fn(namm_config)
    namm_config.constraint.type = 'kol+incompress'

  if namm_config.constraint.type == 'count':
    cnn_count_fn = losses.get_cnn_predictions_fn(image_shape)

  def plotter(losses, losses_eval, samples, histogram_type='constraint',
              xlim=(None, None), piecewise=False):
    # Reshape data.
    samples = samples.reshape(-1, *image_shape)

    # Compute constraint losses.
    constr_losses = constraint_losses_fn(samples)
    if piecewise:
      # Apply step function to images.
      samples = jnp.piecewise(samples, [samples < 0.5, samples >= 0.5], [0, 1])
    
    sample = samples[0]
    if namm_config.constraint.type in ['kolmogorov', 'incompress', 'kol+incompress']:
      sample = image_fn(vorticity_fn(sample))

    fig = plt.figure(figsize=(17, 5))

    plt.subplot(1, 3, 1)
    plt.plot(np.log(losses), label='train')

    plt.plot(np.log(losses_eval), label='val')
    plt.legend(fontsize=18)
    plt.title('Score total log-loss', fontsize=21)

    plt.subplot(1, 3, 2)
    plt.imshow(sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title('sample', fontsize=21)

    plt.subplot(1, 3, 3)
    if histogram_type == 'flux':
      fluxes = np.sum(samples, axis=(1, 2, 3))
      plt.hist(fluxes, label='samples')
      plt.title('total flux', fontsize=21)
    elif histogram_type == 'count':
      counts = cnn_count_fn(samples).reshape(-1)
      plt.hist(counts, label='samples')
      plt.title('CNN count', fontsize=21)
    elif histogram_type == 'constraint' and not np.isnan(constr_losses).any() and not np.isinf(constr_losses).any():
      plt.hist(constr_losses, label='samples')
      plt.title('constraint loss', fontsize=21)
    plt.legend(fontsize=18)
    plt.xlim(*xlim)

    return fig
  

  def two_constraint_plotter(losses, losses_eval, samples,
                             xlim=(None, None), piecewise=False):
    # Reshape data.
    samples = samples.reshape(-1, *image_shape)

    # Compute constraint losses.
    kol_constr_losses = kol_losses_fn(samples)
    div_constr_losses = div_losses_fn(samples)
    if piecewise:
      # Apply step function to images.
      samples = jnp.piecewise(samples, [samples < 0.5, samples >= 0.5], [0, 1])
    
    sample = samples[0]
    if namm_config.constraint.type in ['kolmogorov', 'incompress', 'kol+incompress']:
      sample = image_fn(vorticity_fn(sample))

    fig = plt.figure(figsize=(15, 15))

    plt.subplot(2, 2, 1)
    plt.plot(np.log(losses), label='train')
    plt.plot(np.log(losses_eval), label='val')
    plt.legend(fontsize=18)
    plt.title('Score total log-loss', fontsize=21)

    plt.subplot(2, 2, 2)
    plt.imshow(sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title('sample', fontsize=21)

    plt.subplot(2, 2, 3)
    plt.hist(kol_constr_losses, label='samples')
    plt.title('constraint loss (PDE)', fontsize=21)
    plt.legend(fontsize=18)
    plt.xlim(*xlim)
    
    plt.subplot(2, 2, 4)
    plt.hist(div_constr_losses, label='samples')
    plt.title('constraint loss (div)', fontsize=21)
    plt.legend(fontsize=18)
    plt.xlim(*xlim)
    return fig
  
  if namm_config.constraint.type == 'flux':
    h = 0.05 * namm_config.constraint.total_flux
    xlim = (namm_config.constraint.total_flux - h, namm_config.constraint.total_flux + h)
    return partial(plotter, histogram_type='flux', xlim=xlim, piecewise=False)
  elif namm_config.constraint.type == 'periodic':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.05), piecewise=False)
  elif namm_config.constraint.type == 'burgers':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.05), piecewise=False)
  elif namm_config.constraint.type == 'kolmogorov':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.1), piecewise=False)
  elif namm_config.constraint.type == 'incompress':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.3), piecewise=False)
  elif namm_config.constraint.type == 'kol+incompress':
    return partial(two_constraint_plotter, xlim=(0., 0.1), piecewise=False)
  elif namm_config.constraint.type == 'count':
    return partial(plotter, histogram_type='count', xlim=(5, 11), piecewise=False)


def get_mdm_progress_plotter(score_config, namm_config):
  """Returns a function that plots MDM training progress."""
  image_shape = (score_config.data.height, score_config.data.width, score_config.data.num_channels)
  constraint_losses_fn = losses.get_constraint_losses_fn(namm_config)
  cmap = 'viridis'
  if namm_config.constraint.type in ['kolmogorov', 'incompress', 'kol+incompress']:
    if namm_config.data.kolmogorov_representation == 'volume':
      vorticity_fn = pdes.get_vorticity_fn(vmap=False)
      image_fn = partial(pdes.make_image_from_trajectory,
                      n_per_row=namm_config.data.num_kolmogorov_states_per_row)
    elif namm_config.data.kolmogorov_representation == 'image':
      vorticity_fn = pdes.get_vorticity_image_fn(vmap=False)
      image_fn = lambda inp: inp
    cmap = sns.cm.icefire
  if namm_config.constraint.type == 'kol+incompress':
    namm_config.constraint.type = 'kolmogorov'
    kol_losses_fn = losses.get_constraint_losses_fn(namm_config)
    namm_config.constraint.type = 'incompress'
    div_losses_fn = losses.get_constraint_losses_fn(namm_config)
    namm_config.constraint.type = 'kol+incompress'
  
  if namm_config.constraint.type == 'count':
    cnn_count_fn = losses.get_cnn_predictions_fn(image_shape)

  def plotter(losses_score, losses_eval, x, x_fwd, y, y_bwd, histogram_type='constraint',
              xlim=(None, None), piecewise=False):
    # Reshape data.
    x = x.reshape(-1, *image_shape)
    x_fwd = x_fwd.reshape(-1, *image_shape)
    y = y.reshape(-1, *image_shape)
    y_bwd = y_bwd.reshape(-1, *image_shape)

    # Compute constraint losses.
    constr_losses = constraint_losses_fn(y_bwd)
    if piecewise:
      # Apply step function to images.
      y_bwd = jnp.piecewise(y_bwd, [y_bwd < 0.5, y_bwd >= 0.5], [0, 1])

    x_sample = x[0]
    x_fwd_sample = x_fwd[0]
    y_sample = y[0]
    y_bwd_sample = y_bwd[0]
    if namm_config.constraint.type in ['kolmogorov', 'incompress', 'kol+incompress']:
      x_sample = image_fn(vorticity_fn(x_sample))
      x_fwd_sample = image_fn(vorticity_fn(x_fwd_sample))
      y_sample = image_fn(vorticity_fn(y_sample))
      y_bwd_sample = image_fn(vorticity_fn(y_bwd_sample))

    fig = plt.figure(figsize=(17, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(x_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$x$', fontsize=21)

    plt.subplot(2, 3, 2)
    plt.imshow(x_fwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$G(x)$', fontsize=21)

    plt.subplot(2, 3, 3)
    plt.plot(np.log(losses_score), label='train')
    plt.plot(np.log(losses_eval), label='val')
    plt.legend(fontsize=18)
    plt.title('Score total log-loss', fontsize=21)

    plt.subplot(2, 3, 4)
    plt.imshow(y_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$\tilde{x}\sim \tilde{p}_\theta$', fontsize=21)

    plt.subplot(2, 3, 5)
    plt.imshow(y_bwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(\tilde{x})$', fontsize=21)

    plt.subplot(2, 3, 6)
    if histogram_type == 'flux':
      y_bwd_fluxes = np.sum(y_bwd, axis=(1, 2, 3))
      plt.hist(y_bwd_fluxes, label=r'$F(\tilde{x})$')
      plt.title('total flux', fontsize=21)
    elif histogram_type == 'count':
      counts = cnn_count_fn(y_bwd).reshape(-1)
      plt.hist(counts, label=r'$F(\tilde{x})$')
      plt.title('CNN count', fontsize=21)
    elif histogram_type == 'constraint' and not np.isnan(constr_losses).any():
      plt.hist(constr_losses, label=r'$F(\tilde{x})$')
      plt.title('constraint loss', fontsize=21)
    plt.legend(fontsize=18)
    plt.xlim(*xlim)

    return fig
  
  def two_constraint_plotter(losses_score, losses_eval, x, x_fwd, y, y_bwd,
                             xlim=(None, None), piecewise=False):
    # Reshape data.
    x = x.reshape(-1, *image_shape)
    x_fwd = x_fwd.reshape(-1, *image_shape)
    y = y.reshape(-1, *image_shape)
    y_bwd = y_bwd.reshape(-1, *image_shape)

    # Compute constraint losses.
    kol_constr_losses = kol_losses_fn(y_bwd)
    div_constr_losses = div_losses_fn(y_bwd)
    if piecewise:
      # Apply step function to images.
      y_bwd = jnp.piecewise(y_bwd, [y_bwd < 0.5, y_bwd >= 0.5], [0, 1])

    x_sample = x[0]
    x_fwd_sample = x_fwd[0]
    y_sample = y[0]
    y_bwd_sample = y_bwd[0]
    if namm_config.constraint.type in ['kolmogorov', 'incompress', 'kol+incompress']:
      x_sample = image_fn(vorticity_fn(x_sample))
      x_fwd_sample = image_fn(vorticity_fn(x_fwd_sample))
      y_sample = image_fn(vorticity_fn(y_sample))
      y_bwd_sample = image_fn(vorticity_fn(y_bwd_sample))

    fig = plt.figure(figsize=(17, 15))

    plt.subplot(3, 3, 1)
    plt.imshow(x_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$x$', fontsize=21)

    plt.subplot(3, 3, 2)
    plt.imshow(x_fwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$G(x)$', fontsize=21)

    plt.subplot(3, 3, 3)
    plt.plot(np.log(losses_score), label='train')
    plt.plot(np.log(losses_eval), label='val')
    plt.legend(fontsize=18)
    plt.title('Score total log-loss', fontsize=21)

    plt.subplot(3, 3, 4)
    plt.imshow(y_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$\tilde{x}\sim \tilde{p}_\theta$', fontsize=21)

    plt.subplot(3, 3, 5)
    plt.imshow(y_bwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(\tilde{x})$', fontsize=21)

    plt.subplot(3, 3, 7)
    plt.hist(kol_constr_losses, label=r'$F(\tilde{x})$')
    plt.title('constraint loss (PDE)', fontsize=21)
    plt.legend(fontsize=18)
    plt.xlim(*xlim)

    plt.subplot(3, 3, 8)
    plt.hist(div_constr_losses, label=r'$F(\tilde{x})$')
    plt.title('constraint loss (div)', fontsize=21)
    plt.legend(fontsize=18)
    plt.xlim(0, 0.5)

    return fig
  
  if namm_config.constraint.type == 'flux':
    h = 0.05 * namm_config.constraint.total_flux
    xlim = (namm_config.constraint.total_flux - h, namm_config.constraint.total_flux + h)
    return partial(plotter, histogram_type='flux', xlim=xlim, piecewise=False)
  elif namm_config.constraint.type == 'periodic':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.05), piecewise=False)
  elif namm_config.constraint.type == 'burgers':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.05), piecewise=False)
  elif namm_config.constraint.type == 'kolmogorov':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.1), piecewise=False)
  elif namm_config.constraint.type == 'incompress':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.3), piecewise=False)
  elif namm_config.constraint.type == 'kol+incompress':
    return partial(two_constraint_plotter, xlim=(0., 0.1), piecewise=False)
  elif namm_config.constraint.type == 'count':
    return partial(plotter, histogram_type='count', xlim=(5, 11), piecewise=False)

  def plotter_kolmogorov(losses_total, losses_cycle, losses_constraint, losses_score,
                         val_losses, val_losses_cycle, val_losses_constraint, val_losses_score,
                         val_steps, x, x_fwd, x_fwdbwd, y, y_bwd, stds, mdm_samples, inverse_mdm_samples,
                         cycle_weight, constraint_weight, regularization_weight,
                         histogram_type='constraint',
                         xlim=(None, None)):
    # Reshape data.
    x = x.reshape(-1, *image_shape)
    x_fwd = x_fwd.reshape(-1, *image_shape)
    x_fwdbwd = x_fwdbwd.reshape(-1, *image_shape)
    y = y.reshape(-1, *image_shape)
    y_bwd = y_bwd.reshape(-1, *image_shape)
    if stds is not None:
      stds = stds.reshape(-1)
    mdm_samples = mdm_samples.reshape(-1, *image_shape)
    inverse_mdm_samples = inverse_mdm_samples.reshape(-1, *image_shape)

    # Compute constraint losses.
    constr_losses = constraint_losses_fn(y_bwd)
    mdm_constr_losses = constraint_losses_fn(inverse_mdm_samples)

    x_sample = image_fn(vorticity_fn(x[0]))
    x_fwd_sample = image_fn(vorticity_fn(x_fwd[0]))
    x_fwdbwd_sample = image_fn(vorticity_fn(x_fwdbwd[0]))
    y_sample = image_fn(vorticity_fn(y[0]))
    y_bwd_sample = image_fn(vorticity_fn(y_bwd[0]))
    mdm_sample = image_fn(vorticity_fn(mdm_samples[0]))
    inverse_mdm_sample = image_fn(vorticity_fn(inverse_mdm_samples[0]))

    fig = plt.figure(figsize=(25, 20))
    # Plot loss curves.
    plt.subplot(3, 4, 1)
    plt.plot(np.log(losses_total), label='train')
    plt.plot(val_steps, np.log(val_losses), label='val')
    plt.legend(fontsize=18)
    plt.title('total log-loss', fontsize=21)
    plt.subplot(3, 4, 2)
    plt.plot(np.log(losses_cycle), label='train')
    plt.plot(val_steps, np.log(val_losses_cycle), label='val')
    plt.legend(fontsize=18)
    plt.title(f'cycle log-loss (w = {np.around(cycle_weight, 6)})', fontsize=21)
    plt.subplot(3, 4, 3)
    plt.plot(np.log(losses_constraint), label='train')
    plt.plot(val_steps, np.log(val_losses_constraint), label='val')
    plt.legend(fontsize=18)
    plt.title(f'constraint log-loss (w = {np.around(constraint_weight, 6)})', fontsize=21)
    plt.subplot(3, 4, 4)
    plt.plot(np.log(losses_score), label='train')
    plt.plot(val_steps, np.log(val_losses_score), label='val')
    plt.legend(fontsize=18)
    plt.title(f'DSM log-loss (w = {np.around(regularization_weight, 6)})', fontsize=21)

    # Plot training sample and constraint histogram.
    plt.subplot(3, 4, 5)
    plt.imshow(x_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$x$', fontsize=21)
    plt.subplot(3, 4, 6)
    plt.imshow(x_fwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$G(x)$', fontsize=21) 
    plt.subplot(3, 4, 7)
    plt.imshow(x_fwdbwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(G(x))$', fontsize=21)     
    plt.subplot(3, 4, 8)
    if histogram_type == 'flux':
      y_bwd_fluxes = np.sum(y_bwd, axis=(1, 2, 3))
      mdm_fluxes = np.sum(mdm_samples, axis=(1, 2, 3))
      plt.hist(y_bwd_fluxes, label='LAMM')
      plt.hist(mdm_fluxes, label='MDM')
      plt.title('total flux', fontsize=21)
    elif histogram_type == 'constraint':
      plt.hist(constr_losses, label='LAMM')
      plt.hist(mdm_constr_losses, label='MDM')
      plt.title('constraint loss', fontsize=21)
    plt.legend(fontsize=18)
    plt.xlim(*xlim)

    # Plot inverse LAMM and MDM samples.
    plt.subplot(3, 4, 9)
    plt.imshow(y_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$\tilde{x}=G(x)+\sigma z$' + f' ($\sigma$ = {stds[0]:.2f})', fontsize=21)
    plt.subplot(3, 4, 10)
    plt.imshow(y_bwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(\tilde{x})$ (LAMM)', fontsize=21)
    plt.subplot(3, 4, 11)
    plt.imshow(mdm_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$\tilde{x}\sim p_\theta$', fontsize=21)
    plt.subplot(3, 4, 12)
    plt.imshow(inverse_mdm_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(\tilde{x})$ (MDM)', fontsize=21)
    
    return fig
  
  if namm_config.constraint.type == 'flux':
    h = 0.05 * namm_config.constraint.total_flux
    xlim = (namm_config.constraint.total_flux - h, namm_config.constraint.total_flux + h)
    return partial(plotter, histogram_type='flux', xlim=xlim, piecewise=False)
  elif namm_config.constraint.type == 'periodic':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.05), piecewise=False)
  elif namm_config.constraint.type == 'burgers':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.05), piecewise=False)
  elif namm_config.constraint.type == 'kolmogorov':
    return partial(plotter_kolmogorov, xlim=(0., 0.1))
  elif namm_config.constraint.type == 'incompress':
    return partial(plotter_kolmogorov, xlim=(0., 0.3))
  elif namm_config.constraint.type == 'kol+incompress':
    return partial(plotter_kolmogorov, xlim=(0., 0.1))
  elif namm_config.constraint.type == 'count':
    return partial(plotter, histogram_type='constraint', xlim=(0, 1), piecewise=False)


def get_namm_progress_plotter(namm_config):
  """Returns a function that plots LMM training progress."""
  image_shape = (namm_config.data.height, namm_config.data.width, namm_config.data.num_channels)
  constraint_losses_fn = losses.get_constraint_losses_fn(namm_config)
  cmap = 'viridis'
  if namm_config.constraint.type in ['kolmogorov', 'incompress', 'kol+incompress']:
    if namm_config.data.kolmogorov_representation == 'volume':
      vorticity_fn = pdes.get_vorticity_fn(vmap=False)
      image_fn = partial(pdes.make_image_from_trajectory,
                      n_per_row=namm_config.data.num_kolmogorov_states_per_row)
    elif namm_config.data.kolmogorov_representation == 'image':
      vorticity_fn = pdes.get_vorticity_image_fn(vmap=False)
      image_fn = lambda inp: inp
    cmap = sns.cm.icefire
  if namm_config.constraint.type == 'kol+incompress':
    namm_config.constraint.type = 'kolmogorov'
    kol_losses_fn = losses.get_constraint_losses_fn(namm_config)
    namm_config.constraint.type = 'incompress'
    div_losses_fn = losses.get_constraint_losses_fn(namm_config)
    namm_config.constraint.type = 'kol+incompress'
  if namm_config.constraint.type == 'count':
    cnn_count_fn = losses.get_cnn_predictions_fn(image_shape)

  def plotter(losses_total, losses_cycle, losses_constraint, losses_reg,
              val_losses, x, x_fwd, x_fwdbwd, y, y_bwd, stds,
              cycle_weight, constraint_weight, regularization_weight,
              histogram_type='constraint',
              xlim=(None, None), piecewise=False):
    sample_from_mdm = stds is None
    # Reshape data.
    x = x.reshape(-1, *image_shape)
    x_fwd = x_fwd.reshape(-1, *image_shape)
    x_fwdbwd = x_fwdbwd.reshape(-1, *image_shape)
    y = y.reshape(-1, *image_shape)
    y_bwd = y_bwd.reshape(-1, *image_shape)
    if not sample_from_mdm:
      stds = stds.reshape(-1)

    # Compute constraint losses.
    constr_losses = constraint_losses_fn(y_bwd)
    if piecewise:
      # Apply step function to images.
      y_bwd = jnp.piecewise(y_bwd, [y_bwd < 0.5, y_bwd >= 0.5], [0, 1])

    x_sample = x[0]
    x_fwd_sample = x_fwd[0]
    x_fwdbwd_sample = x_fwdbwd[0]
    y_sample = y[0]
    y_bwd_sample = y_bwd[0]

    fig = plt.figure(figsize=(17, 20))
    plt.subplot(4, 3, 1)
    plt.plot(np.log(losses_total), label='train')
    plt.plot(np.log(val_losses), label='val')
    plt.legend(fontsize=18)
    plt.title('LMM total log-loss', fontsize=21)
    plt.subplot(4, 3, 2)
    plt.plot(np.log(losses_cycle))
    plt.title(f'cycle log-loss (w = {np.around(cycle_weight, 6)})', fontsize=21)
    plt.subplot(4, 3, 3)
    plt.plot(np.log(losses_constraint))
    plt.title(f'constraint log-loss (w = {np.around(constraint_weight, 6)})', fontsize=21)
    plt.subplot(4, 3, 4)
    plt.plot(np.log(losses_reg))
    plt.title(f'reg. log-loss (w = {np.around(regularization_weight, 6)})', fontsize=21)
    plt.subplot(4, 3, 7)
    plt.imshow(x_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$x$', fontsize=21)
    plt.subplot(4, 3, 8)
    plt.imshow(x_fwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$G(x)$', fontsize=21) 
    plt.subplot(4, 3, 9)
    plt.imshow(x_fwdbwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(G(x))$', fontsize=21)     
    plt.subplot(4, 3, 10)
    plt.imshow(y_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    if sample_from_mdm:
      plt.title(r'$\tilde{x}\sim p_\theta$', fontsize=21)
    else:
      plt.title(r'$\tilde{x}=G(x)+\sigma z$' + f' ($\sigma$ = {stds[0]:.2f})', fontsize=21)
    plt.subplot(4, 3, 11)
    plt.imshow(y_bwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(\tilde{x})$', fontsize=21)

    plt.subplot(4, 3, 12)
    if histogram_type == 'flux':
      y_bwd_fluxes = np.sum(y_bwd, axis=(1, 2, 3))
      plt.hist(y_bwd_fluxes, label=r'$F(\tilde{x})$')
      plt.title('total flux', fontsize=21)
    elif histogram_type == 'count':
      counts = cnn_count_fn(y_bwd).reshape(-1)
      plt.hist(counts, label=r'$F(\tilde{x})$')
      plt.title('CNN count', fontsize=21)
    elif histogram_type == 'constraint':
      plt.hist(constr_losses, label=r'$F(\tilde{x})$')
      plt.title('constraint loss', fontsize=21)
    plt.legend(fontsize=18)
    plt.xlim(*xlim)
    
    return fig

  def plotter_kolmogorov(losses_total, losses_cycle, losses_constraint, losses_reg,
                         val_losses, x, x_fwd, x_fwdbwd, y, y_bwd, stds,
                         cycle_weight, constraint_weight, regularization_weight,
                         xlim=(None, None)):
    sample_from_mdm = stds is None
    # Reshape data.
    x = x.reshape(-1, *image_shape)
    x_fwd = x_fwd.reshape(-1, *image_shape)
    x_fwdbwd = x_fwdbwd.reshape(-1, *image_shape)
    y = y.reshape(-1, *image_shape)
    y_bwd = y_bwd.reshape(-1, *image_shape)
    if not sample_from_mdm:
      stds = stds.reshape(-1)

    # Compute constraint losses.
    if namm_config.constraint.type == 'kol+incompress':
      kol_constr_losses = kol_losses_fn(y_bwd)
      div_constr_losses = div_losses_fn(y_bwd)
    else:
      constr_losses = constraint_losses_fn(y_bwd)

    x_sample = image_fn(vorticity_fn(x[0]))
    x_fwd_sample = image_fn(vorticity_fn(x_fwd[0]))
    x_fwdbwd_sample = image_fn(vorticity_fn(x_fwdbwd[0]))
    y_sample = image_fn(vorticity_fn(y[0]))
    y_bwd_sample = image_fn(vorticity_fn(y_bwd[0]))

    y_vx = image_fn(y[0][:, :, ::2])
    y_vy = image_fn(y[0][:, :, 1::2])
    y_bwd_vx = image_fn(y_bwd[0][:, :, ::2])
    y_bwd_vy = image_fn(y_bwd[0][:, :, 1::2])

    fig = plt.figure(figsize=(17, 30))
    plt.subplot(6, 3, 1)
    plt.plot(np.log(losses_total), label='train')
    plt.plot(np.log(val_losses), label='val')
    plt.legend(fontsize=18)
    plt.title('LMM total log-loss', fontsize=21)
    plt.subplot(6, 3, 2)
    plt.plot(np.log(losses_cycle))
    plt.title(f'cycle log-loss (w = {np.around(cycle_weight, 6)})', fontsize=21)
    plt.subplot(6, 3, 3)
    plt.plot(np.log(losses_constraint))
    plt.title(f'constraint log-loss (w = {np.around(constraint_weight, 6)})', fontsize=21)
    plt.subplot(6, 3, 4)
    plt.plot(np.log(losses_reg))
    plt.title(f'reg. log-loss (w = {np.around(regularization_weight, 6)})', fontsize=21)
    plt.subplot(6, 3, 7)
    plt.imshow(x_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$x$', fontsize=21)
    plt.subplot(6, 3, 8)
    plt.imshow(x_fwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$G(x)$', fontsize=21) 
    plt.subplot(6, 3, 9)
    plt.imshow(x_fwdbwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(G(x))$', fontsize=21)     
    plt.subplot(6, 3, 10)
    plt.imshow(y_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    if sample_from_mdm:
      plt.title(r'$\tilde{x}\sim p_\theta$', fontsize=21)
    else:
      plt.title(r'$\tilde{x}=G(x)+\sigma z$' + f' ($\sigma$ = {stds[0]:.2f})', fontsize=21)
    plt.subplot(6, 3, 11)
    plt.imshow(y_bwd_sample, cmap=cmap); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(\tilde{x})$', fontsize=21)
    if namm_config.constraint.type == 'kol+incompress':
      plt.subplot(6, 3, 12)
      plt.hist(kol_constr_losses, label=r'$F(\tilde{x})$')
      plt.title('constraint loss (PDE)', fontsize=21)
      plt.legend(fontsize=18)
      plt.xlim(*xlim)
      plt.subplot(6, 3, 15)
      plt.hist(div_constr_losses, label=r'$F(\tilde{x})$')
      plt.title('constraint loss (div)', fontsize=21)
      plt.legend(fontsize=18)
      plt.xlim(0, 0.5)
    else:
      plt.subplot(6, 3, 12)
      plt.hist(constr_losses, label=r'$F(\tilde{x})$')
      plt.title('constraint loss', fontsize=21)
      plt.legend(fontsize=18)
      plt.xlim(*xlim)
    plt.subplot(6, 3, 13)
    plt.imshow(y_vx); plt.axis('off'); plt.colorbar()
    plt.title(r'$\tilde{x}=G(x)+\sigma z$: $v_x$', fontsize=21)
    plt.subplot(6, 3, 14)
    plt.imshow(y_bwd_vx); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(\tilde{x})$: $v_x$', fontsize=21)
    plt.subplot(6, 3, 16)
    plt.imshow(y_vy); plt.axis('off'); plt.colorbar()
    plt.title(r'$\tilde{x}=G(x)+\sigma z$: $v_y$', fontsize=21)
    plt.subplot(6, 3, 17)
    plt.imshow(y_bwd_vy); plt.axis('off'); plt.colorbar()
    plt.title(r'$F(\tilde{x})$: $v_y$', fontsize=21)
    
    return fig
  
  if namm_config.constraint.type == 'flux':
    h = 0.05 * namm_config.constraint.total_flux
    xlim = (namm_config.constraint.total_flux - h, namm_config.constraint.total_flux + h)
    return partial(plotter, histogram_type='flux', xlim=xlim, piecewise=False)
  elif namm_config.constraint.type == 'periodic':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.05), piecewise=False)
  elif namm_config.constraint.type == 'burgers':
    return partial(plotter, histogram_type='constraint', xlim=(0., 0.05), piecewise=False)
  elif namm_config.constraint.type == 'kolmogorov':
    return partial(plotter_kolmogorov, xlim=(0., 0.1))
  elif namm_config.constraint.type == 'incompress':
    return partial(plotter_kolmogorov, xlim=(0., 0.3))
  elif namm_config.constraint.type == 'kol+incompress':
    return partial(plotter_kolmogorov, xlim=(0., 0.1))
  elif namm_config.constraint.type == 'count':
    return partial(plotter, histogram_type='count', xlim=(5, 11), piecewise=False)