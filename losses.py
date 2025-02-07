import diffrax
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
import optax

from model import CNN
import pdes
from score_flow import losses as score_losses
from score_flow import utils
from score_flow.sde_lib import VESDE, VPSDE


def tv_loss_fn(x):
  pixel_dif1 = x[:, 1:, :, :] - x[:, :-1, :, :]
  pixel_dif2 = x[:, :, 1:, :] - x[:, :, :-1, :]
  tv = (
      jnp.mean(jnp.abs(pixel_dif1), axis=(1, 2, 3)) +
      jnp.mean(jnp.abs(pixel_dif2), axis=(1, 2, 3)))
  return jnp.mean(tv)


def get_constraint_weight_fn(config):
  """Get sigmoidal annealing function for constraint weight."""
  init_weight = config.optim.constraint_init_weight
  max_weight = config.optim.constraint_weight
  pivot_steps = config.optim.constraint_annealing_pivot
  rate = np.log(max_weight / init_weight - 1) / pivot_steps
  def constraint_weight_fn(step):
    w = max_weight / (1 + np.exp(-(step - pivot_steps) * rate))
    return w
  return constraint_weight_fn


def get_regularization_weight_fn(config):
  """Get sigmoidal annealing function for constraint weight."""
  init_weight = config.optim.regularization_init_weight
  max_weight = config.optim.regularization_max_weight
  pivot_steps = config.optim.regularization_annealing_pivot
  rate = np.log(max_weight / init_weight - 1) / pivot_steps
  def weight_fn(step):
    w = max_weight / (1 + np.exp(-(step - pivot_steps) * rate))
    return w
  return weight_fn


def get_divergence_fn(nt, representation='image', n_per_row=4):
  # Fixed parameters:
  size = 64
  grid = cfd.grids.Grid(
    shape=(size, size),
    domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
  )
  bc = cfd.boundaries.periodic_boundary_conditions(2)
  n_rows = nt // n_per_row

  def divergence(v_image):
    v0 = cfd.initial_conditions.wrap_variables(
      var=(v_image[:, :, 0], v_image[:, :, 1]),
      grid=grid,
      bcs=(bc, bc))
    div = cfd.finite_differences.divergence(v0).data
    return div

  @jax.vmap
  def divergence_images(image):
    # Reshape image into state array of shape (nt, h, w, 2).
    if representation == 'volume':
      # Assume image has shape (h, w, 2 * nt).
      nt = image.shape[-1] // 2
      v_images = jnp.stack(tuple(
        [image[:, :, 2 * i:2 * (i + 1)] for i in range(nt)]))
    elif representation == 'image':
      v_images = jnp.stack(tuple(
        [image[i * size:(i + 1) * size, j * size:(j + 1) * size] \
        for i in range(n_rows) for j in range(n_per_row)]))  # (nt, h, w, 2)
    return jax.vmap(divergence)(v_images)  # (nt, h, w)

  return divergence_images


def get_burgers_constraint_losses_fn(t0, dt, inner_steps):
  # Parameters
  mu = 1
  nu = 0.05

  # Spatial grid
  x0 = 0
  x1 = 10
  nx = 64

  # Time grid
  nt = 64
  t1 = t0 + dt * inner_steps * (nt - 1)

  solver = pdes.CrankNicolson(rtol=1e-3, atol=1e-3)
  stepsize_controller = diffrax.ConstantStepSize()
  pde = pdes.BurgersEquation(
    mu, nu, x0, x1, nx, t0, t1, dt, nt, solver=solver, stepsize_controller=stepsize_controller)

  def step(y_init, t_init):
    def _scan_fn(ycurr, tcurr):
      ynext, _, _, _, _ = solver.step(
        terms=diffrax.ODETerm(pde.dydt),
        t0=tcurr,
        t1=tcurr + dt,
        y0=ycurr,
        args=[],
        solver_state=None,
        made_jump=None)
      return ynext, ynext
  
    t = jnp.linspace(t_init, t_init + dt * inner_steps, inner_steps)
    y1, _ = jax.lax.scan(_scan_fn, y_init, t)
    return y1

  ts = jnp.linspace(t0, t1 - dt * inner_steps, nt - 1)
  @jax.vmap
  def constraint_losses_fn(x):
    # Assume flipped images of shape (nx, nt, 1).
    x = jnp.flipud(x[:, :, 0])
    xnext = jax.vmap(step, in_axes=(1, 0), out_axes=1)(x[:, :-1], ts)
    residual = x[:, 1:] - xnext
    return jnp.mean(jnp.abs(residual))

  return constraint_losses_fn

  
def get_kolmogorov_constraint_losses_fn(reynolds, nt, dt, inner_steps, kolmogorov_forcing=True, representation='image', n_per_row=4):
  # Fixed parameters:
  size = 64
  density = 1.
  grid = cfd.grids.Grid(
    shape=(size, size),
    domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
  )
  bc = cfd.boundaries.periodic_boundary_conditions(2)
  if kolmogorov_forcing:
    forcing = cfd.forcings.simple_turbulence_forcing(
      grid=grid,
      constant_magnitude=1.,
      constant_wavenumber=4.,
      linear_coefficient=-0.1,
      forcing_type='kolmogorov',
    )
  else:
    forcing = None

  step_fn = cfd.funcutils.repeated(
    f=cfd.equations.semi_implicit_navier_stokes(
      grid=grid,
      forcing=forcing,
      dt=dt,
      density=density,
      viscosity=1 / reynolds,
      time_stepper=cfd.time_stepping.forward_euler
    ),
    steps=inner_steps
  )

  def one_step(v_image):
    """Run forward simulator one step.
  
    Args:
      v_image: np.ndarray of shape (h, w, 2), where
        `vimage[:, :, 0]` and `vimage[:, :, 1]` are
        x and y velocity, respectively.

    Returns:
      vnext_image: image at next time step.
    """
    v0 = cfd.initial_conditions.wrap_variables(
      var=(v_image[:, :, 0], v_image[:, :, 1]),
      grid=grid,
      bcs=(bc, bc))
    v = step_fn(v0)
    vnext_image = jnp.stack((v[0].data, v[1].data), axis=-1)
    return vnext_image

  @jax.vmap
  def image_constraint_losses_fn(image):
    n_rows = nt // n_per_row
    # Assume image of shape (h * n_rows, w * n_per_row, 2).
    v_images = jnp.stack(tuple(
      [image[i * size:(i + 1) * size, j * size:(j + 1) * size] \
      for i in range(n_rows) for j in range(n_per_row)]))  # (nt, h, w, 2)

    vnext_images = jax.vmap(one_step)(v_images[:-1])
    residual = v_images[1:] - vnext_images

    return jnp.mean(jnp.abs(residual))

  @jax.vmap
  def volume_constraint_losses_fn(volume):
    # Assume volume of shape (h, w, nt * 2).
    # Reshape to (nt, h, w, 2).
    nt = volume.shape[-1] // 2
    v_images = jnp.stack(tuple(
      [volume[:, :, 2 * i:2 * (i + 1)] for i in range(nt)]))

    vnext_images = jax.vmap(one_step)(v_images[:-1])
    residual = v_images[1:] - vnext_images

    return jnp.mean(jnp.abs(residual))

  if representation == 'image':
    return image_constraint_losses_fn
  elif representation == 'volume':
    return volume_constraint_losses_fn


def get_cnn_predictions_fn(image_shape, cnn_ckpt_path):
  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)

  cnn = CNN()
  params = cnn.init(init_rng, jnp.ones([1, *image_shape]))['params']
  tx = optax.adam(learning_rate=1e-3)

  state = train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
  # Restore checkpoint.
  state = checkpoints.restore_checkpoint(cnn_ckpt_path, state)
  print(f'Counting CNN checkpoint: step {state.step}')

  def predictions_fn(x):
    return CNN().apply({'params': state.params}, x)

  return predictions_fn


def get_constraint_losses_fn(config):
  image_shape = (config.data.height, config.data.width, config.data.num_channels)
  if config.constraint.type == 'flux':
    def constraint_losses_fn(x):
      fluxes = jnp.sum(x, axis=(1, 2, 3))
      return jnp.abs(fluxes - config.constraint.total_flux)
  elif config.constraint.type == 'periodic':
    image_size = config.data.height
    def constraint_losses_fn(x):
      xp = x
      true_a = jnp.tile(xp[:, :image_size // 2, :image_size // 2, :], (1, 2, 2, 1))
      true_b = jnp.tile(xp[:, image_size // 2:, :image_size // 2, :], (1, 2, 2, 1))
      true_c = jnp.tile(xp[:, :image_size // 2, image_size // 2:, :], (1, 2, 2, 1))
      true_d = jnp.tile(xp[:, image_size // 2:, image_size // 2:, :], (1, 2, 2, 1))
      constraint_losses = (
        jnp.mean(jnp.abs(xp - true_a), axis=(1, 2, 3)) +
        jnp.mean(jnp.abs(xp - true_b), axis=(1, 2, 3)) + 
        jnp.mean(jnp.abs(xp - true_c), axis=(1, 2, 3)) +
        jnp.mean(jnp.abs(xp - true_d), axis=(1, 2, 3)))
      return constraint_losses
  elif config.constraint.type == 'burgers':
    constraint_losses_fn = get_burgers_constraint_losses_fn(
      config.constraint.burgers_t0, config.constraint.burgers_dt, config.constraint.burgers_inner_steps)
  elif config.constraint.type == 'kolmogorov':
    reynolds = config.constraint.reynolds
    constraint_losses_fn = get_kolmogorov_constraint_losses_fn(
      reynolds=reynolds,
      nt=config.data.num_kolmogorov_states,
      dt=config.constraint.kolmogorov_dt,
      inner_steps=config.constraint.inner_steps,
      kolmogorov_forcing=config.constraint.kolmogorov_forcing,
      representation=config.data.kolmogorov_representation,
      n_per_row=config.data.num_kolmogorov_states_per_row)
  elif config.constraint.type == 'incompress':
    divergence_fn = get_divergence_fn(
      nt=config.data.num_kolmogorov_states,
      representation=config.data.kolmogorov_representation,
      n_per_row=config.data.num_kolmogorov_states_per_row)
    def constraint_losses_fn(x):
      div = divergence_fn(x)
      return jnp.mean(jnp.abs(div), axis=(1, 2, 3))
  elif config.constraint.type == 'kol+incompress':
    reynolds = config.constraint.reynolds
    divergence_fn = get_divergence_fn(
      nt=config.data.num_kolmogorov_states,
      representation=config.data.kolmogorov_representation,
      n_per_row=config.data.num_kolmogorov_states_per_row)
    kol_losses_fn = get_kolmogorov_constraint_losses_fn(
      reynolds=reynolds,
      nt=config.data.num_kolmogorov_states,
      dt=config.constraint.kolmogorov_dt,
      inner_steps=config.constraint.inner_steps,
      kolmogorov_forcing=config.constraint.kolmogorov_forcing,
      representation=config.data.kolmogorov_representation,
      n_per_row=config.data.num_kolmogorov_states_per_row)
    def constraint_losses_fn(x):
      div = divergence_fn(x)
      div_losses = jnp.mean(jnp.abs(div), axis=(1, 2, 3))
      kol_losses = kol_losses_fn(x)
      return div_losses + kol_losses
  elif config.constraint.type == 'count':
    predictions_fn = get_cnn_predictions_fn(image_shape, config.constraint.counting_cnn_ckpt_path)

    def constraint_losses_fn(x):
      y_pred = predictions_fn(x)
      return jnp.mean(jnp.abs(y_pred - 8), axis=-1)
  else:
    raise ValueError(f'Constraint type {config.constraint.type} not recognized.')
  return constraint_losses_fn


def get_namm_loss_fn(model, constraint_losses_fn, regularization,
                    train, max_sigma=0.1, fixed_sigma=False, fwd_strong_convexity=1.,
                    use_mdm_samples=False):
  """Get loss function for training LMM."""

  def loss_fn(rng, fwd_params, bwd_params, x, mdm_samples,
              cycle_weight, constraint_weight, regularization_weight):
    batch_size = x.shape[0]
    rng, dropout_rng = jax.random.split(rng)
    x_fwd = model.forward(
      {'dropout': dropout_rng}, fwd_params, x, train=train)
    rng, dropout_rng = jax.random.split(rng)
    x_fwdbwd = model.backward(
      {'dropout': dropout_rng}, bwd_params, x_fwd, train=train)

    rng, noise_rng = jax.random.split(rng)
    if fixed_sigma:
      perturb_levels = jnp.ones(batch_size) * max_sigma
    else:
      perturb_levels = jax.random.uniform(noise_rng, (batch_size,), minval=0, maxval=max_sigma)
    rng, noise_rng = jax.random.split(rng)
    z = jax.random.normal(noise_rng, x.shape)
    if use_mdm_samples:
      y = mdm_samples + utils.batch_mul(perturb_levels, z)
    else:
      y = x_fwd + utils.batch_mul(perturb_levels, z)
    rng, dropout_rng = jax.random.split(rng)
    y_bwd = model.backward(
      {'dropout': dropout_rng}, bwd_params, y, train=train)
    rng, dropout_rng = jax.random.split(rng)
    y_bwdfwd = model.forward(
      {'dropout': dropout_rng}, fwd_params, y_bwd, train=train)
    
    # Compute constraint loss.
    losses_constraint = constraint_losses_fn(y_bwd)
    loss_constraint = jnp.mean(losses_constraint)

    # Compute cycle-consistency loss.
    loss_cycle_fwd = jnp.mean(jnp.abs(x_fwdbwd - x))
    loss_cycle_bwd = jnp.mean(jnp.abs(y_bwdfwd - y))
    loss_cycle = 0.5 * loss_cycle_fwd + 0.5 * loss_cycle_bwd

    if regularization == 'fwdid':
      loss_reg = jnp.mean(jnp.abs(x_fwd - x))
    elif regularization == 'sparse_icnn':
      grad = (x_fwd - fwd_strong_convexity * x) / (1 - fwd_strong_convexity)
      loss_reg = jnp.mean(jnp.abs(grad))
    else:
      raise ValueError(f'Regularization type {regularization} not recognized.')
    
    loss = (
      cycle_weight * loss_cycle
      + constraint_weight * loss_constraint
      + regularization_weight * loss_reg)
    
    return loss, (loss_cycle, loss_constraint, loss_reg, x_fwd, x_fwdbwd, y, y_bwd, perturb_levels)
  
  return loss_fn


def get_namm_step_fn(model, fwd_tx, bwd_tx, constraint_losses_fn, regularization,
                    train, max_sigma=0.1, fixed_sigma=False, fwd_strong_convexity=1.,
                    use_mdm_samples=False, update_fwd=True, update_bwd=True):
  """Get one-step training/evaluation function for NAMM."""
  loss_fn = get_namm_loss_fn(
    model, constraint_losses_fn, regularization, train, max_sigma, fixed_sigma=fixed_sigma,
    fwd_strong_convexity=fwd_strong_convexity, use_mdm_samples=use_mdm_samples)
  grad_fn = jax.value_and_grad(loss_fn, argnums=(1, 2), has_aux=True)
  
  def step_fn(carry_state, x, mdm_samples):
    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    
    if train:
      (loss, (loss_cycle, loss_constraint, loss_reg, x_fwd, x_fwdbwd, y, y_bwd, stds)), (fwd_grads, bwd_grads) = grad_fn(
        step_rng, state.fwd_params, state.bwd_params, x, mdm_samples,
        state.cycle_weight, state.constraint_weight, state.regularization_weight)
      
      loss = jax.lax.pmean(loss, axis_name='batch')
      loss_cycle = jax.lax.pmean(loss_cycle, axis_name='batch')
      loss_constraint = jax.lax.pmean(loss_constraint, axis_name='batch')
      loss_reg = jax.lax.pmean(loss_reg, axis_name='batch')
      fwd_grads = jax.lax.pmean(fwd_grads, axis_name='batch')
      bwd_grads = jax.lax.pmean(bwd_grads, axis_name='batch')

      # Update forward params.
      if update_fwd:
        updates, new_fwd_opt_state = fwd_tx.update(fwd_grads, state.fwd_opt_state, state.fwd_params)
        new_fwd_params = optax.apply_updates(state.fwd_params, updates)
        new_fwd_params_ema = jax.tree_map(
          lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
          state.fwd_params, new_fwd_params)
      else:
        new_fwd_opt_state = state.fwd_opt_state
        new_fwd_params = state.fwd_params
        new_fwd_params_ema = state.fwd_params_ema
      # Update backward params.
      if update_bwd:
        updates, new_bwd_opt_state = bwd_tx.update(bwd_grads, state.bwd_opt_state, state.bwd_params)
        new_bwd_params = optax.apply_updates(state.bwd_params, updates)
        new_bwd_params_ema = jax.tree_map(
          lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
          state.bwd_params, new_bwd_params)
      else:
        new_bwd_opt_state = state.bwd_opt_state
        new_bwd_params = state.bwd_params
        new_bwd_params_ema = state.bwd_params_ema
      new_step = state.step + 1
      new_state = state.replace(
        step=new_step,
        fwd_params=new_fwd_params,
        bwd_params=new_bwd_params,
        fwd_params_ema=new_fwd_params_ema,
        bwd_params_ema=new_bwd_params_ema,
        fwd_opt_state=new_fwd_opt_state,
        bwd_opt_state=new_bwd_opt_state,
        rng=rng)
    else:
      (loss, (loss_cycle, loss_constraint, loss_reg, x_fwd, x_fwdbwd, y, y_bwd, stds)), _ = grad_fn(
        step_rng, state.fwd_params_ema, state.bwd_params_ema, x, mdm_samples,
        state.cycle_weight, state.constraint_weight, state.regularization_weight)
      new_state = state
    
    return new_state, loss, loss_cycle, loss_constraint, loss_reg, x_fwd, x_fwdbwd, y, y_bwd, stds
  
  return step_fn


def get_score_loss_fn(sde, model, train, reduce_mean=False, continuous=True,
                      likelihood_weighting=False, eps=1e-3):
  if continuous:
    loss_fn = score_losses.get_sde_loss_fn(
      sde, model, train, reduce_mean=reduce_mean,
      continuous=True, likelihood_weighting=likelihood_weighting, eps=eps)
  else:
    assert not likelihood_weighting, 'Likelihood weighting is not supported for original SMLD/DDPM training.'
    if isinstance(sde, VESDE):
      loss_fn = score_losses.get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = score_losses.get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f'Discrete training for {sde.__class__.__name__} is not recommended.')
  return loss_fn


def get_score_step_fn(sde, model, optimizer, namm, namm_state, train,
                      optimize_fn=None, reduce_mean=False, continuous=True,
                      likelihood_weighting=False, eps=1e-3):
  """Create a one-step training/evaluation function for score model."""
  loss_fn = get_score_loss_fn(
    sde, model, train, reduce_mean, continuous, likelihood_weighting, eps)
  grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.
  
    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.
  
    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.
  
    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """
    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)

    if train:
      params = state.params
      states = state.model_state

      # Apply mirror map.
      rng, dropout_rng = jax.random.split(rng)  # shouldn't matter bc `train=False`

      y = namm.forward(
        {'dropout': dropout_rng}, namm_state.fwd_params_ema, batch, train=False)

      # Compute loss.
      (loss, new_model_state), grad = grad_fn(step_rng, params, states, y)
      grad = jax.lax.pmean(grad, axis_name='batch')

      # Apply updates.
      new_params, new_opt_state = optimize_fn(state, grad, optimizer)
      new_params_ema = jax.tree_map(
          lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
          params, new_params)
      step = state.step + 1
      new_state = state.replace(
          step=step,
          opt_state=new_opt_state,
          model_state=new_model_state,
          params=new_params,
          params_ema=new_params_ema)
      
      loss = jax.lax.pmean(loss, axis_name='batch')
      new_carry_state = (rng, new_state)
      return new_carry_state, loss, y
    else:
      # Apply mirror map.
      rng, dropout_rng = jax.random.split(rng)  # shouldn't matter bc `train=False`

      y = namm.forward(
        {'dropout': dropout_rng}, namm_state.fwd_params_ema, batch, train=False)

      loss, _ = loss_fn(step_rng, state.params, state.model_state, y)
      return loss

  return step_fn


def get_mdm_sampling_fn(namm, sde_sampling_fn, apply_inverse=True):
    
  def mdm_sampling_fn(rng, score_state):
    # Sample from MDM.
    rng, sample_rng = jax.random.split(rng)
    samples, _ = sde_sampling_fn(sample_rng, score_state)
    return samples

  def sampling_fn(rng, score_state, namm_state):
    # Sample from MDM.
    rng, sample_rng = jax.random.split(rng)
    samples, _ = sde_sampling_fn(sample_rng, score_state)

    # Apply inverse mirror map to samples.
    rng, dropout_rng = jax.random.split(rng)

    # NOTE: This does not use the EMA parameters.
    inverse_samples = namm.backward(
      {'dropout': dropout_rng}, namm_state.bwd_params, samples, train=False)
    
    return samples, inverse_samples

  if apply_inverse:
    return sampling_fn
  else:
    return mdm_sampling_fn