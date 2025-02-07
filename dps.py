from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def get_data_fit_loss(y, forward_model, sigma, constraint_weight=0, constraint_losses_fn=None):
  if constraint_weight > 0 and constraint_losses_fn is None:
    raise ValueError('constraint_losses_fn must be provided if constraint_weight > 0')

  def data_fit_loss(x):
    """Compute data fit loss for DPS.

    Note that this doesn't equal the usual unnormalized -logp(y|x).
    For example, official DPS experiments divide loss by `norm2(residual)`.
    
    Args:
      x: images, ndarray of shape (b, h, w, c).

    Returns: data-fit loss of each image, ndarray of shape (b,).
    """
    y_pred = forward_model(x)
    residual = y[None, ...] - y_pred
    data_loss = jnp.linalg.norm(residual.reshape(x.shape[0], -1) / sigma, axis=-1)
    if constraint_weight != 0:
      data_loss += constraint_weight * jnp.mean(constraint_losses_fn(x))
    return data_loss
  
  return data_fit_loss


def get_marginal_dist_params_fn(sde):
  def marginal_dist_params(t):
    all_ones = jnp.ones((1, 1))
    t_batch = jnp.ones(1) * t
    mean, std = sde.marginal_prob(all_ones, t_batch)
    alpha_t = mean[0][0]
    beta_t = std[0]
    return alpha_t, beta_t
  return marginal_dist_params


def get_dps_params(sde):
  alphas_cumprod = jnp.cumprod(sde.alphas, axis=0)
  alphas_cumprod_prev = jnp.append(1.0, alphas_cumprod[:-1])
  posterior_mean_coef1 = (
      sde.discrete_betas * jnp.sqrt(alphas_cumprod_prev) /
      (1.0 - alphas_cumprod)
  )
  posterior_mean_coef2 = (
      (1.0 - alphas_cumprod_prev) * jnp.sqrt(sde.alphas) /
      (1.0 - alphas_cumprod)
  )

  posterior_variance = (
      sde.discrete_betas * (1.0 - alphas_cumprod_prev) /
      (1.0 - alphas_cumprod)
  )
  model_variance = jnp.append(posterior_variance[1], sde.discrete_betas[1:])
  model_log_variance = jnp.log(model_variance)
  return posterior_mean_coef1, posterior_mean_coef2, model_log_variance


def get_valid_samples(samples):
  """Filter out any samples that have NaNs in them."""
  valid_samples = []
  for sample in samples:
    if not np.isnan(sample).any():
      valid_samples.append(sample)
  return np.array(valid_samples)


def get_mirror_dps_sampler(y, forward_model, sigma, sde, score_fn,
                           inverse_mirror_fn, shape,
                           inverse_scaler=lambda x: x, eps=1e-3):
  posterior_mean_coef1, posterior_mean_coef2, model_log_variance = get_dps_params(sde)
  marginal_dist_params = get_marginal_dist_params_fn(sde)
  data_fit_loss = get_data_fit_loss(y, forward_model, sigma, constraint_weight=0)

  @jax.vmap
  @partial(jax.value_and_grad, has_aux=True)
  def val_and_grad_fn(x, t):
    score = score_fn(x[None, :], jnp.ones(1) * t)[0]

    # Predict x_0 | x_t in dual space.
    alpha_t, beta_t = marginal_dist_params(t)
    x0_hat = (x + beta_t**2 * score) / alpha_t

    # Get likelihood score.
    x0_hat_primal = inverse_mirror_fn(x0_hat[None, ...])[0]
    neg_log_llh = data_fit_loss(x0_hat_primal[None, ...])[0]

    return neg_log_llh, (x0_hat,)
  
  def step_fn(rng, xt, t_batch, t_idx, scale):
    # Predict x_0 | x_t.
    (_, (x0_hat,)), gradient = val_and_grad_fn(xt, t_batch)

    # Sample from q(x_{t-1} | x_t, x_0).
    coef1 = posterior_mean_coef1[t_idx]
    coef2 = posterior_mean_coef2[t_idx]
    xt_prime = coef1 * x0_hat + coef2 * xt

    log_variance = model_log_variance[t_idx]
    noise = jax.random.normal(rng, xt.shape)
    xt_prime += jnp.exp(0.5 * log_variance) * noise

    # Apply gradient.
    xt = xt_prime - scale * gradient
    return xt

  @partial(jax.pmap, in_axes=(0, None))
  @jax.jit
  def dps_sampler(rng, scale):
    timesteps = jnp.linspace(sde.T, eps, sde.N)

    # Initial sample.
    rng, step_rng = jax.random.split(rng)
    x = sde.prior_sampling(step_rng, shape)

    def loop_body(carry, i):
      rng, x = carry
      t = timesteps[i]
      idx = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      vec_t = jnp.ones(shape[0]) * t

      rng, step_rng = jax.random.split(rng)
      x = step_fn(step_rng, x, vec_t, idx, scale)

      return (rng, x), x

    _, all_samples = jax.lax.scan(
        loop_body, (rng, x), jnp.arange(0, sde.N), length=sde.N)
    output = all_samples[-1]

    return inverse_scaler(output)
  
  return dps_sampler


def get_dps_sampler(y, forward_model, sigma, sde, score_fn, shape,
                    constraint_weight=0, constraint_losses_fn=None,
                    inverse_scaler=lambda x: x, eps=1e-3):
  posterior_mean_coef1, posterior_mean_coef2, model_log_variance = get_dps_params(sde)
  marginal_dist_params = get_marginal_dist_params_fn(sde)
  data_fit_loss = get_data_fit_loss(y, forward_model, sigma, constraint_weight, constraint_losses_fn)

  @jax.vmap
  @partial(jax.value_and_grad, has_aux=True)
  def val_and_grad_fn(x, t):
    score = score_fn(x[None, :], jnp.ones(1) * t)[0]
  
    # Predict x_0 | x_t.
    alpha_t, beta_t = marginal_dist_params(t)
    x0_hat = (x + beta_t**2 * score) / alpha_t
  
    # Get likelihood score.
    neg_log_llh = data_fit_loss(x0_hat[None, ...])[0]
    
    return neg_log_llh, (x0_hat,)

  
  def step_fn(rng, xt, t_batch, t_idx, scale):
    # Predict x_0 | x_t.
    (_, (x0_hat,)), gradient = val_and_grad_fn(xt, t_batch)
  
    # Sample from q(x_{t-1} | x_t, x_0).
    coef1 = posterior_mean_coef1[t_idx]
    coef2 = posterior_mean_coef2[t_idx]
    xt_prime = coef1 * x0_hat + coef2 * xt
  
    log_variance = model_log_variance[t_idx]
    noise = jax.random.normal(rng, xt.shape)
    xt_prime += jnp.exp(0.5 * log_variance) * noise
  
    # Apply gradient.
    xt = xt_prime - scale * gradient
    return xt

  
  @partial(jax.pmap, in_axes=(0, None))
  def dps_sampler(rng, scale):
    timesteps = jnp.linspace(sde.T, eps, sde.N)
  
    # Initial sample.
    rng, step_rng = jax.random.split(rng)
    x = sde.prior_sampling(step_rng, shape)
  
    def loop_body(carry, i):
      rng, x = carry
      t = timesteps[i]
      idx = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      vec_t = jnp.ones(shape[0]) * t
  
      rng, step_rng = jax.random.split(rng)
      x = step_fn(step_rng, x, vec_t, idx, scale)
  
      return (rng, x), x
  
    _, all_samples = jax.lax.scan(
      loop_body, (rng, x), jnp.arange(0, sde.N), length=sde.N)
    output = all_samples[-1]
  
    return inverse_scaler(output)
  
  return dps_sampler