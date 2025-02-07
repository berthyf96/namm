"""Implementation of input-convex neural networks (ICNN).

Source: https://github.com/hyt35/icnn-md
"""
from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp

def get_nonneg_initializer(min_val=0., max_val=0.001):
  def initializer(rng, shape, dtype=jnp.float32):
    return min_val + (max_val - min_val) * jax.random.uniform(rng, shape, dtype)
  return initializer


class ICNN(nn.Module):
  n_in_channels: int = 1
  n_layers: int = 10
  n_filters: int = 64
  kernel_size: int = 5
  strong_convexity: float = 0.5
  negative_slope: float = 0.2  # slope of leaky ReLU

  def setup(self):
    self.padding = (self.kernel_size - 1) // 2
    self.nonneg_init = get_nonneg_initializer()

    # These layers should have non-negative weights.
    self.wz = [nn.Conv(
      features=self.n_filters,
      kernel_size=[self.kernel_size, self.kernel_size],
      strides=1,
      padding='CIRCULAR',
      use_bias=False,
      kernel_init=self.nonneg_init
    ) for _ in range(self.n_layers)]

    # These layers can have arbitrary weights.
    self.wx_quad = [nn.Conv(
      features=self.n_filters,
      kernel_size=[self.kernel_size, self.kernel_size],
      strides=1,
      padding='CIRCULAR',
      use_bias=False
    ) for _  in range(self.n_layers + 1)]
    self.wx_lin = [nn.Conv(
      features=self.n_filters,
      kernel_size=[self.kernel_size, self.kernel_size],
      strides=1,
      padding='CIRCULAR',
      use_bias=True
    ) for _ in range(self.n_layers + 1)]

    # One final conv layer with nonnegative weights.
    self.final_conv2d = nn.Conv(
      features=self.n_in_channels,
      kernel_size=[self.kernel_size, self.kernel_size],
      strides=1,
      padding='CIRCULAR',
      use_bias=False,
      kernel_init=self.nonneg_init
    )
  
  @partial(jax.vmap, in_axes=(None, 0))
  @partial(jax.grad, argnums=1)
  def scalar(self, x):
    z = nn.activation.leaky_relu(
      self.wx_quad[0](x[None, ...])**2 + self.wx_lin[0](x[None, ...]),
      negative_slope=self.negative_slope)
    for layer in range(self.n_layers):
      z = nn.activation.leaky_relu(
        self.wz[layer](z) + self.wx_quad[layer+1](x[None, ...])**2 + self.wx_lin[layer+1](x[None, ...]),
        negative_slope=self.negative_slope)
    z = self.final_conv2d(z)[0]  # (H, W, C)
    z_avg = jnp.sum(jnp.mean(z, axis=(1, 2)))
    return z_avg
  
  # @partial(jax.vmap, in_axes=(None, 0))
  # @partial(jax.grad, argnums=1)
  # def scalar_previous(self, x):
  #   z = nn.activation.leaky_relu(
  #     self.wx_quad[0](x[None, ...])**2 + self.wx_lin[0](x[None, ...]),
  #     negative_slope=self.negative_slope)
  #   for layer in range(self.n_layers):
  #     z = nn.activation.leaky_relu(
  #       self.wz[layer](z) + self.wx_quad[layer+1](x[None, ...])**2 + self.wx_lin[layer+1](x[None, ...]),
  #       negative_slope=self.negative_slope)
  #   z = self.final_conv2d(z)[0]  # (H, W, C)
  #   z_avg = jnp.mean(z)
  #   return z_avg
  

  @nn.compact
  def __call__(self, x, train=True):
    grad = self.scalar(x)
    return (1 - self.strong_convexity) * grad + self.strong_convexity * x