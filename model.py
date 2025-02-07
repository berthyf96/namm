"""Model definitions for learned mirror map (LMM).

Some code borrowed from JAX CycleGAN: github.com/dlzou/cyclegan-jax
"""
from functools import partial
import logging
from typing import Callable, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp

from icnn import ICNN


class NAMM:
  def __init__(self,
               output_nc,
               fwd_n_filters=64,
               bwd_n_filters=64,
               n_res_blocks=6,
               dropout_rate=0.5,
               n_downsample_layers=2,
               upsample_mode='deconv',
               fwd_residual=False,
               bwd_residual=False,
               fwd_network='resnet',
               bwd_network='resnet',
               fwd_activation=None,
               bwd_activation='relu',
               fwd_icnn_n_filters=64,
               bwd_icnn_n_filters=64,
               fwd_icnn_n_layers=6,
               bwd_icnn_n_layers=6,
               fwd_icnn_kernel_size=3,
               bwd_icnn_kernel_size=3,
               fwd_strong_convexity=0.3,
               bwd_strong_convexity=0.3):
    if fwd_network == 'resnet':
      self.fwd_generator = Generator(
        output_nc=output_nc,
        ngf=fwd_n_filters,
        n_res_blocks=n_res_blocks,
        dropout_rate=dropout_rate,
        n_downsample_layers=n_downsample_layers,
        upsample_mode=upsample_mode,
        final_activation=fwd_activation,
        residual=fwd_residual
      )
    elif fwd_network == 'icnn':
      self.fwd_generator = ICNN(
        n_in_channels=output_nc,
        n_layers=fwd_icnn_n_layers,
        n_filters=fwd_icnn_n_filters,
        kernel_size=fwd_icnn_kernel_size,
        strong_convexity=fwd_strong_convexity,
        negative_slope=0.2
      )

    if bwd_network == 'resnet':
      self.bwd_generator = Generator(
        output_nc=output_nc,
        ngf=bwd_n_filters,
        n_res_blocks=n_res_blocks,
        dropout_rate=dropout_rate,
        n_downsample_layers=n_downsample_layers,
        upsample_mode=upsample_mode,
        final_activation=bwd_activation,
        residual=bwd_residual
      )
    elif bwd_network == 'icnn':
      self.bwd_generator = ICNN(
        n_in_channels=output_nc,
        n_layers=bwd_icnn_n_layers,
        n_filters=bwd_icnn_n_filters,
        kernel_size=bwd_icnn_kernel_size,
        strong_convexity=bwd_strong_convexity,
        negative_slope=0.2
      )


  def get_generator_params(self, rngs, input_shape):
    fwd_params = self.fwd_generator.init(rngs, jnp.ones(input_shape), train=False)['params']
    bwd_params = self.bwd_generator.init(rngs, jnp.ones(input_shape), train=False)['params']
    logging.info('Forward mirror map shape: %s', input_shape)
    logging.info('Backward mirror map shape: %s', input_shape)
    return fwd_params, bwd_params

  def forward(self, rngs, fwd_params, x, train=True):
    y = self.fwd_generator.apply({'params': fwd_params}, x, train=train, rngs=rngs)
    return y

  def backward(self, rngs, bwd_params, y, train=True):
    x = self.bwd_generator.apply(
      {'params': bwd_params},
      y,
      train=train,
      rngs=rngs)
    return x

  def discriminate(self, params, inp):
    out = self.discriminator.apply({'params': params}, inp)
    return out
  
class ResNetBlock(nn.Module):
  features: int
  dropout_layer: Callable
  initializer: Callable = jax.nn.initializers.normal(stddev=0.02)
  activation: str = 'relu'

  @nn.compact
  def __call__(self, x):
    try:
      activation_layer = getattr(nn.activation, self.activation)
    except:
      raise ValueError(f'Generator activation {self.activation} is not recognized')
    model = [
      nn.Conv(
        features=self.features,
        kernel_size=[3, 3],
        padding='SAME',
        kernel_init=self.initializer,
      ),
      nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
      activation_layer,
      self.dropout_layer(),
      nn.Conv(
        features=self.features,
        kernel_size=[3, 3],
        padding='SAME',
        kernel_init=self.initializer,
      ),
      nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
    ]
    seq_model = nn.Sequential(model)
    return x + seq_model(x)
  

class Generator(nn.Module):
  output_nc: int = 3
  ngf: int = 32
  n_res_blocks: int = 6
  dropout_rate: float = 0.5
  upsample_mode: str = 'deconv'
  initializer: Callable = jax.nn.initializers.normal(stddev=0.02)
  n_downsample_layers: int = 2
  activation: str = 'relu'
  final_activation: Optional[str] = None
  residual: bool = False
  is_3d: bool = False

  @nn.compact
  def __call__(self, x, train):
    input_x = x

    try:
      activation_layer = getattr(nn.activation, self.activation)
    except:
      raise ValueError(f'Generator activation {self.activation} is not recognized')

    # first convolution layer
    first_conv = nn.Sequential([
      nn.Conv(
        features=self.ngf,
        kernel_size=[7, 7],
        strides=[1, 1],
        padding='SAME',
        kernel_init=self.initializer,
      ),
      nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
      activation_layer,
    ])
    x = first_conv(x)

    # downsampling layers
    model = []
    for i in range(self.n_downsample_layers):
      mult = 2**i
      model += [
        nn.Conv(
          features=self.ngf * mult * 2,
          kernel_size=[3, 3],
          strides=[2, 2],
          padding='SAME',
          kernel_init=self.initializer,
        ),
        nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
        activation_layer,
      ]
    downsample = nn.Sequential(model)
    x = downsample(x)

    # ResNet transformation blocks
    mult = 2**self.n_downsample_layers
    model = []
    for i in range(self.n_res_blocks):
      model += [
        ResNetBlock(
          features=self.ngf * mult,
          dropout_layer=partial(
            nn.Dropout, rate=self.dropout_rate, deterministic=not train
          ),
          initializer=self.initializer,
          activation=self.activation
        )
      ]
    transform = nn.Sequential(model)
    x = transform(x)

    # upsampling layers
    model = []
    if self.upsample_mode == 'bilinear':
      for i in range(self.n_downsample_layers):
        if self.is_3d:
          resize_shape = (
            x.shape[0],
            x.shape[1] * 2 ** (i + 1),
            x.shape[2] * 2 ** (i + 1),
            x.shape[3] * 2 ** (i + 1),
            x.shape[4])
        else:
          resize_shape = (
            x.shape[0],
            x.shape[1] * 2 ** (i + 1),
            x.shape[2] * 2 ** (i + 1),
            x.shape[3])
        mult = 2 ** (self.n_downsample_layers - i)
        model += [
          partial(
            jax.image.resize,
            shape=resize_shape,
            method='bilinear',
          ),
          nn.Conv(
            features=(self.ngf * mult) // 2,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='SAME',
            kernel_init=self.initializer,
          ),
          nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
          activation_layer,
        ]
    elif self.upsample_mode == 'deconv':
      for i in range(self.n_downsample_layers):
        mult = 2 ** (self.n_downsample_layers - i)
        model += [
          nn.ConvTranspose(
            features=(self.ngf * mult) // 2,
            kernel_size=[3, 3],
            strides=[2, 2],
            padding='SAME',
            kernel_init=self.initializer,
          ),
          nn.GroupNorm(num_groups=None, group_size=1),  # instance norm
          activation_layer,
        ]
    else:
      raise NotImplementedError(f'Generator upsample_mode {self.upsample_mode} is not recognized')

    upsample = nn.Sequential(model)
    x = upsample(x)

    # last convolution layer
    model = [
      nn.Conv(
        features=self.output_nc,
        kernel_size=[7, 7],
        strides=[1, 1],
        padding='SAME',
        kernel_init=self.initializer,
      )
    ]
    last_conv = nn.Sequential(model)
    x = last_conv(x)

    if self.residual:
      x = x + input_x

    if self.final_activation is not None:
      try:
        act = getattr(nn.activation, self.final_activation)
        x = act(x)
      except:
        raise ValueError(f'Generator final_activation {self.final_activation} is not recognized')

    return x

  
class CNN(nn.Module):
  """Counting CNN (borrowed from https://github.com/8bitmp3/JAX-Flax-Tutorial-Image-Classification-with-Linen)."""
  @nn.compact
  # Provide a constructor to register a new parameter 
  # and return its initial value
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1)) # Flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1)(x)
    return x