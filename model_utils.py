"""Model utils for learned mirror map (LMM)."""
import flax
import jax
from jaxtyping import PyTree
import optax

from model import NAMM


@flax.struct.dataclass
class State:
  """CycleGAN training state."""
  step: int
  epoch: int
  rng: jax.Array
  fwd_opt_state: optax.OptState
  bwd_opt_state: optax.OptState
  ema_rate: float
  fwd_params: PyTree
  bwd_params: PyTree
  fwd_params_ema: PyTree
  bwd_params_ema: PyTree
  cycle_weight: float
  constraint_weight: float
  regularization_weight: float
  dsm_weight: float = 0.


def get_model(config):
  fwd_act = None if config.model.fwd_activation == 'none' else config.model.fwd_activation
  bwd_act = None if config.model.bwd_activation == 'none' else config.model.bwd_activation
  return NAMM(
    output_nc=config.data.num_channels,
    fwd_n_filters=config.model.fwd_n_filters,
    bwd_n_filters=config.model.bwd_n_filters,
    n_res_blocks=config.model.n_res_blocks,
    dropout_rate=config.model.dropout_rate,
    n_downsample_layers=config.model.n_downsample_layers,
    upsample_mode=config.model.upsample_mode,
    fwd_residual=config.model.fwd_residual,
    bwd_residual=config.model.bwd_residual,
    fwd_network=config.model.fwd_network,
    bwd_network=config.model.bwd_network,
    fwd_activation=fwd_act,
    bwd_activation=bwd_act,
    fwd_icnn_n_filters=config.model.fwd_icnn_n_filters,
    bwd_icnn_n_filters=config.model.bwd_icnn_n_filters,
    fwd_icnn_n_layers=config.model.fwd_icnn_n_layers,
    bwd_icnn_n_layers=config.model.bwd_icnn_n_layers,
    fwd_icnn_kernel_size=config.model.fwd_icnn_kernel_size,
    bwd_icnn_kernel_size=config.model.bwd_icnn_kernel_size,
    fwd_strong_convexity=config.model.fwd_strong_convexity,
    bwd_strong_convexity=config.model.bwd_strong_convexity)


def init_state(config, model):
  input_shape = (
    config.training.batch_size // jax.local_device_count(),
    config.data.height, config.data.width, config.data.num_channels)

  # Initialize forward and backward generator params.
  rng = jax.random.PRNGKey(config.seed)
  rng, params_rng = jax.random.split(rng)
  rng, dropout_rng = jax.random.split(rng)
  fwd_params, bwd_params = model.get_generator_params(
    {'params': params_rng, 'dropout': dropout_rng}, input_shape)
  # Create training state.
  state = State(
    step=0,
    epoch=0,
    rng=rng,
    ema_rate=config.model.ema_rate,
    fwd_opt_state=None,
    bwd_opt_state=None,
    fwd_params=fwd_params,
    bwd_params=bwd_params,
    fwd_params_ema=fwd_params,
    bwd_params_ema=bwd_params,
    cycle_weight=config.optim.cycle_weight,
    constraint_weight=config.optim.constraint_weight,
    regularization_weight=config.optim.regularization_weight,
    dsm_weight=config.optim.dsm_weight,)
  return state


def init_optimizer(config, state):
  fwd_tx_chain, bwd_tx_chain = [], []

  # Add gradient clipping if specified.
  if config.optim.grad_clip > 0:
    fwd_tx_chain.append(optax.clip_by_global_norm(config.optim.grad_clip))
    bwd_tx_chain.append(optax.clip_by_global_norm(config.optim.grad_clip))

  # Add Adam optimizer.
  lr = config.optim.learning_rate
  fwd_tx_chain.append(optax.adam(lr, b1=config.optim.adam_beta1))
  bwd_tx_chain.append(optax.adam(lr, b1=config.optim.adam_beta1))

  # Make certain weights non-negative for ICNN.
  if config.model.fwd_network == 'icnn':
    is_nonneg_mask = {}
    for param in state.fwd_params.keys():
      is_nonneg_mask[param] = ('wz' in param or 'final_conv2d' in param)
    fwd_tx_chain.append(optax.masked(optax.keep_params_nonnegative(), is_nonneg_mask))
  if config.model.bwd_network == 'icnn':
    is_nonneg_mask = {}
    for param in state.bwd_params.keys():
      is_nonneg_mask[param] = ('wz' in param or 'final_conv2d' in param)
    bwd_tx_chain.append(optax.masked(optax.keep_params_nonnegative(), is_nonneg_mask))

  if config.optim.zero_nans:
    fwd_tx_chain.append(optax.zero_nans())
    bwd_tx_chain.append(optax.zero_nans())

  fwd_tx = optax.chain(*fwd_tx_chain)
  bwd_tx = optax.chain(*bwd_tx_chain)

  fwd_opt_state = fwd_tx.init(state.fwd_params)
  bwd_opt_state = bwd_tx.init(state.bwd_params)
  state = state.replace(fwd_opt_state=fwd_opt_state, bwd_opt_state=bwd_opt_state)

  return state, fwd_tx, bwd_tx