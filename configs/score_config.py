"""Default config file."""

import ml_collections


def get_config():
  """Returns the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  # training
  # need to define: training.n_iters, training.batch_size,
  # training.snapshot_freq, training.log_freq, training.eval_freq
  config.training = training = ml_collections.ConfigDict()
  training.sde = 'vpsde'
  training.likelihood_weighting = False
  training.importance_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.smallest_time = 1e-3
  training.n_epochs = 1000
  training.batch_size = 64
  training.log_freq = 100
  training.snapshot_epoch_freq = 50
  training.ckpt_epoch_freq = 50

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.17

  # data
  # need to define: data.dataset, data.image_size, data.num_channels
  config.data = data = ml_collections.ConfigDict()
  data.random_flip = False
  data.random_rotation = False
  data.random_zoom = False
  data.uniform_dequantization = False
  data.centered = False
  data.dataset = ''
  data.tfds_dir = './data'
  data.height = 32
  data.width = 32
  data.num_channels = 1
  data.antialias = True
  data.constant_flux = False  # whether to scale images to have the same total flux
  data.total_flux = 120.  # total flux if `constant_flux` is True
  data.num_kolmogorov_states = 8
  data.num_kolmogorov_states_per_row = 4
  data.kolmogorov_representation = 'image'  # image | volume

  # constraint
  config.constraint = constraint = ml_collections.ConfigDict()
  constraint.type = 'flux'
  constraint.total_flux = 120.  # for 64x64
  constraint.reynolds = 1000.
  constraint.inner_steps = 20
  constraint.kolmogorov_dt = 0.01
  constraint.kolmogorov_forcing = True
  constraint.kolmogorov_t0 = 3
  constraint.burgers_t0 = 0
  constraint.burgers_dt = 0.025
  constraint.burgers_inner_steps = 5
  constraint.counting_cnn_ckpt_path = './checkpoints/counting_cnn'

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ncsnpp'
  model.num_scales = 1000
  model.sigma_min = 0.002
  model.sigma_max = 50.
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 64  # 128
  model.ch_mult = (1, 2, 2, 2)  # (1, 1, 2, 2, 4, 4)
  model.num_res_blocks = 4  # 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.embedding_type = 'positional'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.interpolation = 'bilinear'  # NCSNv2

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  # evaluation
  config.eval = evaluation = ml_collections.ConfigDict()
  evaluation.batch_size = 32

  config.seed = 42

  return config