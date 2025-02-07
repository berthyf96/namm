import ml_collections

def get_config():
  config = ml_collections.ConfigDict()

  data = config.data = ml_collections.ConfigDict()
  data.height = 32
  data.width = 32
  data.num_channels = 1
  data.dataset = ''
  data.random_flip = False
  data.random_rotation = False
  data.random_zoom = False
  data.uniform_dequantization = False
  data.centered = False
  data.dataset = ''
  data.tfds_dir = './data'
  data.antialias = True
  data.constant_flux = False
  data.total_flux = 120.
  data.num_kolmogorov_states = 8
  data.num_kolmogorov_states_per_row = 4
  data.kolmogorov_representation = 'image'  # image | volume

  constraint = config.constraint = ml_collections.ConfigDict()
  constraint.type = 'flux'
  constraint.total_flux = 120.
  constraint.reynolds = 1000.
  constraint.inner_steps = 20
  constraint.kolmogorov_dt = 0.01
  constraint.kolmogorov_forcing = True
  constraint.kolmogorov_t0 = 3
  constraint.burgers_t0 = 0
  constraint.burgers_dt = 0.025
  constraint.burgers_inner_steps = 5
  constraint.counting_cnn_ckpt_path = './checkpoints/counting_cnn'

  model = config.model = ml_collections.ConfigDict()
  model.fwd_n_filters = 64
  model.bwd_n_filters = 64
  model.n_res_blocks = 6
  model.dropout_rate = 0.5
  model.n_downsample_layers = 2
  model.upsample_mode = 'deconv'
  # model.residual = False  # if `True`, ResNet estimates residual
  model.fwd_residual = True
  model.bwd_residual = False
  model.fwd_network = 'icnn'
  model.bwd_network = 'resnet'
  model.fwd_activation = 'none'
  model.bwd_activation = 'softplus'
  model.fwd_strong_convexity = 0.9
  model.bwd_strong_convexity = 0.1
  model.fwd_icnn_n_filters = 32
  model.bwd_icnn_n_filters = 64
  model.fwd_icnn_n_layers = 3
  model.bwd_icnn_n_layers = 5
  model.fwd_icnn_kernel_size = 3
  model.bwd_icnn_kernel_size = 3
  model.ema_rate = 0.999

  optim = config.optim = ml_collections.ConfigDict()
  optim.mdm_finetune = False  # whether to fine-tune with dual samples from MDM instead of perturbed G(x)
  optim.grad_clip = -1.  # negative value means no clipping
  optim.learning_rate = 2e-4
  optim.zero_nans = False
  optim.adam_beta1 = 0.5
  optim.cycle_weight = 1.
  optim.regularization_weight = 1e-3
  optim.dsm_weight = 1e-3
  optim.constraint_weight = 0.001  # max weight if annealing constraint
  optim.regularization = 'sparse_icnn'
  optim.max_sigma = 0.1  # maximum perturbation level (i.e., noise std. dev. or magnitude of diffusion gamma)
  optim.fixed_sigma = False
  optim.divergence_weight = 0.1

  training = config.training = ml_collections.ConfigDict()
  training.batch_size = 16
  training.n_epochs = 50
  training.log_freq = 100
  training.snapshot_epoch_freq = 5
  training.ckpt_epoch_freq = 50

  config.eval = evaluation = ml_collections.ConfigDict()
  evaluation.batch_size = 16

  config.seed = 42
  return config