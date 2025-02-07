"""Implement PDE solvers and PDE functions.

Much code is taken from https://docs.kidger.site/diffrax/examples/nonlinear_heat_pde/.
"""
import abc

import diffrax
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd


def make_image_from_trajectory(trajectory, n_per_row):
  # Assume trajectory is of shape (h, w, nt).
  nt = trajectory.shape[-1]
  n_rows = nt // n_per_row  
  image = jnp.concatenate(tuple([
    jnp.concatenate(tuple([trajectory[:, :, row * n_per_row + col] for col in range(n_per_row)]), axis=1) \
    for row in range(n_rows)]), axis=0)
  return jnp.expand_dims(image, axis=-1)


def make_trajectory_from_image(image, n_per_row):
  # Assume image is of shape (h * n_rows, w * n_per_row, 1).
  size = image.shape[1] // n_per_row
  n_rows = image.shape[0] // size
  volume = jnp.stack(tuple(
    [image[i * size:(i + 1) * size, j * size:(j + 1) * size, 0] \
    for i in range(n_rows) for j in range(n_per_row)]), axis=-1)  # (h, w, nt)
  return volume


def get_vorticity_fn(vmap=True):
  size = 64
  grid = cfd.grids.Grid(
      shape=(size, size),
      domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
  )
  bc = cfd.boundaries.periodic_boundary_conditions(2)

  def get_vorticity(image):
    """Create vorticity version of image.

    Args:
      image: ndarray of shape (h, w, nt * 2).
    """
    # Reshape image into state array of shape (nt, h, w, 2).
    nt = image.shape[-1] // 2
    v_images = jnp.stack(tuple(
      [image[:, :, 2 * i:2 * (i + 1)] for i in range(nt)]))
    vorticity = jnp.zeros((image.shape[0], image.shape[1], nt))
    for i, state in enumerate(v_images):
      ustate = state[:, :, 0]
      vstate = state[:, :, 1]
      v = cfd.initial_conditions.wrap_variables(
        var=(ustate, vstate),
        grid=grid,
        bcs=(bc, bc))
      vorticity = vorticity.at[:, :, i].set(cfd.finite_differences.curl_2d(v).data)
    return vorticity

  if vmap:
    return jax.vmap(get_vorticity)
  return get_vorticity


def get_vorticity_image_fn(vmap=True):
  size = 64
  grid = cfd.grids.Grid(
      shape=(size, size),
      domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
  )
  bc = cfd.boundaries.periodic_boundary_conditions(2)

  def get_vorticity(image):
    """Create vorticity version of image.

    Args:
      image: ndarray of shape (h, w, 2).
    """
    # Reshape image into state array of shape (nt, h, w, 2).
    n_rows = image.shape[0] // size
    n_per_row = image.shape[1] // size
    v_images = jnp.stack(tuple(
      [image[i * size:(i + 1) * size, j * size:(j + 1) * size] \
      for i in range(n_rows) for j in range(n_per_row)]))
    vorticity = jnp.zeros((v_images.shape[:3]))
    for i, state in enumerate(v_images):
      ustate = state[:, :, 0]
      vstate = state[:, :, 1]
      v = cfd.initial_conditions.wrap_variables(
        var=(ustate, vstate),
        grid=grid,
        bcs=(bc, bc))
      vorticity = vorticity.at[i].set(cfd.finite_differences.curl_2d(v).data)
    
    vorimage = jnp.zeros((image.shape[0], image.shape[1]))
    for i in range(n_rows):
      for j in range(n_per_row):
        row_idxs = slice(i * size, (i + 1) * size)
        col_idxs = slice(j * size, (j + 1) * size)
        vorimage = vorimage.at[row_idxs, col_idxs].set(vorticity[i * n_per_row + j])
    return jnp.expand_dims(vorimage, axis=-1)  # (h, w, 1)

  if vmap:
    return jax.vmap(get_vorticity)
  return get_vorticity


class CrankNicolson(diffrax.AbstractSolver):
  """Custom solver for Crank-Nicolson method."""
  rtol: float
  atol: float

  term_structure = diffrax.ODETerm
  interpolation_cls = diffrax.LocalLinearInterpolation

  def order(self, terms):
    return 2

  def init(self, terms, t0, t1, y0, args):
    return None

  def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
    del solver_state, made_jump
    δt = t1 - t0
    f0 = terms.vf(t0, y0, args)

    @jax.jit
    def fixed_point_iteration(val):
      y1, _ = val
      new_y1 = y0 + 0.5 * δt * (f0 + terms.vf(t1, y1, args))
      diff = jnp.abs(new_y1 - y1)
      max_y1 = jnp.maximum(jnp.abs(y1), jnp.abs(new_y1))
      scale = self.atol + self.rtol * max_y1
      not_converged = jnp.any(diff > scale)
      return new_y1, not_converged

    euler_y1 = y0 + δt * f0
    y1 = euler_y1
    not_converged = False
    while not_converged:
      y1, not_converged = fixed_point_iteration((euler_y1, not_converged))

    y_error = y1 - euler_y1
    dense_info = dict(y0=y0, y1=y1)

    solver_state = None
    result = diffrax.RESULTS.successful
    return y1, y_error, dense_info, solver_state, result

  def func(self, terms, t0, y0, args):
    return terms.vf(t0, y0, args)
  

class PDE(abc.ABC):
  """PDE module with certain discretization."""
  def __init__(self,
               x0: float, x1: float, nx: int,
               t0: float, t1: float, dt: float, nt: int,
               solver: diffrax.AbstractSolver = CrankNicolson(rtol=1e-10, atol=1e-10),
               stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
                 pcoeff=0.3, icoeff=0.4, rtol=1e-10, atol=1e-10, dtmax=0.001)):
    # Spatial discretization
    self.x0 = x0
    self.x1 = x1
    self.nx = nx
    self.xs = jnp.linspace(x0, x1, nx)
    self.dx = self.xs[1] - self.xs[0]
    # Temporal discretization
    self.t0 = t0
    self.t1 = t1
    self.dt = dt  # used in `diffrax.diffeqsolve`
    self.nt = nt  # used for `diffrax.SaveAt`
    self.ts = jnp.linspace(t0, t1, nt)
    self.saveat = diffrax.SaveAt(ts=self.ts)
    # Solver
    self.solver = solver
    self.stepsize_controller = stepsize_controller

  @abc.abstractmethod
  def dydt(self, t: float, y: jnp.ndarray) -> jnp.ndarray:
    """The time derivative of y."""

  def gradient(self, y: jnp.ndarray) -> jnp.ndarray:
    """Approximate derivative with central difference."""
    y_next = jnp.roll(y, shift=-1)
    y_prev = jnp.roll(y, shift=1)
    dy = (y_next - y_prev) / (2 * self.dx)
    return dy

  def laplacian(self, y: jnp.ndarray, zero_bc: bool = False) -> jnp.ndarray:
    """Approximate second derivative with central difference."""
    y_next = jnp.roll(y, shift=-1)
    y_prev = jnp.roll(y, shift=1)
    Δy = (y_next - 2 * y + y_prev) / (self.dx**2)
    if zero_bc:
      Δy = Δy.at[0].set(0)
      Δy = Δy.at[-1].set(0)
    return Δy

  def solve(self, y0: jnp.ndarray):
    """Solve the PDE with the given initial condition."""
    sol = diffrax.diffeqsolve(
      diffrax.ODETerm(self.dydt),
      self.solver,
      self.t0,
      self.t1,
      self.dt,
      y0,
      saveat=self.saveat,
      stepsize_controller=self.stepsize_controller,
      max_steps=8192
    )
    return sol


class NonlinearHeatEquation(PDE):
  def dydt(self, t, y, args):
    return (1 - y) * self.laplacian(y, True)


class BurgersEquation(PDE):
  def __init__(self, mu, nu, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.mu = mu
    self.nu = nu

  def dydt(self, t, y, args):
    dy = -self.mu * y * self.gradient(y) + self.nu * self.laplacian(y, False)
    return dy


def get_burgers_residual_fn(mu=1, nu=0.05, x0=0, x1=10, nx=64, t0=0, dt=0.025, inner_steps=5, nt=64):
  """Define a function that gives the residual of an image from the 
  1D Burgers' constraint. In particular, for each state (column), we compute
  the error from the predicted state given by the 1D Burgers' PDE."""
  t1 = t0 + dt * inner_steps * (nt - 1)

  solver = CrankNicolson(rtol=1e-3, atol=1e-3)
  stepsize_controller = diffrax.ConstantStepSize()
  pde = BurgersEquation(
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
  def residual_fn(x):
    # Assume flipped images of shape (nx, nt, 1).
    x = jnp.flipud(x[:, :, 0])
    xnext = jax.vmap(step, in_axes=(1, 0), out_axes=1)(x[:, :-1], ts)
    residual = x[:, 1:] - xnext
    return residual

  return residual_fn


def get_incompress_residual_fn(reynolds=1000., density=1, size=64, dt=0.01, inner_steps=20):
  grid = cfd.grids.Grid(
    shape=(size, size),
    domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
  )
  bc = cfd.boundaries.periodic_boundary_conditions(2)
  forcing = cfd.forcings.simple_turbulence_forcing(
    grid=grid,
    constant_magnitude=1.,
    constant_wavenumber=4.,
    linear_coefficient=-0.1,
    forcing_type='kolmogorov',
  )

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
  def residual_fn(image):
    # Assume image of shape (h * n_rows, w * n_per_row, 2).
    # Reshape to (nt, h, w, 2).
    # nt = image.shape[-1] // 2
    # v_images = jnp.stack(tuple(
    #   [image[:, :, 2 * i:2 * (i + 1)] for i in range(nt)]))
    n_per_row = image.shape[1] // size
    v_images = jnp.stack(
      (jnp.moveaxis(make_trajectory_from_image(image[:, :, 0:1], n_per_row), -1, 0),
       jnp.moveaxis(make_trajectory_from_image(image[:, :, 1:2], n_per_row), -1, 0),), axis=-1)

    vnext_images = jax.vmap(one_step)(v_images[:-1])
    residual = v_images[1:] - vnext_images
    return residual
  
  return residual_fn