"""Code to generate figure 1."""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import plotting
import scipy.integrate
import tornadox

import pnmol


def solve_pde_pnmol_white(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.white.LinearWhiteNoiseEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )
    sol = ek1.solve(pde, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    means, stds = read_mean_and_std(sol, E0)
    gamma = jnp.sqrt(sol.diffusion_squared_calibrated)
    print(gamma)
    return means, gamma * stds, sol.t, pde.mesh_spatial.points


def solve_pde_pnmol_latent(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.latent.LinearLatentForceEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )
    sol = ek1.solve(pde, progressbar=progressbar)
    E0 = ek1.state_iwp.projection_matrix(0)
    means, stds = read_mean_and_std_latent(sol, E0)
    gamma = jnp.sqrt(sol.diffusion_squared_calibrated)
    print(gamma)
    return means, gamma * stds, sol.t, pde.mesh_spatial.points


def solve_pde_tornadox(pde, *, dt, nu, progressbar):
    steprule = tornadox.step.ConstantSteps(dt)
    ivp = pde.to_tornadox_ivp()
    ek1 = tornadox.ek1.ReferenceEK1ConstantDiffusion(
        num_derivatives=nu,
        steprule=steprule,
        initialization=tornadox.init.Stack(use_df=False),
    )
    sol, gamma = ek1.solve(ivp, progressbar=progressbar)
    E0 = ek1.P0
    means, stds = read_mean_and_std(sol, E0)
    gamma = 1.0  # ???

    return means, gamma * stds, sol.t, pde.mesh_spatial.points


def solve_pde_reference(pde, *, dt, high_res_factor_dx, high_res_factor_dt):
    t_eval = jnp.arange(pde.t0, pde.tmax, step=dt)
    ivp = pde.to_tornadox_ivp()
    sol = scipy.integrate.solve_ivp(ivp.f, ivp.t_span, ivp.y0, t_eval=t_eval)

    means = sol.y.T
    stds = 0.0 * sol.y.T
    ts = t_eval[::high_res_factor_dt]

    means = jnp.pad(means, pad_width=1, mode="constant", constant_values=0.0)[
        1:-1, ...
    ][::high_res_factor_dt, ::high_res_factor_dx]
    stds = jnp.pad(stds, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...][
        ::high_res_factor_dt, ::high_res_factor_dx
    ]

    return means, stds, ts, pde.mesh_spatial.points[::high_res_factor_dx]


def read_mean_and_std(sol, E0):
    means = sol.mean[:, 0]
    cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0, 2, 1))
    stds = jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2) @ E0.T)
    return means, stds


def read_mean_and_std_latent(sol, E0):
    means = jnp.split(sol.mean, 2, axis=-1)[0]
    means = means[:, 0, :]
    cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0, 2, 1))
    vars = jnp.diagonal(cov, axis1=1, axis2=2)
    stds = jnp.sqrt(jnp.split(vars, 2, axis=-1)[0] @ E0.T)
    return means, stds


def save_result(result, /, *, prefix, path="experiments/results"):
    path = pathlib.Path(path) / "figure1"
    if not path.is_dir():
        path.mkdir(parents=True)

    means, stds, ts, xs = result
    path_means = path / (prefix + "_means.npy")
    path_stds = path / (prefix + "_stds.npy")
    path_ts = path / (prefix + "_ts.npy")
    path_xs = path / (prefix + "_xs.npy")
    jnp.save(path_means, means)
    jnp.save(path_stds, stds)
    jnp.save(path_ts, ts)
    jnp.save(path_xs, xs)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="A script that processes command line parameters for hyperparameters."
    )

    # Add arguments with default values
    parser.add_argument(
        "--PDE",
        type=str,
        default="burgers",
        help="PDE to solve, either 'burgers' or 'heat' (default: 'burgers')",
    )
    parser.add_argument(
        "--DT", type=float, default=0.05, help="Time step size (default: 0.05)"
    )
    parser.add_argument(
        "--DX", type=float, default=0.2, help="Spatial step size (default: 0.2)"
    )
    parser.add_argument(
        "--HIGH_RES_FACTOR_DX",
        type=int,
        default=12,
        help="High resolution factor for DX (default: 12)",
    )
    parser.add_argument(
        "--HIGH_RES_FACTOR_DT",
        type=int,
        default=8,
        help="High resolution factor for DT (default: 8)",
    )
    parser.add_argument(
        "--NUM_DERIVATIVES",
        type=int,
        default=2,
        help="Number of derivatives (default: 2)",
    )
    parser.add_argument(
        "--NUGGET_COV_FD",
        type=float,
        default=1e-12,
        help="Nugget covariance for finite differences (default: 1e-12)",
    )
    parser.add_argument(
        "--STENCIL_SIZE", type=int, default=3, help="Stencil size (default: 3)"
    )
    parser.add_argument(
        "--PROGRESSBAR", action="store_true", help="Show progress bar (default: True)"
    )
    parser.add_argument(
        "--no-PROGRESSBAR",
        dest="PROGRESSBAR",
        action="store_false",
        help="Do not show progress bar",
    )
    parser.set_defaults(PROGRESSBAR=True)

    parser.add_argument(
        "--INPUT_SCALE",
        type=float,
        default=3.2,
        help="Input scale for the kernel (default: 3.2)",
    )
    parser.add_argument(
        "--OUTPUT_SCALE",
        type=float,
        default=0.1,
        help="Output scale for the kernel (default: 0.1)",
    )

    parser.add_argument(
        "--T0", type=float, default=0.0, help="Initial time (default: 0.0)"
    )
    parser.add_argument(
        "--TMAX", type=float, default=3.0, help="Maximum time (default: 3.0)"
    )
    parser.add_argument(
        "--DIFFUSION_RATE",
        type=float,
        default=0.025,  # 0.035
        help="Diffusion rate (default: 0.025)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # PDE function
    if args.PDE == "burgers":
        pde_fun = pnmol.pde.examples.burgers_1d_discretized
    if args.PDE == "heat":
        pde_fun = pnmol.pde.examples.heat_1d_discretized

    # Access the arguments
    DT = args.DT
    DX = args.DX
    HIGH_RES_FACTOR_DX = args.HIGH_RES_FACTOR_DX
    HIGH_RES_FACTOR_DT = args.HIGH_RES_FACTOR_DT
    NUM_DERIVATIVES = args.NUM_DERIVATIVES
    NUGGET_COV_FD = args.NUGGET_COV_FD
    STENCIL_SIZE = args.STENCIL_SIZE
    PROGRESSBAR = args.PROGRESSBAR

    INPUT_SCALE = args.INPUT_SCALE
    OUTPUT_SCALE = args.OUTPUT_SCALE
    KERNEL = pnmol.kernels.SquareExponential(
        input_scale=INPUT_SCALE, output_scale=OUTPUT_SCALE
    )

    T0 = args.T0
    TMAX = args.TMAX
    DIFFUSION_RATE = args.DIFFUSION_RATE

    # Print the arguments (for demonstration purposes)
    print(f"PDE: {args.PDE}")
    print(f"DT: {DT}")
    print(f"DX: {DX}")
    print(f"HIGH_RES_FACTOR_DX: {HIGH_RES_FACTOR_DX}")
    print(f"HIGH_RES_FACTOR_DT: {HIGH_RES_FACTOR_DT}")
    print(f"NUM_DERIVATIVES: {NUM_DERIVATIVES}")
    print(f"NUGGET_COV_FD: {NUGGET_COV_FD}")
    print(f"STENCIL_SIZE: {STENCIL_SIZE}")
    print(f"PROGRESSBAR: {PROGRESSBAR}")
    print(f"INPUT_SCALE: {INPUT_SCALE}")
    print(f"OUTPUT_SCALE: {OUTPUT_SCALE}")
    print(f"T0: {T0}")
    print(f"TMAX: {TMAX}")
    print(f"DIFFUSION_RATE: {DIFFUSION_RATE}")

    # PDE problems
    with jax.disable_jit():
        PDE_PNMOL = pde_fun(
            t0=T0,
            tmax=TMAX,
            dx=DX,
            stencil_size_interior=STENCIL_SIZE,
            stencil_size_boundary=STENCIL_SIZE + 1,
            diffusion_rate=DIFFUSION_RATE,
            kernel=KERNEL,
            nugget_gram_matrix_fd=NUGGET_COV_FD,
            bcond="dirichlet",
        )
        PDE_TORNADOX = pde_fun(
            t0=T0,
            tmax=TMAX,
            dx=DX,
            stencil_size_interior=STENCIL_SIZE,
            stencil_size_boundary=STENCIL_SIZE + 1,
            diffusion_rate=DIFFUSION_RATE,
            kernel=KERNEL,
            nugget_gram_matrix_fd=NUGGET_COV_FD,
            bcond="dirichlet",
        )
        PDE_REFERENCE = pde_fun(
            t0=T0,
            tmax=TMAX,
            dx=DX / HIGH_RES_FACTOR_DX,
            stencil_size_interior=STENCIL_SIZE,
            stencil_size_boundary=STENCIL_SIZE + 1,
            diffusion_rate=DIFFUSION_RATE,
            kernel=KERNEL,
            nugget_gram_matrix_fd=NUGGET_COV_FD,
            bcond="dirichlet",
        )

    # Solve the PDE with the different methods
    KERNEL_NUGGET = pnmol.kernels.WhiteNoise(output_scale=1e-12)
    KERNEL_DIFFUSION_PNMOL = KERNEL  # + KERNEL_NUGGET

    # with jax.disable_jit():
    #     RESULT_TORNADOX = solve_pde_tornadox(
    #         PDE_TORNADOX, dt=DT, nu=NUM_DERIVATIVES, progressbar=PROGRESSBAR
    #     )

    RESULT_PNMOL_WHITE = solve_pde_pnmol_white(
        PDE_TORNADOX,
        dt=DT,
        nu=NUM_DERIVATIVES,
        progressbar=PROGRESSBAR,
        kernel=KERNEL_DIFFUSION_PNMOL,
    )
    RESULT_PNMOL_LATENT = solve_pde_pnmol_latent(
        PDE_PNMOL,
        dt=DT,
        nu=NUM_DERIVATIVES,
        progressbar=PROGRESSBAR,
        kernel=KERNEL_DIFFUSION_PNMOL,
    )
    RESULT_REFERENCE = solve_pde_reference(
        PDE_REFERENCE,
        dt=DT / HIGH_RES_FACTOR_DT,
        high_res_factor_dt=HIGH_RES_FACTOR_DT,
        high_res_factor_dx=HIGH_RES_FACTOR_DX,
    )
    save_result(RESULT_PNMOL_WHITE, prefix="pnmol_white")
    save_result(RESULT_PNMOL_LATENT, prefix="pnmol_latent")
    # save_result(RESULT_TORNADOX, prefix="tornadox")
    save_result(RESULT_REFERENCE, prefix="reference")

    plotting.figure_1()
    plotting.figure_1_singlerow()


if __name__ == "__main__":
    main()
