"""Code to generate figure 1."""

import itertools
import pathlib
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy
import plotting
import scipy.integrate
import tornadox
import tqdm

import pnmol


def solve_pde_reference(pde, *, high_res_factor_dx):
    t_eval = jnp.array([pde.tmax])
    ivp = pde.to_tornadox_ivp()
    sol = scipy.integrate.solve_ivp(
        jax.jit(ivp.f), ivp.t_span, ivp.y0, t_eval=t_eval, atol=1e-10, rtol=1e-10
    )

    mean = sol.y.T
    assert mean.shape == (1, ivp.y0.size)
    mean = mean.squeeze()
    print("reference", mean)

    i_mean = mean[high_res_factor_dx - 1 :: high_res_factor_dx]

    return i_mean


def solve_pde_pnmol_white(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.white.LinearWhiteNoiseEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )

    start = time.time()
    with jax.disable_jit():
        final_state, _ = ek1.simulate_final_state(pde, progressbar=progressbar)
    elapsed_time = time.time() - start

    E0 = ek1.iwp.projection_matrix(0)
    mean, std, cov = read_mean_and_std_and_cov(final_state, E0)
    print("white", mean)

    i_mean, i_std = mean[1:-1], std[1:-1]

    i_cov = cov[1:-1, 1:-1]

    return i_mean, i_std, i_cov, elapsed_time


def solve_pde_tornadox(pde, *, dt, nu, progressbar):
    steprule = tornadox.step.ConstantSteps(dt)
    ivp = pde.to_tornadox_ivp()
    ek1 = tornadox.ek1.ReferenceEK1ConstantDiffusion(
        num_derivatives=nu,
        steprule=steprule,
        initialization=tornadox.init.Stack(use_df=False),
    )

    start = time.time()
    with jax.disable_jit():
        final_state, info = ek1.simulate_final_state(ivp, progressbar=progressbar)
    elapsed_time = time.time() - start

    E0 = ek1.iwp.projection_matrix(0)
    mean, std, cov = read_mean_and_std_and_cov(final_state, E0)
    print("tornadox", mean)
    i_mean, i_std = mean, std

    i_cov = cov

    return i_mean, i_std, i_cov, elapsed_time


def read_mean_and_std_and_cov(final_state, E0):
    # print("White")
    # print(final_state.y.mean.shape, final_state.y.cov_sqrtm.shape)
    mean = final_state.y.mean[0, :]
    cov = E0 @ (final_state.y.cov_sqrtm @ final_state.y.cov_sqrtm.T) @ E0.T
    std = jnp.sqrt(jnp.diagonal(cov))
    return mean, std, cov


def save_result(result, /, *, prefix, path="experiments/results"):
    path = pathlib.Path(path) / "figure3"
    if not path.is_dir():
        path.mkdir(parents=True)

    path_error_abs = path / (prefix + "_error_abs.npy")
    path_error_rel = path / (prefix + "_error_rel.npy")
    path_std = path / (prefix + "_std.npy")
    path_runtime = path / (prefix + "_runtime.npy")
    path_chi2 = path / (prefix + "_chi2.npy")
    path_dt = path / (prefix + "_dt.npy")
    path_dx = path / (prefix + "_dx.npy")

    jnp.save(path_error_abs, result["error_abs"])
    jnp.save(path_error_rel, result["error_rel"])
    jnp.save(path_std, result["std"])
    jnp.save(path_runtime, result["runtime"])
    jnp.save(path_chi2, result["chi2"])
    jnp.save(path_dt, result["dt"])
    jnp.save(path_dx, result["dx"])


#
# # Ranges
# DTs = jnp.logspace(
#     # numpy.log10(0.001), numpy.log10(0.5), num=10, endpoint=True, base=10
#     numpy.log10(0.01),
#     numpy.log10(2.5),
#     num=9,
#     endpoint=True,
#     base=10,
# )
DTs = 2.0 ** jnp.arange(2, -7, step=-0.5)

DXs = 1.0 / (2.0 ** jnp.arange(2, 7))

# Hyperparameters (method)
HIGH_RES_FACTOR_DX = 10
NUM_DERIVATIVES = 2
NUGGET_COV_FD = 0.0
STENCIL_SIZE = 3
PROGRESSBAR = True

# Hyperparameters (problem)
T0, TMAX = 0.0, 6.0
DIFFUSION_RATE = 0.035


RESULT_WHITE, RESULT_TORNADOX = [
    {
        "error_abs": numpy.zeros((len(DXs), len(DTs))),
        "error_rel": numpy.zeros((len(DXs), len(DTs))),
        "std": numpy.zeros((len(DXs), len(DTs))),
        "runtime": numpy.zeros((len(DXs), len(DTs))),
        "chi2": numpy.zeros((len(DXs), len(DTs))),
        "dx": numpy.zeros((len(DXs), len(DTs))),
        "dt": numpy.zeros((len(DXs), len(DTs))),
    }
    for _ in range(2)
]

i_exp = 0
num_exp_total = len(DXs) * len(DTs)


# Solve the PDE with the different methods
# KERNEL_DIFFUSION_PNMOL = pnmol.kernels.duplicate(
#     pnmol.kernels.Matern52() + pnmol.kernels.WhiteNoise(output_scale=1e-3), num=3
# )

KERNEL_DIFFUSION_PNMOL = pnmol.kernels.Matern52() + pnmol.kernels.WhiteNoise()
for i_dx, dx in enumerate(sorted(DXs)):
    # PDE problems
    PDE_PNMOL = pnmol.pde.examples.burgers_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=dx,
        stencil_size_interior=STENCIL_SIZE,
        stencil_size_boundary=STENCIL_SIZE + 2,
        nugget_gram_matrix_fd=NUGGET_COV_FD,
        kernel=pnmol.kernels.SquareExponential(),
    )

    # print(jnp.diag(PDE_PNMOL.E_sqrtm))
    PDE_REFERENCE = pnmol.pde.examples.burgers_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=dx / HIGH_RES_FACTOR_DX,
        stencil_size_interior=STENCIL_SIZE,
        stencil_size_boundary=STENCIL_SIZE + 1,
        nugget_gram_matrix_fd=NUGGET_COV_FD,
        kernel=pnmol.kernels.SquareExponential(),
    )
    mean_reference = solve_pde_reference(
        PDE_REFERENCE, high_res_factor_dx=HIGH_RES_FACTOR_DX
    )
    for i_dt, dt in enumerate(sorted(DTs)):
        i_exp = i_exp + 1

        dim = PDE_PNMOL.y0.size
        print(
            f"\n======| Experiment {i_exp} of {num_exp_total} +++ dt={dt}, dx={dx} (state dimension: {dim} = 3 * {dim/3}) \n"
        )

        mean_white, std_white, cov_white, elapsed_time_white = solve_pde_pnmol_white(
            PDE_PNMOL,
            dt=dt,
            nu=NUM_DERIVATIVES,
            progressbar=PROGRESSBAR,
            kernel=KERNEL_DIFFUSION_PNMOL,
        )

        (
            mean_tornadox,
            std_tornadox,
            cov_tornadox,
            elapsed_time_tornadox,
        ) = solve_pde_tornadox(
            PDE_PNMOL, dt=dt, nu=NUM_DERIVATIVES, progressbar=PROGRESSBAR
        )

        error_white_abs = jnp.abs(mean_white - mean_reference)
        error_tornadox_abs = jnp.abs(mean_tornadox - mean_reference)
        rmse_white_rel = jnp.linalg.norm(error_white_abs / mean_reference) / jnp.sqrt(
            mean_reference.size
        )
        rmse_white_abs = jnp.linalg.norm(error_white_abs) / jnp.sqrt(
            mean_reference.size
        )
        rmse_tornadox_rel = jnp.linalg.norm(
            error_tornadox_abs / mean_reference
        ) / jnp.sqrt(mean_reference.size)
        rmse_tornadox_abs = jnp.linalg.norm(error_tornadox_abs) / jnp.sqrt(
            mean_reference.size
        )

        chi2_white = (
            error_white_abs
            @ jnp.linalg.solve(cov_white, error_white_abs)
            / error_white_abs.size
        )
        chi2_tornadox = (
            error_tornadox_abs
            @ jnp.linalg.solve(cov_tornadox, error_tornadox_abs)
            / error_tornadox_abs.size
        )

        mean_std_white = jnp.mean(std_white)
        mean_std_tornadox = jnp.mean(std_tornadox)

        RESULT_WHITE["error_abs"][i_dx, i_dt] = rmse_white_abs
        RESULT_WHITE["error_rel"][i_dx, i_dt] = rmse_white_rel
        RESULT_WHITE["std"][i_dx, i_dt] = mean_std_white
        RESULT_WHITE["runtime"][i_dx, i_dt] = elapsed_time_white
        RESULT_WHITE["chi2"][i_dx, i_dt] = chi2_white
        RESULT_WHITE["dt"][i_dx, i_dt] = dt
        RESULT_WHITE["dx"][i_dx, i_dt] = dx

        RESULT_TORNADOX["error_abs"][i_dx, i_dt] = rmse_tornadox_abs
        RESULT_TORNADOX["error_rel"][i_dx, i_dt] = rmse_tornadox_rel
        RESULT_TORNADOX["std"][i_dx, i_dt] = mean_std_tornadox
        RESULT_TORNADOX["runtime"][i_dx, i_dt] = elapsed_time_tornadox
        RESULT_TORNADOX["chi2"][i_dx, i_dt] = chi2_tornadox
        RESULT_TORNADOX["dt"][i_dx, i_dt] = dt
        RESULT_TORNADOX["dx"][i_dx, i_dt] = dx

        print(
            f"MOL:\n\tRMSE={rmse_tornadox_rel}, chi2={chi2_tornadox}, time={elapsed_time_tornadox}"
        )
        print(
            f"PNMOL(white):\n\tRMSE={rmse_white_rel}, chi2={chi2_white}, time={elapsed_time_white}"
        )


save_result(RESULT_WHITE, prefix="pnmol_white")
save_result(RESULT_TORNADOX, prefix="tornadox")
