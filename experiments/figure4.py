"""Work-precision diagrams and so on."""

import argparse
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox
from scipy.integrate import solve_ivp

import pnmol

parser = argparse.ArgumentParser()
parser.add_argument("--t0", type=float, default=0.0)
parser.add_argument("--tmax", type=float, default=6.0)
parser.add_argument("--ref_scale", type=int, default=7)

args = parser.parse_args()

# print the arguments
print(f"t0: {args.t0}")
print(f"tmax: {args.tmax}")
print(f"ref_scale: {args.ref_scale}")

# PDE
pde_kwargs = {"t0": args.t0, "tmax": args.tmax}

for dx in [0.01, 0.05, 0.2]:
    pde = pnmol.pde.examples.burgers_1d_discretized(
        **pde_kwargs,
        dx=dx,
        stencil_size_interior=3,
        stencil_size_boundary=4,
    )
    ivp = pde.to_tornadox_ivp()

    ref_scale = args.ref_scale
    pde_ref = pnmol.pde.examples.burgers_1d_discretized(
        **pde_kwargs,
        dx=dx / ref_scale,
    )
    ivp_ref = pde_ref.to_tornadox_ivp()
    print("assembled")

    print(f"spatial_mesh shape", pde.mesh_spatial.shape)
    print(f"ref spatial_mesh shape", pde_ref.mesh_spatial.shape)

    print(f"y0 shape", ivp.y0.shape)

    ref_sol = solve_ivp(
        jax.jit(ivp_ref.f),
        ivp_ref.t_span,
        y0=ivp_ref.y0,
        method="LSODA",
        atol=1e-10,
        rtol=1e-10,
        t_eval=(ivp_ref.t0, ivp_ref.tmax),
    )

    print("solved")
    u_reference_full = ref_sol.y[:, -1]
    u_reference = u_reference_full[(ref_scale - 1) :: ref_scale]
    print(f"u_reference_full.shape={u_reference_full.shape}")
    print(f"u_reference.shape={u_reference.shape}")

    k = pnmol.kernels.Matern52() + pnmol.kernels.WhiteNoise()

    pnmol_white_rmse = []
    pnmol_white_nsteps = []
    pnmol_white_chi2 = []
    pnmol_white_time = []

    pnmol_latent_rmse = []
    pnmol_latent_nsteps = []
    pnmol_latent_chi2 = []
    pnmol_latent_time = []

    mol_rmse = []
    mol_nsteps = []
    mol_chi2 = []
    mol_time = []

    dts = jnp.logspace(0.0, -2.5, 12, endpoint=True)
    for dt in dts:

        # [PNMOL-LATENT] Solve
        steps = pnmol.odetools.step.Constant(dt)
        solver = pnmol.latent.LinearLatentForceEK1(  # should be semi-linear?
            num_derivatives=2, steprule=steps, spatial_kernel=k
        )
        time_start = time.time()
        sol_pnmol_latent, sol_pnmol_latent_info = solver.simulate_final_state(
            pde, progressbar=True
        )
        time_pnmol_latent = time.time() - time_start

        # [PNMOL-LATENT] Extract mean
        mean_state = sol_pnmol_latent.y.mean[0]
        u_pnmol_latent_full = jnp.split(mean_state, 2)[0]
        u_pnmol_latent = u_pnmol_latent_full[1:-1]

        print(f"u_pnmol_latent_full.shape={u_pnmol_latent_full.shape}")
        print(f"cov_sqrtm.shape={sol_pnmol_latent.y.cov_sqrtm.shape}")
        print(f"E0.shape={solver.E0.shape}")

        # [PNMOL-LATENT] Extract covariance: first remove xi, then remove "v"
        cov_final_latent = sol_pnmol_latent.y.cov_sqrtm @ sol_pnmol_latent.y.cov_sqrtm.T
        cov_final_no_xi = jnp.split(
            jnp.split(cov_final_latent, 2, axis=-1)[0], 2, axis=0
        )[0]
        print(f"cov_final_no_xi.shape={cov_final_no_xi.shape}")
        cov_final_latent_interesting = solver.E0 @ cov_final_no_xi @ solver.E0.T

        print(
            f"cov_final_latent_interesting.shape={cov_final_latent_interesting.shape}"
        )

        cov_final_latent_u = cov_final_latent_interesting[1:-1, 1:-1]

        # [PNMOL-LATENT] Compute error and calibration
        error_pnmol_latent_abs = jnp.abs(u_pnmol_latent - u_reference)
        error_pnmol_latent_rel = error_pnmol_latent_abs / jnp.abs(u_reference)
        rmse_pnmol_latent = jnp.linalg.norm(error_pnmol_latent_rel) / jnp.sqrt(
            u_pnmol_latent.size
        )
        chi2_pnmol_latent = (
            error_pnmol_latent_abs
            @ jnp.linalg.solve(cov_final_latent_u, error_pnmol_latent_abs)
            / error_pnmol_latent_abs.shape[0]
        )

        ################################################################
        ################################################################

        # [PNMOL-WHITE] Solve
        steps = pnmol.odetools.step.Constant(dt)
        solver = pnmol.white.LinearWhiteNoiseEK1(
            num_derivatives=2, steprule=steps, spatial_kernel=k
        )
        time_start = time.time()
        sol_pnmol_white, sol_pnmol_white_info = solver.simulate_final_state(
            pde, progressbar=True
        )
        time_pnmol_white = time.time() - time_start

        # [PNMOL-WHITE] Extract mean
        u_pnmol_white_full = sol_pnmol_white.y.mean[0]
        u_pnmol_white = u_pnmol_white_full[1:-1]

        # [PNMOL-WHITE] Extract covariance
        cov_final_white = sol_pnmol_white.y.cov_sqrtm @ sol_pnmol_white.y.cov_sqrtm.T
        cov_final_white_interesting = solver.E0 @ cov_final_white @ solver.E0.T
        cov_final_white_u = cov_final_white_interesting[1:-1, 1:-1]

        # [PNMOL-WHITE] Compute error and calibration
        error_pnmol_white_abs = jnp.abs(u_pnmol_white - u_reference)
        error_pnmol_white_rel = error_pnmol_white_abs / jnp.abs(u_reference)
        rmse_pnmol_white = jnp.linalg.norm(error_pnmol_white_rel) / jnp.sqrt(
            u_pnmol_white.size
        )
        chi2_pnmol_white = (
            error_pnmol_white_abs
            @ jnp.linalg.solve(cov_final_white_u, error_pnmol_white_abs)
            / error_pnmol_white_abs.shape[0]
        )

        ################################################################
        ################################################################

        # [MOL] Solve
        steps = tornadox.step.ConstantSteps(dt)
        ek1 = tornadox.ek1.ReferenceEK1(
            num_derivatives=2,
            steprule=steps,
            initialization=tornadox.init.Stack(use_df=False),
        )
        time_start = time.time()
        sol_mol, sol_mol_info = ek1.simulate_final_state(ivp, progressbar=True)
        time_mol = time.time() - time_start

        # [MOL] Extract mean
        u_mol = sol_mol.y.mean[0]

        # [MOL] Extract covariance
        cov_final_mol = sol_mol.y.cov_sqrtm @ sol_mol.y.cov_sqrtm.T
        cov_final_interesting_mol = ek1.P0 @ cov_final_mol @ ek1.P0.T
        cov_final_u_mol = cov_final_interesting_mol

        # [MOL] Compute error and calibration
        error_mol_abs = jnp.abs(u_mol - u_reference)

        error_mol_rel = error_mol_abs / jnp.abs(u_reference)
        rmse_mol = jnp.linalg.norm(error_mol_rel) / jnp.sqrt(u_mol.size)
        chi2_mol = (
            error_mol_abs
            @ jnp.linalg.solve(cov_final_u_mol, error_mol_abs)
            / error_mol_abs.shape[0]
        )

        ################################################################
        ################################################################

        # Print results
        print(
            f"MOL:\n\tRMSE={rmse_mol}, chi2={chi2_mol}, nsteps={sol_mol_info['num_steps']}, time={time_mol}"
        )
        print(
            f"PNMOL(white):\n\tRMSE={rmse_pnmol_white}, chi2={chi2_pnmol_white}, nsteps={sol_pnmol_white_info['num_steps']}, time={time_pnmol_white}"
        )
        print(
            f"PNMOL(latent):\n\tRMSE={rmse_pnmol_latent}, chi2={chi2_pnmol_latent}, nsteps={sol_pnmol_latent_info['num_steps']}, time={time_pnmol_latent}"
        )

        pnmol_latent_rmse.append(rmse_pnmol_latent)
        pnmol_latent_chi2.append(chi2_pnmol_latent)
        pnmol_latent_nsteps.append(sol_pnmol_latent_info["num_steps"])
        pnmol_latent_time.append(time_pnmol_latent)

        pnmol_white_rmse.append(rmse_pnmol_white)
        pnmol_white_chi2.append(chi2_pnmol_white)
        pnmol_white_nsteps.append(sol_pnmol_white_info["num_steps"])
        pnmol_white_time.append(time_pnmol_white)

        mol_rmse.append(rmse_mol)
        mol_chi2.append(chi2_mol)
        mol_nsteps.append(sol_mol_info["num_steps"])
        mol_time.append(time_mol)

        print()

    path = "experiments/results/figure4/" + f"dx_{dx}_"

    jnp.save(path + "pnmol_white_rmse.npy", jnp.asarray(pnmol_white_rmse))
    jnp.save(path + "pnmol_white_chi2.npy", jnp.asarray(pnmol_white_chi2))
    jnp.save(path + "pnmol_white_nsteps.npy", jnp.asarray(pnmol_white_nsteps))
    jnp.save(path + "pnmol_white_time.npy", jnp.asarray(pnmol_white_time))

    jnp.save(path + "pnmol_latent_rmse.npy", jnp.asarray(pnmol_latent_rmse))
    jnp.save(path + "pnmol_latent_chi2.npy", jnp.asarray(pnmol_latent_chi2))
    jnp.save(path + "pnmol_latent_nsteps.npy", jnp.asarray(pnmol_latent_nsteps))
    jnp.save(path + "pnmol_latent_time.npy", jnp.asarray(pnmol_latent_time))

    jnp.save(path + "mol_rmse.npy", jnp.asarray(mol_rmse))
    jnp.save(path + "mol_chi2.npy", jnp.asarray(mol_chi2))
    jnp.save(path + "mol_nsteps.npy", jnp.asarray(mol_nsteps))
    jnp.save(path + "mol_time.npy", jnp.asarray(mol_time))

    jnp.save(path + "dts.npy", jnp.asarray(dts))
