"""Figure 2."""

import argparse

import pathlib
from functools import partial

import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import plotting
from tqdm import tqdm

import pnmol
from pnmol import diffops

# Part 1: Maximum likelihood estimation of the input scale.


def input_scale_mle(*, mesh_points, obj, num_trial_points, num_mesh_points):
    """Compute the MLE over the input_scale parameter in a square exponential kernel."""
    y = obj(mesh_points[:, None]).squeeze()
    input_scale_trials = jnp.logspace(-3, 3, num_trial_points)

    log_likelihood_values = jnp.stack(
        [
            input_scale_to_log_likelihood(
                l=l,
                y=y,
                n=num_mesh_points,
                mesh_points=mesh_points,
            )
            for l in input_scale_trials
        ]
    )

    index_max = jnp.argmax(log_likelihood_values)
    return input_scale_trials[index_max]


def input_scale_to_log_likelihood(*, l, y, n, mesh_points):
    kernel = pnmol.kernels.SquareExponential(input_scale=l)
    K = kernel(mesh_points[:, None], mesh_points[None, :])
    return log_likelihood(gram_matrix=K, y=y, n=n)


def log_likelihood(*, gram_matrix, y, n):
    a = y @ jnp.linalg.solve(gram_matrix, y)
    b = jnp.log(jnp.linalg.det(gram_matrix))
    c = n * jnp.log(2 * jnp.pi)
    return -0.5 * (a + b + c)


def input_scale_to_rmse(
    input_scale, output_scale, stencil_size, *, diffop, mesh, obj_fun, truth_fun
):
    kernel = pnmol.kernels.SquareExponential(
        input_scale=input_scale, output_scale=output_scale
    )
    l, e = pnmol.discretize.fd_probabilistic(
        diffop=diffop,
        mesh_spatial=mesh,
        kernel=kernel,
        stencil_size_interior=stencil_size,
        stencil_size_boundary=stencil_size,
    )
    x = mesh.points
    fx = obj_fun(x).squeeze()
    dfx = truth_fun(x).squeeze()
    error_abs = jnp.abs(l @ fx - dfx) / jnp.abs(dfx)
    rmse = jnp.linalg.norm(error_abs) / jnp.sqrt(error_abs.size)
    return rmse, (l, e)


def sample(key, kernel, mesh_points, nugget_gram_matrix=1e-12):
    N = mesh_points.shape[0]
    gram_matrix = kernel(mesh_points, mesh_points.T)
    gram_matrix += nugget_gram_matrix * jnp.eye(N)

    sample = jax.random.normal(key, shape=(N, 2))
    return jnp.linalg.cholesky(gram_matrix) @ sample


def save_array(arr, /, *, suffix, path="experiments/results/figure2/"):
    path_path = pathlib.Path(path) / "figure1"
    if not path_path.is_dir():
        path_path.mkdir(parents=True)
    _assert_not_nan(arr)
    path_with_suffix = path + suffix
    jnp.save(path_with_suffix, arr)


def _assert_not_nan(arr):
    assert not jnp.any(jnp.isnan(arr))


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
        "--num_mesh_points",
        type=int,
        default=20,
        help="Number of mesh points (default: 20)",
    )
    parser.add_argument(
        "--num_trial_points",
        type=int,
        default=20,
        help="Number of trial points for MLE (default: 20)",
    )
    parser.add_argument(
        "--input_scales",
        type=float,
        nargs="+",
        default=[2.0, 0.8, 3.2],
        help="Input scales (default: [2.0, 0.8, 3.2])",
    )
    parser.add_argument(
        "--nugget_cholesky_E",
        type=float,
        default=1e-12,
        help="Nugget for Cholesky E (default: 1e-12)",
    )
    parser.add_argument(
        "--nugget_gram_matrix",
        type=float,
        default=1e-12,
        help="Nugget for Gram matrix (default: 1e-12)",
    )
    parser.add_argument(
        "--symmetrize_cholesky_E",
        action="store_true",
        help="Symmetrize Cholesky E (default: True)",
    )
    parser.add_argument(
        "--no-symmetrize_cholesky_E",
        dest="symmetrize_cholesky_E",
        action="store_false",
        help="Do not symmetrize Cholesky E",
    )
    parser.set_defaults(symmetrize_cholesky_E=True)
    parser.add_argument(
        "--kernel_input_scale",
        type=float,
        default=3.2,
        help="Input scale for the kernel (default: 3.2)",
    )
    parser.add_argument(
        "--kernel_output_scale",
        type=float,
        default=0.01,
        help="Output scale for the kernel (default: 0.01)",
    )
    parser.add_argument(
        "--nu", type=float, default=0.9, help="Value of nu (default: 0.9)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    pde = args.PDE
    num_mesh_points = args.num_mesh_points
    num_trial_points = args.num_trial_points
    input_scales = jnp.array(args.input_scales)
    nugget_cholesky_E = args.nugget_cholesky_E
    nugget_gram_matrix = args.nugget_gram_matrix
    symmetrize_cholesky_E = args.symmetrize_cholesky_E
    kernel_input_scale = args.kernel_input_scale
    kernel_output_scale = args.kernel_output_scale
    nu = args.nu

    # print the arguments
    print(f"PDE: {pde}")
    print(f"num_mesh_points: {num_mesh_points}")
    print(f"num_trial_points: {num_trial_points}")
    print(f"input_scales: {input_scales}")
    print(f"nugget_cholesky_E: {nugget_cholesky_E}")
    print(f"nugget_gram_matrix: {nugget_gram_matrix}")
    print(f"symmetrize_cholesky_E: {symmetrize_cholesky_E}")
    print(f"kernel_input_scale: {kernel_input_scale}")
    print(f"kernel_output_scale: {kernel_output_scale}")

    # Define the basic setup: target function, etc.
    obj_fun_scalar = lambda x: jnp.exp(x.dot(x))
    obj_fun = jax.vmap(obj_fun_scalar)

    nu = diffops.constant(nu)  # Use the nu value from command line arguments

    identity = diffops.identity()
    derivative = diffops.divergence(is_1d=True)
    laplace = diffops.laplace()

    if pde == "burgers":
        diffop = nu * laplace - identity * derivative
    elif pde == "heat":
        diffop = nu * laplace
    else:
        raise ValueError(f"Unknown PDE: {pde}")

    truth_fun = jax.vmap(diffop(obj_fun_scalar))

    # Choose a mesh
    mesh = pnmol.mesh.RectangularMesh(
        jnp.linspace(0, 1, num_mesh_points, endpoint=True)[:, None],
        bbox=jnp.asarray([0.0, 1.0])[:, None],
    )

    # Compute the MLE estimate (for comparison)
    scale_mle = input_scale_mle(
        mesh_points=mesh.points.squeeze(),
        obj=obj_fun,
        num_trial_points=num_trial_points,
        num_mesh_points=num_mesh_points,
    )
    print(f"Best MLE input scale selected: {scale_mle}")

    # Compute all RMSEs
    stencil_sizes = jnp.arange(3, len(mesh), step=2)
    e = partial(
        input_scale_to_rmse,
        output_scale=1.0,
        diffop=diffop,
        mesh=mesh,
        obj_fun=obj_fun,
        truth_fun=truth_fun,
    )

    rmse_all = jnp.asarray(
        [
            [e(input_scale=l, stencil_size=s)[0] for l in input_scales]
            for s in tqdm(stencil_sizes)
        ]
    )
    rmse_all = jnp.nan_to_num(rmse_all, nan=100.0)

    # Compute L and E for a number of stencil sizes
    L_sparse, E_sparse = e(input_scale=scale_mle, stencil_size=3)[1]
    L_dense, E_dense = pnmol.discretize.collocation_global(
        diffop=diffop,
        mesh_spatial=mesh,
        kernel=pnmol.kernels.SquareExponential(
            input_scale=kernel_input_scale, output_scale=kernel_output_scale
        ),
        nugget_cholesky_E=nugget_cholesky_E,
        nugget_gram_matrix=nugget_gram_matrix,
        symmetrize_cholesky_E=symmetrize_cholesky_E,
    )

    # Plotting purposes...
    xgrid = jnp.linspace(0, 1, 150)[:, None]
    fx = obj_fun(xgrid).squeeze()
    dfx = truth_fun(xgrid).squeeze()

    # Sample from the different priors
    key = jax.random.PRNGKey(seed=123)
    k1 = pnmol.kernels.SquareExponential(input_scale=input_scales[0])
    k2 = pnmol.kernels.SquareExponential(input_scale=input_scales[1])
    k3 = pnmol.kernels.SquareExponential(input_scale=input_scales[2])
    s1 = sample(key=key, kernel=k1, mesh_points=xgrid)
    _, key = jax.random.split(key)
    s2 = sample(key=key, kernel=k2, mesh_points=xgrid)
    _, key = jax.random.split(key)
    s3 = sample(key=key, kernel=k3, mesh_points=xgrid)

    save_array(rmse_all, suffix="rmse_all")
    save_array(input_scales, suffix="input_scales")
    save_array(stencil_sizes, suffix="stencil_sizes")
    save_array(L_sparse, suffix="L_sparse")
    save_array(L_dense, suffix="L_dense")
    save_array(E_sparse, suffix="E_sparse")
    save_array(E_dense, suffix="E_dense")
    save_array(xgrid, suffix="xgrid")
    save_array(fx, suffix="fx")
    save_array(dfx, suffix="dfx")
    save_array(s1, suffix="s1")
    save_array(s2, suffix="s2")
    save_array(s3, suffix="s3")

    plotting.figure_2()


if __name__ == "__main__":
    main()
