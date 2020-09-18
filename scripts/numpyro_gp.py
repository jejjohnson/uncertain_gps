# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Gaussian Process
================
In this example we show how to use NUTS to sample from the posterior
over the hyperparameters of a gaussian process.
"""
from pyprojroot import here
import sys
sys.path.append(str(here()))
from src.models.jaxgp.data import near_square_wave
import pathlib
import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as onp

import jax
from jax import vmap
import jax.numpy as np
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

matplotlib.use("Agg")  # noqa: E402

# FIG_PATH = pathlib.Path(here()).joinpath()


# squared exponential kernel with diagonal noise term
def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = np.power((X[:, None] - Z) / length, 2.0)
    k = var * np.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * np.eye(X.shape[0])
    return k


def model(X, Y):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))

    # compute kernel
    k = kernel(X, X, var, length, noise)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=np.zeros(X.shape[0]), covariance_matrix=k),
        obs=Y,
    )


# helper function for doing hmc inference
def run_inference(model, args, rng_key, X, Y):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        args.num_warmup,
        args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
def predict(rng_key, X, Y, X_test, var, length, noise):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)
    K_xx_inv = np.linalg.inv(k_XX)
    K = k_pp - np.matmul(k_pX, np.matmul(K_xx_inv, np.transpose(k_pX)))
    sigma_noise = np.sqrt(np.clip(np.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )
    mean = np.matmul(k_pX, np.matmul(K_xx_inv, Y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


# create artificial regression dataset
def get_data(N=30, sigma_obs=0.15, N_test=400):
    onp.random.seed(0)
    X = np.linspace(-1, 1, N)
    Y = X + 0.2 * np.power(X, 3.0) + 0.5 * np.power(0.5 + X, 2.0) * np.sin(4.0 * X)
    Y += sigma_obs * onp.random.randn(N)
    Y -= np.mean(Y)
    Y /= np.std(Y)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = np.linspace(-1.3, 1.3, N_test)

    return X, Y, X_test


def main(args):
    
    if args.dataset == 'square':
        X, Y, X_test, _ = near_square_wave(n_train=args.num_data)
        
    else:
        X, Y, X_test = get_data(N=args.num_data)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y)

    # do prediction
    vmap_args = (
        random.split(rng_key_predict, args.num_samples * args.num_chains),
        samples["kernel_var"],
        samples["kernel_length"],
        samples["kernel_noise"],
    )
    means, predictions = vmap(
        lambda rng_key, var, length, noise: predict(
            rng_key, X, Y, X_test, var, length, noise
        )
    )(*vmap_args)

    mu_y = onp.mean(means, axis=0)
    percentiles = onp.percentile(predictions, [5.0, 95.0], axis=0)

    # make plots
    
    fig, ax = plt.subplots()
    ax.scatter(X, Y, c="red", label="Training Data")
    ax.plot(
        X_test.squeeze(),
        mu_y.squeeze(),
        label=r"Predictive Mean",
        color="black",
        linewidth=3,
    )
    ax.fill_between(
        X_test.squeeze(),
        percentiles[0, :], percentiles[1, :],
        alpha=0.3,
        color="darkorange",
        label=f"Predictive Std (95% Confidence)",
    )
    ax.set_ylim([-3.5, 3.5])
    ax.legend(fontsize=12)
    plt.tight_layout()
    fig.savefig("figures/jaxgp/examples/mcmc/1d_gp_train.png")
    plt.show()

if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.2.4")
    parser = argparse.ArgumentParser(description="Gaussian Process example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-data", nargs="?", default=25, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--dataset", default="default", type=str,)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
