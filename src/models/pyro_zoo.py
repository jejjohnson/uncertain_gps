import argparse
from typing import List, Optional, Tuple, Union

from scipy.cluster.vq import kmeans

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch
from torch.nn import Parameter

pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(0)


def get_exactgp_model():

    return None


def main():

    rng = np.random.RandomState(0)

    # Generate sample data
    noise = 1.0
    input_noise = 0.2
    n_train = 10_000
    n_test = 1_000
    n_inducing = 100
    batch_size = None
    X = 15 * rng.rand(n_train, 1)

    X += input_noise * rng.randn(X.shape[0], X.shape[1])
    y += noise * (0.5 - rng.rand(X.shape[0], X.shape[1]))  # add noise
    X_plot = np.linspace(0, 20, n_test)[:, None]
    X_plot += input_noise * rng.randn(X_plot.shape[0], X_plot.shape[1])
    X_plot = np.sort(X_plot, axis=0)
    pass


if __name__ == "__main__":
    main()
