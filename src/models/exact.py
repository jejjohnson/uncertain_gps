import argparse
from typing import List, Optional, Tuple, Union

from scipy.cluster.vq import kmeans

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch
from torch.nn import Parameter

from pyro.infer.util import torch_backward, torch_item
from pyro.infer import TraceMeanField_ELBO, JitTraceMeanField_ELBO
import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train_models import train

pyro.enable_validation(True)  # can help with debugging


class GPRegressor:
    def __init__(
        self,
        kernel: Optional[gp.kernels.Kernel] = None,
        noise: float = 0.01,
        mean_function: Optional[bool] = None,
        random_state: int = 123,
        jitter: float = 1e-3,
    ):
        self.device = torch.device("cpu")
        self.kernel = kernel
        self.mean_function = mean_function
        self.noise = noise
        self.rng = np.random.RandomState(random_state)
        self.jitter = jitter

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        # get dimensions
        n_samples, d_dimensions = X.shape

        # convert data to tensor
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y.squeeze(), dtype=torch.float32, device=self.device)

        # Default kernel function
        if self.kernel is None:
            self.kernel = gp.kernels.RBF(input_dim=d_dimensions)

        # initialize GP model
        self.model = gp.models.GPRegression(
            X,
            y,
            kernel=self.kernel,
            noise=torch.tensor(self.noise, dtype=torch.float32, device=self.device),
            mean_function=self.mean_function,
            jitter=self.jitter,
        )

        return self

    def optimize(
        self, num_steps: 100, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        self.losses = train(self.model, num_steps=num_steps, optimizer=optimizer)

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        full_cov: bool = False,
        noiseless: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Convert to tensor
        X = torch.tensor(
            X, dtype=torch.float32, device=self.device, requires_grad=False
        )

        with torch.no_grad():

            # get mean and variance of predictions
            mean, var = self.model(X, full_cov=full_cov, noiseless=noiseless)

            if return_std:
                return mean.detach().numpy(), var.detach().sqrt().numpy()
            else:
                return mean.detach().numpy()


def main(args):

    # initialize GP model
    rng = np.random.RandomState(0)

    # Generate sample data
    noise = 1.0
    input_noise = 0.2
    n_train = 1_000
    n_test = 1_000
    n_inducing = 100
    batch_size = None
    X = 15 * rng.rand(n_train, 1)

    def f(x):
        return np.sin(x)

    y = f(X)
    X += input_noise * rng.randn(X.shape[0], X.shape[1])
    y += noise * (0.5 - rng.rand(X.shape[0], X.shape[1]))  # add noise
    X_plot = np.linspace(0, 20, n_test)[:, None]
    X_plot += input_noise * rng.randn(X_plot.shape[0], X_plot.shape[1])
    X_plot = np.sort(X_plot, axis=0)

    gp_model = GPRegressor(random_state=0)

    gp_model.fit(X, y)

    gp_model.optimize(num_steps=args.epochs)

    from sklearn.metrics import r2_score

    ymu, ystd = gp_model.predict(X, return_std=True)

    print(r2_score(y, ymu))
    print(mean_absolute_error(y, ymu))
    print(mean_squared_error(y, ymu))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyro Exact GP Example")
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1_000,
        metavar="N",
        help="number of steps (default: 1_000)",
    )
    args = parser.parse_args()

    pyro.set_rng_seed(args.seed)
    main(args)
