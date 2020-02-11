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

pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(0)


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


def train(gpmodule, optimizer=None, loss_fn=None, retain_graph=None, num_steps=1000):
    """
    A helper to optimize parameters for a GP module.

    :param ~pyro.contrib.gp.models.GPModel gpmodule: A GP module.
    :param ~torch.optim.Optimizer optimizer: A PyTorch optimizer instance.
        By default, we use Adam with ``lr=0.01``.
    :param callable loss_fn: A loss function which takes inputs are
        ``gpmodule.model``, ``gpmodule.guide``, and returns ELBO loss.
        By default, ``loss_fn=TraceMeanField_ELBO().differentiable_loss``.
    :param bool retain_graph: An optional flag of ``torch.autograd.backward``.
    :param int num_steps: Number of steps to run SVI.
    :returns: a list of losses during the training procedure
    :rtype: list
    """
    optimizer = (
        torch.optim.Adam(gpmodule.parameters(), lr=0.01)
        if optimizer is None
        else optimizer
    )
    # TODO: add support for JIT loss
    loss_fn = TraceMeanField_ELBO().differentiable_loss if loss_fn is None else loss_fn

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(gpmodule.model, gpmodule.guide)
        torch_backward(loss, retain_graph)
        return loss

    losses = []
    with tqdm.trange(num_steps) as bar:
        for epoch in bar:
            loss = optimizer.step(closure)
            losses.append(torch_item(loss))
            # print statistics
            postfix = dict(Loss=f"{torch_item(loss):.3f}")
            bar.set_postfix(postfix)
    return losses


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
