import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import tqdm
from jax.experimental import optimizers

from src.models.jaxgp.data import get_data
from src.models.jaxgp.exact import posterior
from src.models.jaxgp.kernels import gram, rbf_kernel
from src.models.jaxgp.loss import marginal_likelihood
from src.models.jaxgp.mean import zero_mean
from src.models.jaxgp.utils import (cholesky_factorization, get_factorizations,
                                    saturate)

plt.style.use(["seaborn-talk"])


def main():
    X, y, Xtest, ytest = get_data(50)

    # PRIOR FUNCTIONS (mean, covariance)
    mu_f = zero_mean
    cov_f = functools.partial(gram, rbf_kernel)
    gp_priors = (mu_f, cov_f)

    # Kernel, Likelihood parameters
    params = {
        "gamma": 2.0,
        # 'length_scale': 1.0,
        # 'var_f': 1.0,
        "likelihood_noise": 1.0,
    }
    # saturate parameters with likelihoods
    params = saturate(params)

    # LOSS FUNCTION
    mll_loss = jax.jit(functools.partial(marginal_likelihood, gp_priors))

    # GRADIENT LOSS FUNCTION
    dloss = jax.jit(jax.grad(mll_loss))

    # STEP FUNCTION
    @jax.jit
    def step(params, X, y, opt_state):
        # calculate loss
        loss = mll_loss(params, X, y)

        # calculate gradient of loss
        grads = dloss(params, X, y)

        # update optimizer state
        opt_state = opt_update(0, grads, opt_state)

        # update params
        params = get_params(opt_state)

        return params, opt_state, loss

    # TRAINING PARARMETERS
    n_epochs = 500
    learning_rate = 0.01
    losses = list()

    # initialize optimizer
    opt_init, opt_update, get_params = optimizers.rmsprop(step_size=learning_rate)

    # initialize parameters
    opt_state = opt_init(params)

    # get initial parameters
    params = get_params(opt_state)

    postfix = {}

    with tqdm.trange(n_epochs) as bar:

        for i in bar:
            # 1 step - optimize function
            params, opt_state, value = step(params, X, y, opt_state)

            # update params
            postfix = {}
            for ikey in params.keys():
                postfix[ikey] = f"{jax.nn.softplus(params[ikey]):.2f}"

            # save loss values
            losses.append(value.mean())

            # update progress bar
            postfix["Loss"] = f"{onp.array(losses[-1]):.2f}"
            bar.set_postfix(postfix)
            # saturate params
            params = saturate(params)

    # Posterior Predictions
    mu_y, var_y = posterior(params, gp_priors, X, y, Xtest, True, False)

    # Uncertainty
    uncertainty = 1.96 * jnp.sqrt(var_y.squeeze())

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].scatter(X, y, c="red", label="Training Data")
    ax[0].plot(
        Xtest.squeeze(),
        mu_y.squeeze(),
        label=r"Predictive Mean",
        color="black",
        linewidth=3,
    )
    ax[0].fill_between(
        Xtest.squeeze(),
        mu_y.squeeze() + uncertainty,
        mu_y.squeeze() - uncertainty,
        alpha=0.3,
        color="darkorange",
        label=f"Predictive Std (95% Confidence)",
    )
    ax[0].legend(fontsize=12)
    ax[1].plot(losses, label="losses")
    plt.tight_layout()
    fig.savefig("figures/jaxgp/examples/1d_example.png")
    plt.show()


if __name__ == "__main__":
    main()
