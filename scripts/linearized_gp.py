import functools
from typing import Callable, Dict, Tuple
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import tqdm
from jax.experimental import optimizers

from src.models.jaxgp.data import get_data
from src.models.jaxgp.exact import predictive_mean, predictive_variance
from src.models.jaxgp.kernels import gram, rbf_kernel, ard_kernel
from src.models.jaxgp.loss import marginal_likelihood
from src.models.jaxgp.mean import zero_mean
from src.models.jaxgp.utils import cholesky_factorization, get_factorizations, saturate


def main(args):
    # sigma_inputs = 0.15
    input_cov = jnp.array([args.input_noise]).reshape(-1, 1)
    X, y, Xtest, ytest = get_data(
        N=args.num_train,
        input_noise=args.input_noise,
        output_noise=args.output_noise,
        N_test=args.num_test,
    )

    # PRIOR FUNCTIONS (mean, covariance)
    mu_f = zero_mean
    cov_f = functools.partial(gram, ard_kernel)
    gp_priors = (mu_f, cov_f)

    # Kernel, Likelihood parameters
    params = {
        # "gamma": 2.0,
        "length_scale": 1.0,
        "var_f": 1.0,
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
    n_epochs = 500 if not args.smoke_test else 2
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
    mu_y = predictive_mean(params, gp_priors, X, y, Xtest)
    var_y = predictive_variance(params, gp_priors, X, y, Xtest, True, False)

    # ===========================
    # 1st Order Taylor Expansion
    # ===========================
    mean_f = functools.partial(predictive_mean, params, gp_priors, X, y)
    pred_grad_f = jax.jit(jax.vmap(jax.grad(mean_f, argnums=(0)), in_axes=(0)))

    dmu_y = pred_grad_f(Xtest)

    var_correction_o1 = jnp.diag(jnp.dot(jnp.dot(dmu_y, input_cov), dmu_y.T))

    # ===========================
    # 2nd Order Taylor Expansion
    # ===========================
    var_f = functools.partial(predictive_variance, params, gp_priors, X, y)

    pred_var_f = jax.jit(
        jax.vmap(jax.hessian(var_f, argnums=(0)), in_axes=(0, None, None))
    )

    d2var_y2 = pred_var_f(Xtest, True, False)

    d2var_y2 = jnp.dot(d2var_y2, input_cov)

    var_correction_o2 = jnp.trace(d2var_y2, axis1=1, axis2=2)

    # Uncertainty
    uncertainty = 1.96 * jnp.sqrt(var_y.squeeze())

    uncertainty_t1 = 1.96 * jnp.sqrt(var_y.squeeze() + var_correction_o1.squeeze())

    uncertainty_t2 = 1.96 * jnp.sqrt(
        var_y.squeeze() + var_correction_o1.squeeze() + var_correction_o2.squeeze()
    )

    fig, ax = plt.subplots(nrows=3, figsize=(5, 10))
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
    ax[0].set_ylim([-3.5, 3.5])
    ax[0].legend(fontsize=12)

    # =======================
    # CORRECTION (1st Order)
    # =======================
    ax[1].scatter(X, y, c="red", label="Training Data")
    ax[1].plot(
        Xtest.squeeze(),
        mu_y.squeeze(),
        label=r"Predictive Mean",
        color="black",
        linewidth=3,
    )
    ax[1].fill_between(
        Xtest.squeeze(),
        mu_y.squeeze() + uncertainty_t1,
        mu_y.squeeze() - uncertainty_t1,
        alpha=0.3,
        color="darkorange",
        label=f"Predictive Std Taylor 1st Order",
    )
    ax[1].set_ylim([-3.5, 3.5])
    ax[1].legend(fontsize=12)
    # =======================
    # CORRECTION (2nd Order)
    # =======================
    ax[2].scatter(X, y, c="red", label="Training Data")
    ax[2].plot(
        Xtest.squeeze(),
        mu_y.squeeze(),
        label=r"Predictive Mean",
        color="black",
        linewidth=3,
    )
    ax[2].fill_between(
        Xtest.squeeze(),
        mu_y.squeeze() + uncertainty_t2,
        mu_y.squeeze() - uncertainty_t2,
        alpha=0.3,
        color="darkorange",
        label=f"Predictive Std Taylor 2nd Order",
    )
    ax[2].set_ylim([-3.5, 3.5])
    ax[2].legend(fontsize=12)
    plt.tight_layout()
    fig.savefig("figures/jaxgp/examples/1d_example_egp.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GP w. Taylor Series Expansion Corrected Variances"
    )
    parser.add_argument("-n", "--num_train", default=30, type=int)
    parser.add_argument("-t", "--num_test", default=1000, type=int)
    parser.add_argument("-sm", "--smoke_test", action="store_true")
    parser.add_argument("-xn", "--input_noise", default=0.15, type=float)
    parser.add_argument("-yn", "--output_noise", default=0.15, type=float)
    args = parser.parse_args()
    main(args)
