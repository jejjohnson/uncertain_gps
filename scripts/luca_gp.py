# Author: Lucas Rath (lucasrm25@gmail.com)
# JAX implementation inspired on https://github.com/google/jax/blob/master/examples/gaussian_process_regression.py

"""
    Gaussian Process implementation using JAX
"""

from jax import grad, value_and_grad, jit, vmap, random, nn
import jax.numpy as np
import jax.scipy as scipy
import numpy as onp
import matplotlib.pyplot as plt


def mu(x):
    """ zero mean function: mean[f(x)] = 0
    Args:
        x: <N,n>
    Returns:
        mean: <N,1>
    """
    return np.zeros([np.size(x, 0), 1])


@jit
def SEkernel(params, x1, x2):
    """ Squared Exponential (SE) kernel
    """
    return params["var_f"] * np.exp(
        -0.5 * (x1 - x2).T @ np.linalg.solve(params["M"], (x1 - x2))
    )


def K(params, x1, x2):
    """Gramm matrix: cov[f(x1),f(x2)] = k(x1,x2) = var_f * exp( - 0.5 * ||x1-x2||^2_M )
    
    Args:
        x1: <N1,n>
        x2: <N2,n>
    Returns:
        kernel: <p,N1,N2>
    """
    # M = params['M']
    mapx1 = vmap(lambda x1, x2: SEkernel(params, x1, x2), in_axes=(0, None), out_axes=0)
    mapx2 = vmap(lambda x1, x2: mapx1(x1, x2), in_axes=(None, 0), out_axes=1)
    return mapx2(x1, x2)


def cholfac(params, X, Y):
    """ helper matrices [Rasmussen, pg19]
    """
    N = np.size(Y)
    KXX = K(params, X, X)
    L = scipy.linalg.cholesky(KXX + np.eye(N) * (params["var_n"] + 1e-5), lower=True)
    alpha = scipy.linalg.solve_triangular(
        L.T, scipy.linalg.solve_triangular(L, Y - mu(X), lower=True)
    )
    return L, alpha


def prior(params, x):
    return mu(x), K(params, x, x)


@jit
def posterior(params, X, Y, x):
    """ Evaluate GP posterior at points x.
    This is a fast implementation from [Rasmussen, pg19]
    Args:
        x: <Nx,n> point coordinates
    Returns:
        muy:  <Nx,n>    E[Y] = E[gp(x)]
        vary: <Nx,1>      Var[Y]  = Var[gp(x)]
    """
    Kxx = K(params, x, x)
    KxX = K(params, x, X)
    L, alpha = cholfac(params, X, Y)  # calculate Gram matrices
    mu_y = mu(x) + KxX @ alpha
    v = scipy.linalg.solve_triangular(L, KxX.T, lower=True)
    var_y = Kxx - v.T @ v
    return mu_y, var_y


def marginal_likelihood(params, X, Y):
    """ calculate the log likelihood: log(p(Y|X,theta)),
        where theta are the hyperparameters and (X,Y) the training data
        This is a fast implementation from [Rasmussen, pg19]
    """
    L, alpha = cholfac(params, X, Y)
    n = X.shape[1]
    return (
        -(0.5 * Y.T @ alpha).squeeze()
        - np.sum(np.log(np.diag(L)))
        - n / 2 * np.log(2 * np.pi)
    )


def optimize_hyperparams(
    params, X, Y, lr=0.01, tol=1e-6, maxsteps=1000, verbose=False
) -> dict:
    """ Maximize log likelihood: log(p(Y|X,theta)) w.r.t. the hyperparameters theta
        
        Returns:
            optimized hyperparameters
    """

    def saturate_params(p):
        return {k: nn.softplus(v) for k, v in p.items()}

    val_grad_fun = jit(
        value_and_grad(lambda p, X, Y: marginal_likelihood(saturate_params(p), X, Y))
    )

    paramsOpt = params.copy()
    old_ll = np.inf
    for i in range(maxsteps):
        ll, grads = val_grad_fun(
            paramsOpt, X, Y
        )  # evaluate loglikelihood and gradients
        for k in paramsOpt:
            paramsOpt[k] += lr * grads[k]  # gradient ascent

        if verbose and not i % 10:
            print("iter:{}    ll:{}".format(i, ll))
        if np.abs(old_ll - ll) <= tol:
            break
        old_ll = ll
    return saturate_params(paramsOpt)


def plotPosterior(ax, params, X, Y, x):
    assert np.size(X, 1) == 1, "method not implemented yet for input dimensions > 1"
    # evaluate posterior
    muY, varY = posterior(params, X, Y, x)
    stdY = np.sqrt(varY).diagonal()
    # plot
    if ax is None:
        ax = plt.gca()
    ax.scatter(X, Y)
    ax.plot(x, muY, color="purple")
    ax.fill_between(
        x.flatten(),
        muY.flatten() - 2 * stdY,
        muY.flatten() + 2 * stdY,
        facecolor="purple",
        alpha=0.3,
    )


def main():
    """ Test methods
    """
    # observations
    def truefun(x):
        return np.sin(x) + np.log(x)

    Xo = np.array([[1, 2, 3, 5, 6, 7]]).T
    Yo = truefun(Xo)

    # init hyperparameters
    params = {
        "M": np.diag(np.array([3.0])),  # length scale}
        "var_f": 9.0,  # output variance
        "var_n": 1e-8,
    }  # measurement noise variance

    # predict GP posterior
    x = np.arange(1, 7, 0.1).reshape(-1, 1)
    muY, varY = prior(params, x)
    muY, varY = posterior(params, Xo, Yo, x)

    plotPosterior(None, params, Xo, Yo, x)
    plt.savefig("luca_gp_plot_before.png")
    plt.tight_layout()

    # optimize hyperparameters
    paramsOpt = optimize_hyperparams(
        params, Xo, Yo, lr=0.1, tol=1e-6, maxsteps=1000, verbose=True
    )
    print("Original hyperparameters: ", params)
    print("Optimized hyperparameters: ", paramsOpt)

    plotPosterior(None, params, Xo, Yo, x)
    plt.savefig("luca_gp_plot_after.png")
    plt.tight_layout()


if __name__ == "__main__":
    main()
