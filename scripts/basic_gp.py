import functools
import tqdm
import jax
from jax.experimental import optimizers
import numpy as onp
import jax.numpy as jnp

# Plotting libraries
import matplotlib.pyplot as plt


def get_data(N=30, sigma_obs=0.15, N_test=400):
    onp.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    Y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
    Y += sigma_obs * onp.random.randn(N)
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = jnp.linspace(-1.3, 1.3, N_test)

    return X[:, None], Y, X_test[:, None], None


# observations


# def get_data(N=30):
#     def truefun(x):
#         return jnp.sin(x) + jnp.log(x)

#     Xo = jnp.linspace(1, 7, N)[:, None]
#     Yo = truefun(Xo)
#     x = jnp.arange(1, 7, int(N * 2)).reshape(-1, 1)
#     return Xo, Yo.squeeze(), x, None


# def get_data(n_points=100):
#     rng = onp.random.RandomState(0)

#     # Generate sample data
#     X = 15 * rng.rand(100, 1)
#     y = np.sin(X).ravel()
#     y += 3 * (0.5 - rng.rand(X.shape[0]))  # add noise
# return X, y


# def get_data(numpts=7):
#     key = jax.random.PRNGKey(0)
#     # Create a really simple toy 1D function
#     y_fun = lambda x: jnp.sin(x) + 0.1 * jax.random.normal(key, shape=(x.shape[0], 1))
#     x = (jax.random.uniform(key, shape=(numpts, 1)) * 4.0) + 1
#     # print(x.shape)
#     y = y_fun(x).squeeze()
#     xtest = jnp.linspace(0, 6.0, 200)[:, None]
#     ytest = y_fun(xtest)
#     return x, y, xtest, ytest


# Squared Euclidean Distance Formula
def sqeuclidean_distance(x, y):

    return jnp.sum((x - y) ** 2)


# RBF Kernel
def rbf_kernel(params, x, y):

    # return the rbf kernel
    return params["var_f"] * jnp.exp(-params["length_scale"] * jnp.sum((x - y) ** 2))


# ARD Kernel
def ard_kernel(params, x, y):

    # # divide by the length scale
    # x = x / params["length_scale"]
    # y = y / params["length_scale"]

    x = x / params["length_scale"]
    y = y / params["length_scale"]

    # return the ard kernel
    k = params["var_f"] * jnp.exp(-0.5 * sqeuclidean_distance(x, y))

    return k


def covariance_matrix(params, x, y):
    mapx1 = jax.vmap(
        lambda x, y: ard_kernel(params, x, y), in_axes=(0, None), out_axes=0
    )
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)


def zero_mean(x):
    return jnp.zeros(x.shape[0])


def gp_prior(params, mu_f, cov_f, x):
    return mu_f(x), cov_f(params, x, x)


def cholesky_factorization(K, Y):

    # cho factor the cholesky
    L = jax.scipy.linalg.cho_factor(K, lower=True)

    # weights
    weights = jax.scipy.linalg.cho_solve(L, Y)

    return L, weights


def posterior(params, gp_priors, X, Y, X_new):

    (mu_func, cov_func) = gp_priors

    # ==========================
    # 1. GP PRIOR
    # ==========================
    mu_x, Kxx = gp_prior(params, mu_f=mu_func, cov_f=cov_func, x=X)

    # check outputs
    assert mu_x.shape == (X.shape[0],), f"{mu_x.shape} =/= {(X.shape[0],)}"
    assert Kxx.shape == (
        X.shape[0],
        X.shape[0],
    ), f"{Kxx.shape} =/= {(X.shape[0],X.shape[0])}"

    # ===========================
    # 2. CHOLESKY FACTORIZATION
    # ===========================

    # 1 STEP
    print(f"Problem: {Kxx.shape},{Y.shape}")
    (L, lower), alpha = cholesky_factorization(
        Kxx + (params["likelihood_noise"] + 1e-6) * jnp.eye(Kxx.shape[0]), Y
    )

    assert L.shape == (
        X.shape[0],
        X.shape[0],
    ), f"L:{L.shape} =/= X..:{(X.shape[0],X.shape[0])}"
    assert alpha.shape == (X.shape[0],), f"alpha: {alpha.shape} =/= X: {X.shape[0]}"

    # ================================
    # 4. PREDICTIVE MEAN DISTRIBUTION
    # ================================

    # calculate transform kernel
    KxX = cov_func(params, X_new, X)
    assert KxX.shape == (
        X_new.shape[0],
        X.shape[0],
    ), f"{KxX.shape} =/= {(X.shape[0],X_new.shape[0])}"

    # Project data
    mu_y = mu_func(X_new) + jnp.dot(KxX, alpha)

    assert mu_y.shape == X_new.shape

    # =====================================
    # 5. PREDICTIVE COVARIANCE DISTRIBUTION
    # =====================================
    #     print(f"K_xX: {KXx.T.shape}, L: {L.shape}")
    print(f"L: {L.shape}, KxX: {KxX.shape}", lower)
    v = jax.scipy.linalg.cho_solve((L, lower), KxX.T)
    print(f"v:", v.shape)

    assert v.shape == (
        X.shape[0],
        X_new.shape[0],
    ), f"v: {v.shape} =/= {(X_new.shape[0])}"

    cov_y = cov_func(params, X_new, X_new)

    assert cov_y.shape == (X_new.shape[0], X_new.shape[0])

    cov_y = cov_y - jnp.dot(v.T, v)

    assert cov_y.shape == (X_new.shape[0], X_new.shape[0])

    # TODO: Bug here for vmap...

    # # =====================================
    # # 6. PREDICTIVE VARIANCE DISTRIBUTION
    # # =====================================

    # Linv = jax.scipy.linalg.solve_triangular(L.T, jnp.eye(L.shape[0]), lower=lower)

    # var_y = jnp.diag(cov_func(params, X_new, X_new))

    # var_y = var_y + jnp.einsum("ij, jk, ki->i", KxX, jnp.dot(Linv, Linv.T), KxX.T)

    return mu_y, cov_y


def marginal_likelihood(prior_funcs, params, Xtrain, Ytrain):

    # unpack params
    (mu_func, cov_func) = prior_funcs

    # ==========================
    # 1. GP Prior
    # ==========================
    mu_x = mu_func(Xtrain)
    # print(Xtrain.shape, Xtrain.shape)
    Kxx = cov_func(params, Xtrain, Xtrain)
    #     print("MLL (GPPR):", Xtrain.shape, Ytrain.shape)
    #     mu_x, Kxx = gp_prior(params, mu_f=mu_func, cov_f=cov_func , x=Xtrain)

    # ===========================
    # 2. GP Likelihood
    # ===========================
    # print(Kxx.shape,)
    K_gp = Kxx + (params["likelihood_noise"] + 1e-6) * jnp.eye(Kxx.shape[0])
    # print(K_gp.shape)
    #     print("MLL (GPLL):", Xtrain.shape, Ytrain.shape)

    # ===========================
    # 3. Built-in GP Likelihood
    # ===========================
    return jax.scipy.stats.multivariate_normal.logpdf(Ytrain, mean=mu_x, cov=K_gp)

    # Nice Trick for better training of params


def nll_scratch(prior_funcs, params, X, Y) -> float:

    # unpack params
    (mu_func, cov_func) = prior_funcs

    # ==========================
    # 1. GP PRIOR
    # ==========================
    mu_x = mu_func(X)

    Kxx = cov_func(params, X, X)

    # ===========================
    # 2. CHOLESKY FACTORIZATION
    # ===========================
    #     print(f"Problem:", X.shape, Y.shape, Kxx.shape)
    #     print(f"Y: {Y.shape}, Kxx: {Kxx.shape}")

    (L, _), alpha = cholesky_factorization(
        Kxx + (params["likelihood_noise"] + 1e-6) * jnp.eye(Kxx.shape[0]), Y - mu_x
    )
    # print("L:", L.shape)
    log_likelihood_dims = -0.5 * (Y + alpha)  # jnp.sum("i,i->", Y, alpha)
    log_likelihood_dims -= jnp.log(L).sum(-1)
    log_likelihood_dims -= (Kxx.shape[0] / 2.0) * jnp.log(2.0 * jnp.pi)
    log_likelihood_dims -= jnp.sum(
        -0.5 * jnp.log(2 * 3.1415) - jnp.log(params["var_f"]) ** 2
    )
    # lognormal prior

    return log_likelihood_dims.sum(-1)


def soften(params):
    return {ikey: softplus(ivalue) for (ikey, ivalue) in params.items()}


def exp_params(params):
    return {ikey: jnp.exp(ivalue) for (ikey, ivalue) in params.items()}


# X, y = get_data()


def softplus(x):
    return jnp.logaddexp(x, 0.0)


def main():

    X, y, Xtest, _ = get_data(100)
    print(X.shape, y.shape, Xtest.shape)

    # MEAN FUNCTION
    mu_f = zero_mean

    # COVARIANCE FUNCTION
    params = {
        "length_scale": 1.0,
        "var_f": 1.0,
        "likelihood_noise": 1e-4,
    }
    # print("BEFORE:\n", params)
    # params = log_params(params)
    cov_f = covariance_matrix  # functools.partial(covariance_matrix, ard_kernel)
    gp_priors = (mu_f, cov_f)

    print("SOFTEN PARAMS:\n", soften(params))
    # print(soften(params))

    # m_ll_vectorized = jax.vmap(marginal_likelihood, in_axes=(None, None, 0, 0))

    # define an explicit loss function (with context variables)
    nll_loss = functools.partial(marginal_likelihood, gp_priors)

    def loss(params, Xtrain, ytrain):
        return -jnp.mean(
            jax.vmap(nll_loss, in_axes=(None, 0, 0))(soften(params), Xtrain, ytrain)
        )

    print("Pre Test", X.shape, y.shape, Xtest.shape)
    # grad function (BATCH)

    print("LOSS N GRAD")
    dloss = jax.value_and_grad(loss)
    # print(y.shape)
    nll, grads = dloss(params, X, y)

    assert nll.shape == ()
    print(nll)
    print(grads)
    # # print(nll)
    # # print(grads["length_scale"])
    # print("VMAP1")
    # dloss_vec = jax.jit(jax.vmap(dloss, in_axes=(None, 0, 0), out_axes=(0, 0)))
    # # dloss = jax.vmap(dloss, in_axes=(None, 0, 0), out_axes=(0, 0))
    # # print(y.shape)
    # nll, grads = dloss_vec(params, X, y)

    # assert nll.shape == (X.shape[0],)
    # print(nll)
    # print(grads)

    # print("VMAP2")
    # dloss_vec = jax.jit(jax.vmap(dloss, in_axes=(None, 0, 0)))
    # # dloss = jax.vmap(dloss, in_axes=(None, 0, 0), out_axes=(0, 0))
    # # print(y.shape)
    # nll, grads = dloss_vec(params, X, y)

    # assert nll.shape == (X.shape[0],)
    # print(nll)
    # print(grads)

    # print(nll.mean())
    # print(grads["length_scale"])
    # print(y.shape)

    @jax.jit
    def step(params, X, y, opt_state):
        # print("BEOFRE!")
        # print(X.shape, y.shape)
        # print("PARAMS", params)
        # print(opt_state)
        # value and gradient of loss function
        value, grads = dloss(params, X, y)
        # # print(f"VALUE:", value)
        # print("During! v", value)
        # print("During! p", params)
        # print("During! g", grads)
        # update parameter state
        opt_state = opt_update(0, grads, opt_state)

        # get new params
        params = get_params(opt_state)
        # print("AFTER! v", value)
        # print("AFTER! p", params)
        # print("AFTER! g", grads)
        return params, opt_state, value

    # initialize optimizer
    opt_init, opt_update, get_params = optimizers.sgd(step_size=1e-2)

    # initialize parameters
    opt_state = opt_init(params)

    # get initial parameters
    params = get_params(opt_state)
    print("PARAMS!", params)

    # momentums = dict([(k, p * 0.0) for k, p in params.items()])
    # scales = dict([(k, p * 0.0 + 1.0) for k, p in params.items()])
    # learning_rate = 0.01

    # def train_step(params, momentums, scales, x, y):
    #     nll, grads = dloss(params, x, y)
    #     for k in params:
    #         momentums[k] = 0.9 * momentums[k] + 0.1 * grads[k].mean()
    #         scales[k] = 0.9 * scales[k] + 0.1 * grads[k].mean() ** 2
    #         params[k] -= learning_rate * momentums[k] / jnp.sqrt(scales[k] + 1e-5)
    #     return params, momentums, scales, nll

    #

    # losses = list()

    # @jax.jit
    # def update(params, x, y):
    #     vals, grads = dloss(params, x, y)
    #     return (
    #         vals,
    #         [
    #             (iparam - learning_rate * igrad.mean())
    #             for iparam, igrad in zip(params, grads)
    #         ],
    #     )

    n_epochs = 10_000
    losses = list()
    with tqdm.trange(n_epochs) as bar:

        for i in bar:
            postfix = {}
            # get nll and grads
            # nll, grads = dloss(params, X, y)

            params, opt_state, value = step(params, X, y, opt_state)

            # update params
            # params, momentums, scales, nll = train_step(params, momentums, scales, X, y)
            for ikey in params.keys():
                postfix[ikey] = f"{params[ikey]:.4f}"
            # params[ikey] += learning_rate * grads[ikey].mean()

            losses.append(value.mean())
            postfix["Loss"] = f"{onp.array(losses[-1]):.4f}"
            bar.set_postfix(postfix)
            # break

    params = soften(params)

    # posterior_map = functools.partial(posterior, )
    posterior_vec = jax.vmap(
        posterior, in_axes=(None, None, None, None, 0), out_axes=(0, 0)
    )
    mu_y, var_y = posterior_vec(params, gp_priors, X, y, Xtest)

    # make plots
    fig, ax = plt.subplots(nrows=2)

    uncertainty = 1.96 * jnp.sqrt(var_y.squeeze())
    bounds = mu_y.squeeze() + uncertainty, mu_y.squeeze() - uncertainty

    # plot training data
    ax[0].plot(X, y, "kx")
    # plot 90% confidence level of predictions
    # ax[0].fill_between(Xtest.squeeze(), bounds[0], bounds[1], color="lightblue")
    # plot mean prediction
    ax[0].plot(Xtest, mu_y, "blue", ls="solid", lw=5.0)
    ax[0].set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

    ax[1].plot(losses, label="Losses")
    ax[1].set(title="Losses")

    plt.savefig("mygp_plot.png")
    plt.tight_layout()

    print(params)
    # print(losses)


if __name__ == "__main__":
    main()
