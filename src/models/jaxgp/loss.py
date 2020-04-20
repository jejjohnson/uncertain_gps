from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0))
def marginal_likelihood(
    prior_params: Tuple[Callable, Callable],
    params: Dict,
    Xtrain: jnp.ndarray,
    Ytrain: jnp.ndarray,
) -> float:

    # unpack params
    (mu_f, cov_f) = prior_params

    # ==========================
    # 1. GP Prior, mu(), cov(,)
    # ==========================
    mu_x = mu_f(Ytrain)
    Kxx = cov_f(params, Xtrain, Xtrain)

    # ===========================
    # 2. GP Likelihood
    # ===========================
    K_gp = Kxx + (params["likelihood_noise"] + 1e-6) * jnp.eye(Kxx.shape[0])

    # ===========================
    # 3. Log Probability
    # ===========================
    log_prob = jax.scipy.stats.multivariate_normal.logpdf(
        x=Ytrain.T, mean=mu_x, cov=K_gp
    )

    # Negative Marginal log-likelihood
    return -log_prob.sum()
