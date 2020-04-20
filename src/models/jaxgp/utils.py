from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp


def cholesky_factorization(K: jnp.ndarray, Y: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:

    # cho factor the cholesky
    L = jax.scipy.linalg.cho_factor(K, lower=True)

    # weights
    weights = jax.scipy.linalg.cho_solve(L, Y)

    return L, weights


def saturate(params):
    return {ikey: jax.nn.softplus(ivalue) for (ikey, ivalue) in params.items()}


# @partial(jax.jit, static_argnums=(0, 1, 2, 3))
def get_factorizations(
    params: Dict,
    prior_params: Tuple[Callable, Callable],
    X: jnp.ndarray,
    Y: jnp.ndarray,
    X_new: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, bool], jnp.ndarray]:

    (mu_func, cov_func) = prior_params

    # ==========================
    # 1. GP PRIOR
    # ==========================
    mu_x = mu_func(X)
    Kxx = cov_func(params, X, X)

    # ===========================
    # 2. CHOLESKY FACTORIZATION
    # ===========================
    L, alpha = cholesky_factorization(
        Kxx + (params["likelihood_noise"] + 1e-7) * jnp.eye(Kxx.shape[0]),
        Y - mu_x.reshape(-1, 1),
    )

    # ================================
    # 4. PREDICTIVE MEAN DISTRIBUTION
    # ================================

    return L, alpha
