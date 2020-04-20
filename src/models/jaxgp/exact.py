from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from src.models.jaxgp.utils import get_factorizations


def gp_prior(
    params: Dict, mu_f: Callable, cov_f: Callable, x: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return mu_f(x), cov_f(params, x, x)


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 5, 6))
def posterior(
    params: Dict,
    prior_params: Tuple[Callable, Callable],
    X: jnp.ndarray,
    Y: jnp.ndarray,
    X_new: jnp.ndarray,
    likelihood_noise: bool = False,
    return_cov: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    (mu_func, cov_func) = prior_params

    # ==============================
    # Get Factorizations (L, alpha)
    # ==============================
    L, alpha = get_factorizations(
        params=params, prior_params=prior_params, X=X, Y=Y, X_new=X_new,
    )

    # ================================
    # 4. PREDICTIVE MEAN DISTRIBUTION
    # ================================

    # calculate transform kernel
    KxX = cov_func(params, X_new, X)

    # Calculate the Mean
    mu_y = jnp.dot(KxX, alpha)

    # =====================================
    # 5. PREDICTIVE COVARIANCE DISTRIBUTION
    # =====================================
    v = jax.scipy.linalg.cho_solve(L, KxX.T)

    # Calculate kernel matrix for inputs
    Kxx = cov_func(params, X_new, X_new)

    cov_y = Kxx - jnp.dot(KxX, v)

    # Likelihood Noise
    if likelihood_noise is True:
        cov_y += params["likelihood_noise"]

    # return variance (diagonals of covaraince)
    if return_cov is not True:
        cov_y = jnp.diag(cov_y)

    return mu_y, cov_y


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def predictive_mean(
    params: Dict,
    prior_params: Tuple[Callable, Callable],
    X: jnp.ndarray,
    Y: jnp.ndarray,
    X_new: jnp.ndarray,
) -> jnp.ndarray:

    (_, cov_func) = prior_params

    # ==============================
    # Get Factorizations (L, alpha)
    # ==============================
    L, alpha = get_factorizations(
        params=params, prior_params=prior_params, X=X, Y=Y, X_new=X_new,
    )

    # ================================
    # 4. PREDICTIVE MEAN DISTRIBUTION
    # ================================

    # calculate transform kernel
    KxX = cov_func(params, X_new, X)

    # Calculate the Mean
    mu_y = jnp.dot(KxX, alpha)

    return mu_y.squeeze()


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 5, 6))
def predictive_variance(
    params: Dict,
    prior_params: Tuple[Callable, Callable],
    X: jnp.ndarray,
    Y: jnp.ndarray,
    X_new: jnp.ndarray,
    likelihood_noise: bool = False,
    return_cov: bool = False,
) -> jnp.ndarray:

    (mu_func, cov_func) = prior_params

    # ==============================
    # Get Factorizations (L, alpha)
    # ==============================
    L, alpha = get_factorizations(
        params=params, prior_params=prior_params, X=X, Y=Y, X_new=X_new,
    )

    # =====================================
    # 5. PREDICTIVE COVARIANCE DISTRIBUTION
    # =====================================

    # calculate transform kernel
    KxX = cov_func(params, X_new, X)

    v = jax.scipy.linalg.cho_solve(L, KxX.T)

    # Calculate kernel matrix for inputs
    Kxx = cov_func(params, X_new, X_new)

    cov_y = Kxx - jnp.dot(KxX, v)

    # Likelihood Noise
    if likelihood_noise is True:
        cov_y += params["likelihood_noise"]

    # return variance (diagonals of covaraince)
    if return_cov is not True:
        cov_y = jnp.diag(cov_y)
    return cov_y.squeeze()
