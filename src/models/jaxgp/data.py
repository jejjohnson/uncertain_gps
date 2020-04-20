from typing import Tuple

import jax.numpy as jnp
import numpy as onp


def get_data(
    N: int = 30,
    input_noise: float = 0.15,
    output_noise: float = 0.15,
    N_test: int = 400,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, None]:
    onp.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    Y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
    Y += output_noise * onp.random.randn(N)
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    X += input_noise * onp.random.randn(N)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = jnp.linspace(-1.2, 1.2, N_test)

    return X[:, None], Y[:, None], X_test[:, None], None
