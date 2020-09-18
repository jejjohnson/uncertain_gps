from typing import Tuple

import jax.numpy as jnp
import numpy as onp
from sklearn.utils import check_random_state


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

def near_square_wave(
    n_train: int=80, 
    input_noise: float=0.15, 
    output_noise: float=0.3, 
    n_test: int=400,
    random_state: int=123,
):
    
    # function
    f = lambda x: jnp.sin(1.0 * jnp.pi / 1.6 * jnp.cos(5 + .5 * x))
    
    # create clean inputs
    x_mu = jnp.linspace(-10, 10, n_train)
    
    # clean outputs
    y = f(x_mu)
    
    # generate noise
    x_rng = check_random_state(random_state)
    y_rng = check_random_state(random_state + 1)
    
    # noisy inputs
    x = x_mu + input_noise * x_rng.randn(x_mu.shape[0])
    
    # noisy outputs
    y = f(x_mu) + output_noise * y_rng.randn(x_mu.shape[0])
    
    # test points
    x_test = jnp.linspace(-12, 12, n_test) + x_rng.randn(n_test)
    y_test = f(jnp.linspace(-12, 12, n_test))
    x_test = jnp.sort(x_test)
    
    return x[:, None], y[:, None], x_test[:, None], y_test