import jax.numpy as jnp


def zero_mean(x):
    return jnp.zeros(x.shape[0])
