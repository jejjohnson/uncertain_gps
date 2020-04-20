import functools

import jax
import jax.numpy as jnp


# Gram Matrix
@functools.partial(jax.jit, static_argnums=(0))
def gram(func, params, x, y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)


# Squared Euclidean Distance Formula
@jax.jit
def sqeuclidean_distance(x: jnp.array, y: jnp.array) -> float:
    return jnp.sum((x - y) ** 2)


# Manhattan Distance
@jax.jit
def manhattan_distance(x: jnp.array, y: jnp.array) -> float:
    return jnp.sum(jnp.abs(x - y))


# RBF Kernel
@jax.jit
def rbf_kernel(params, x, y):
    return jnp.exp(-params["gamma"] * sqeuclidean_distance(x, y))


# ARD Kernel
@jax.jit
def ard_kernel(params, x, y):

    # divide by the length scale
    x = x / params["length_scale"]
    y = y / params["length_scale"]

    # return the ard kernel
    return params["var_f"] * jnp.exp(-sqeuclidean_distance(x, y))


# Rational Quadratic Kernel
@jax.jit
def rq_kernel(params, x, y):

    # divide by the length scale
    x = x / params["length_scale"]
    y = y / params["length_scale"]

    # return the ard kernel
    return params["var_f"] * jnp.exp(1 + sqeuclidean_distance(x, y)) ** (
        -params["scale_mixture"]
    )


# Periodic Kernel
@jax.jit
def periodic_kernel(params, x, y):
    return None
