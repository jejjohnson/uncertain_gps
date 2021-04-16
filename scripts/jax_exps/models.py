from typing import Callable, Tuple, Optional
from chex import Array
import jax
import jax.numpy as jnp
import jax.random as jr

# JAXKERN Package
jaxkern_root = "/home/emmanuel/code/jax_kern"
sys.path.append(str(jaxkern_root))
from jaxkern.gp.uncertain.mcmc import MCMomentTransform, init_mc_moment_transform
from jaxkern.gp.uncertain.unscented import UnscentedTransform, init_unscented_transform
from jaxkern.gp.uncertain.linear import init_taylor_transform, init_taylor_o2_transform


def egp_predictions(
    meanf: Callable,
    varf: Callable,
    Xtest: Array,
    input_cov: Optional[Array] = None,
    model: str = "exact",
    obs_noise: bool = True,
    **kwargs,
) -> Tuple[Array, Array]:

    # fix mean,var functions
    mf = lambda x: jnp.atleast_1d(meanf(x).squeeze())

    model = model.lower()

    Xtest = jnp.array(Xtest, dtype=jnp.float32)

    # initialize random state
    seed = kwargs.get("seed", 123)
    key = jr.PRNGKey(seed)

    # check input_cov shape
    if jnp.ndim(input_cov) == 3:
        cov_vmap = 0
    elif jnp.ndim(input_cov) == 2:
        cov_vmap = None
    else:
        raise ValueError(f"Invalid shape of cov mat: {jnp.ndim(input_cov)}")

    # Exact GP
    if model == "exact":
        mu = meanf(Xtest).squeeze()
        var = varf(Xtest).squeeze()

    elif model == "mc":
        # initialize MC model
        mc_transform = MCMomentTransform(**kwargs)

        # transformation

        mu, var = jax.vmap(
            mc_transform.predict_f, in_axes=(None, 0, cov_vmap, None), out_axes=(0, 0)
        )(mf, Xtest, input_cov, key)

    elif model == "sigma":
        pass
    elif model == "gh":
        pass

    elif model == "taylor_o1":
        pass
    elif model == "taylor_o2":
        pass
    elif model == "mm_mc":
        pass

    elif model == "mm_sigma":
        pass
    elif model == "mm_gh":
        pass
    else:
        raise ValueError(f"Unrecognized egp model: {model}")

    return mu, var
