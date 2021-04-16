import sys, os

from wandb.sdk import wandb_config

# # spyder up to find the root
# from pyprojroot import here
# root = here(project_files=[".here"])
# append to path
jaxkern_root = "/home/emmanuel/code/jax_kern"
sys.path.append(str(jaxkern_root))
isp_data_root = "/home/emmanuel/code/isp_data"
sys.path.append(str(isp_data_root))

from jaxkern.viz import plot_1D_GP
from jaxkern.gp.uncertain.mcmc import MCMomentTransform, init_mc_moment_transform
from jaxkern.gp.uncertain.unscented import UnscentedTransform, init_unscented_transform
from jaxkern.gp.uncertain.linear import init_taylor_transform, init_taylor_o2_transform
from jaxkern.gp.uncertain.quadrature import GaussHermite
from jaxkern.gp.uncertain.predict import moment_matching_predict_f

from isp_data.simulation import uncertain

# jax packages
import itertools

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.config import config
from jax import device_put
from jax import random
import numpy as np
import pandas as pd

# import chex
config.update("jax_enable_x64", False)


# logging
import tqdm
import wandb

# plot methods
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)
from argparse import ArgumentParser
import wandb

from typing import Callable, Tuple, Optional, Dict
from chex import Array, dataclass
import jax
import jax.numpy as jnp
import jax.random as jr

import uncertainty_toolbox as uct
import uncertainty_toolbox.viz as uviz


# ==========================
# HELPER FUNCTIONS
# ==========================


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
    if input_cov is None:
        pass
    elif jnp.ndim(input_cov) == 3:
        cov_vmap = 0
    elif jnp.ndim(input_cov) == 2:
        cov_vmap = None
    else:
        raise ValueError(f"Invalid shape of cov mat: {jnp.ndim(input_cov)}")

    # Exact GP
    if model == "exact":
        mu = meanf(Xtest)
        var = varf(Xtest)

    elif model == "mc":
        # initialize MC model
        mc_transform = MCMomentTransform(**kwargs)

        # transformation
        mu, var = jax.vmap(
            mc_transform.predict_f, in_axes=(None, 0, cov_vmap, None), out_axes=(0, 0)
        )(mf, Xtest, input_cov, key)

    elif model == "sigma":
        # initialize unscented transform
        unscented_transform = UnscentedTransform(**kwargs)

        # compute transformation
        mu, var = jax.vmap(
            unscented_transform.predict_f, in_axes=(None, 0, cov_vmap), out_axes=(0, 0)
        )(mf, Xtest, input_cov)
    elif model == "gh":
        # initialize GH transform
        mm_transform = GaussHermite(**kwargs)

        # compute transformation
        mu, var = jax.vmap(
            mm_transform.predict_f, in_axes=(None, 0, cov_vmap), out_axes=(0, 0)
        )(mf, Xtest, input_cov)

    elif model == "taylor_o1":

        taylor_o1_transform = init_taylor_transform(meanf=mf, varf=varf)

        mu, var = jax.vmap(taylor_o1_transform, in_axes=(0, cov_vmap), out_axes=(0, 0))(
            Xtest, input_cov
        )

    elif model == "taylor_o2":
        taylor_o2_transform = init_taylor_transform(meanf=mf, varf=varf)

        mu, var = jax.vmap(taylor_o2_transform, in_axes=(0, cov_vmap), out_axes=(0, 0))(
            Xtest, input_cov
        )
    else:
        raise ValueError(f"Unrecognized egp model: {model}")

    return mu, var


def egp_mm_predictions(
    posterior: Callable,
    params: Dict,
    training_ds: dataclass,
    Xtest: Array,
    input_cov: Optional[Array] = None,
    model: str = "mc",
    obs_noise: bool = True,
    **kwargs,
) -> Tuple[Array, Array]:
    # check input_cov shape
    if input_cov is None:
        pass
    elif jnp.ndim(input_cov) == 3:
        cov_vmap = 0
    elif jnp.ndim(input_cov) == 2:
        cov_vmap = None
    else:
        raise ValueError(f"Invalid shape of cov mat: {jnp.ndim(input_cov)}")

    if model == "mc":
        # initialize MC model
        mm_transform = MCMomentTransform(**kwargs)

    elif model == "sigma":
        # initialize unscented transform
        mm_transform = UnscentedTransform(**kwargs)
    elif model == "gh":
        mm_transform = GaussHermite(**kwargs)
    else:
        raise ValueError(f"Unrecognized egp model: {model}")

    # initialize moment matching prediction function
    mm_predict_f = moment_matching_predict_f(
        posterior, learned_params, training_ds, mm_transform, obs_noise=obs_noise
    )

    mm_mean_f = jax.vmap(mm_predict_f, in_axes=(0, cov_vmap))

    mu, var = mm_mean_f(Xtest, input_cov)
    return mu, var


def get_uncertainty_stats(mu, var, ytest, name: str, logger: Optional = None):

    # METRICS
    # Compute all uncertainty metrics
    metrics = uct.metrics.get_all_metrics(
        np.array(mu).squeeze(),
        np.array(jnp.sqrt(var)).squeeze(),
        np.array(ytest).squeeze(),
        verbose=False,
    )
    if logger is not None:
        logger.log({f"mae": metrics["accuracy"]["mae"]})
        logger.log({f"rmse": metrics["accuracy"]["rmse"]})
        logger.log({f"marpd": metrics["accuracy"]["marpd"]})
        logger.log({f"mdae": metrics["accuracy"]["mdae"]})
        logger.log({f"r2": metrics["accuracy"]["r2"]})
        logger.log({f"corr": metrics["accuracy"]["corr"]})
        logger.log({f"nll": metrics["scoring_rule"]["nll"]})
        # calibration metrics
        logger.log({f"rms_cal": metrics["avg_calibration"]["rms_cal"]})
        logger.log({f"ma_cal": metrics["avg_calibration"]["ma_cal"]})
        logger.log({f"miscal_area": metrics["avg_calibration"]["miscal_area"]})

    return None


def plot_uncertainty_graphs(
    mu, var, ytest, name: str, n_subset: int = 100, logger: Optional = None
):

    # PLOT INTERVALS
    uviz.plot_intervals(
        np.array(mu).squeeze(),
        np.array(jnp.sqrt(var)).squeeze(),
        np.array(ytest).squeeze(),
        n_subset=n_subset,
    )
    plt.tight_layout()
    if logger is not None:
        logger.log({f"intervals": wandb.Image(plt)})

    # ORDERED INTERVALS
    uviz.plot_intervals_ordered(
        np.array(mu).squeeze(),
        np.array(jnp.sqrt(var)).squeeze(),
        np.array(ytest).squeeze(),
        n_subset=n_subset,
    )
    plt.tight_layout()
    if logger is not None:
        logger.log({f"ordered_intervals": wandb.Image(plt)})

    # CALIBRATION CURVE
    uviz.plot_calibration(
        np.array(mu).squeeze(),
        np.array(jnp.sqrt(var)).squeeze(),
        np.array(ytest).squeeze(),
    )
    plt.tight_layout()
    if logger is not None:
        logger.log({f"calibration": wandb.Image(plt)})

    return None


def get_simulated_data(y_var: str = "lai"):

    X, y = uncertain.load_prosail_training_df()

    return X, y


def _select_y_variable_simulated(y: pd.DataFrame, y_var: str = "lai") -> pd.DataFrame:
    y_var = y_var.lower()

    if y_var == "lai":
        y = y[["LAI"]]
    elif y_var == "fapar":
        y = y[["FAPAR"]]
    else:
        raise NotImplementedError(f"Variable selected is not implemented yet: {y_var}")

    return y


def _select_y_variable_real(y: pd.DataFrame, y_var: str = "lai") -> pd.DataFrame:
    y_var = y_var.lower()

    if y_var == "lai":
        y = y[["Lai_mean"]]
    elif y_var == "fapar":
        y = y[["Fpar_mean"]]
    else:
        raise NotImplementedError(f"Variable selected is not implemented yet: {y_var}")

    return y


def get_real_data():

    ds_real = uncertain.load_histarfm_testing_df()

    # drop nans
    ds_real = ds_real.dropna()

    # parse
    Xreal_mu, Xreal_std, Y_real = uncertain.parse_histarfm_ds(ds_real)

    # apply scaling
    Xreal_mu *= 0.0001
    Xreal_std *= 0.5 / 255
    return Xreal_mu, Xreal_std, Y_real


# def plot_uncertainty_metrics(n_subset: int = 100, logger: Optional=None):

#     uviz.plot_intervals(
#         np.array(y_pred_real),
#         np.array(y_std_real),
#         np.array(Y_real_scaled).squeeze(),
#         n_subset=100,
#     )
#     if logger:
#         wandb.log({""})
#     return None


# ==========================
# PARAMETERS
# ==========================

parser = ArgumentParser(
    description="2D Data Demo with Iterative Gaussianization method"
)

# ======================
# Dataset
# ======================
parser.add_argument(
    "--seed", type=int, default=123, help="number of data points for training",
)
parser.add_argument(
    "--dataset", type=str, default="real", help="Dataset to be used for visualization",
)
parser.add_argument(
    "--y_var",
    type=str,
    default="lai",
    help="The biophysical parameter to be predicted",
)
parser.add_argument(
    "--n_train", default=1_000, type=int, help="number of data points for training",
)
parser.add_argument(
    "--n_test", default=1_000, type=int, help="number of data points for training",
)


# ======================
# Model Training
# ======================
parser.add_argument("--epochs", type=int, default=2_000, help="Number of batches")
parser.add_argument(
    "--learning_rate", type=float, default=0.005, help="Number of batches"
)
# ======================
# PREDICTION PARAMS
# ======================
parser.add_argument(
    "--pred_model", type=str, default="standard", help="Prediction Model"
)
parser.add_argument("--square", type=bool, default=True, help="Prediction Model")

# ======================
# MC Parameters
# ======================
parser.add_argument("--mc_samples", type=int, default=100, help="Number of batches")
# ======================
# Sigma Parameters
# ======================
parser.add_argument("--alpha", type=float, default=1.0, help="Number of batches")
parser.add_argument("--beta", type=float, default=2.0, help="Number of batches")
parser.add_argument("--kappa", type=float, default=None, help="Number of batches")
# ======================
# GH Parameters
# ======================
parser.add_argument("--degree", type=int, default=20, help="Number of batches")

# ======================
# Logger Parameters
# ======================
parser.add_argument("--exp_name", type=str, default="simreal_landsat")
parser.add_argument("--wandb-entity", type=str, default="ipl_uv")
parser.add_argument("--wandb-project", type=str, default="egp1_1")
# =====================
# Testing
# =====================
parser.add_argument(
    "-sm",
    "--smoke-test",
    action="store_true",
    help="to do a smoke test without logging",
)

args = parser.parse_args()
# change this so we don't bug wandb with our BS
if args.smoke_test:
    os.environ["WANDB_MODE"] = "dryrun"
    args.epochs = 1
    args.n_train = 150

# ==========================
# INITIALIZE LOGGER
# ==========================

wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity,)
wandb_logger.config.update(args)
config = wandb_logger.config


# ===================================
# SIMULATED DATA
# ===================================

# load data
X, y = get_simulated_data()

# select variable
y = _select_y_variable_simulated(y, y_var=config.y_var)

# TRAIN/TEST SPLIT
from sklearn.model_selection import train_test_split


Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, train_size=args.n_train, random_state=config.seed,
)

# STANDARDIZATION
from sklearn.preprocessing import StandardScaler

x_transformer = StandardScaler()

Xtrain_scaled = x_transformer.fit_transform(Xtrain)
Xtest_scaled = x_transformer.transform(Xtest)

y_transformer = StandardScaler(with_std=False)

ytrain_scaled = y_transformer.fit_transform(ytrain.values.reshape(-1, 1))
ytest_scaled = y_transformer.transform(ytest.values.reshape(-1, 1))


key = jax.random.PRNGKey(config.seed)


# ==========================
# GP MODEL
# ==========================
from gpjax.gps import Prior
from gpjax.mean_functions import Zero
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.types import Dataset


# GP Prior
mean_function = Zero()
kernel = RBF()
prior = Prior(mean_function=mean_function, kernel=kernel)

# GP Likelihood
lik = Gaussian()

# GP Posterior
posterior = prior * lik

# initialize training dataset
training_ds = Dataset(X=Xtrain_scaled, y=ytrain_scaled)

# PARAMETERS
from gpjax.parameters import initialise
import numpyro.distributions as dist
from gpjax.interfaces.numpyro import numpyro_dict_params, add_constraints


# initialize parameters
params = initialise(posterior)


hyperpriors = {
    "lengthscale": jnp.ones(X.shape[1]),
    "variance": 1.0,
    "obs_noise": 0.01,
}


# convert to numpyro-style params
numpyro_params = numpyro_dict_params(hyperpriors)

# convert to numpyro-style params
numpyro_params = add_constraints(numpyro_params, dist.constraints.softplus_positive)

# INFERENCE
from numpyro.infer.autoguide import AutoDelta
from gpjax.interfaces.numpyro import numpyro_marginal_ll, numpyro_dict_params


# initialize numpyro-style GP model
npy_model = numpyro_marginal_ll(posterior, numpyro_params)

# approximate posterior
guide = AutoDelta(npy_model)

# TRAINING
import numpyro
from numpyro.infer import SVI, Trace_ELBO

# reproducibility
key, opt_key = jr.split(key, 2)
n_iterations = config.epochs
lr = config.learning_rate

# numpyro specific optimizer
optimizer = numpyro.optim.Adam(step_size=lr)

# stochastic variational inference (pseudo)
svi = SVI(npy_model, guide, optimizer, loss=Trace_ELBO())
svi_results = svi.run(opt_key, n_iterations, training_ds)
wandb.log({"losses": np.array(svi_results.losses)})

# Learned Params
learned_params = svi_results.params
p = jax.tree_map(lambda x: np.array(x), learned_params)
wandb.log(p)


# ==============================
# PREDICTIONS
# ==============================
from gpjax import mean, variance

meanf = mean(posterior, learned_params, training_ds)
covarf = variance(posterior, learned_params, training_ds)
varf = lambda x: jnp.atleast_1d(jnp.diag(covarf(x)))

# predictions
y_mu_train = meanf(jnp.array(Xtrain_scaled))
y_mu_test = meanf(jnp.array(Xtest_scaled))

# predictive variance
y_var_train = varf(jnp.array(Xtrain_scaled))
y_var_test = varf(jnp.array(Xtest_scaled))

# rescale data
y_mu_train = y_transformer.inverse_transform(y_mu_train)
y_mu_test = y_transformer.inverse_transform(y_mu_test)
# rescale data
y_var_train = y_transformer.inverse_transform(y_var_train)
y_var_test = y_transformer.inverse_transform(y_var_test)


# ==============================
# REAL DATA - EXACT
# ==============================

# load real data
Xreal_mu, Xreal_std, Y_real = get_real_data()

assert Xreal_mu.ndim == 2
assert Xreal_std.ndim == 2


Y_real = _select_y_variable_real(Y_real, y_var=config.y_var)

Y_real = Y_real.values

assert Y_real.ndim == 2


# SUBSAMPLE
Xreal_mu, _, Xreal_std, _, Y_real, _ = train_test_split(
    Xreal_mu, Xreal_std, Y_real, train_size=args.n_test, random_state=config.seed,
)


# SCALE DATA
Xreal_mu_scaled = x_transformer.transform(Xreal_mu)
Xreal_std_scaled = x_transformer.transform(Xreal_std)

if args.smoke_test:
    print(Xreal_mu_scaled.min(), Xreal_mu_scaled.max())
    print(Xtrain_scaled.min(), Xtrain_scaled.max())
    print(Xreal_std_scaled.min(), Xreal_std_scaled.max())

Y_real_scaled = y_transformer.transform(Y_real)


# ==============================
# PREDICTIONS
# ==============================

# PREDICTIONS
y_mu_real = meanf(jnp.array(Xreal_mu_scaled)).squeeze()
y_var_real = varf(jnp.array(Xreal_mu_scaled)).squeeze()


# ==============================
# PREDICTIONS (NOISY)
# ==============================

input_cov = list(map(lambda x: jnp.diag(x) ** 2, Xreal_std_scaled))
input_cov = jnp.stack(input_cov)

# convert to jax array
X_noisy = jnp.array(Xreal_mu_scaled, dtype=jnp.float64)
input_cov = jnp.array(input_cov, dtype=jnp.float64)

final_params = dict((k, jnp.array(v, jnp.float64)) for k, v in learned_params.items())

if config.square:
    input_cov = jnp.sqrt(input_cov)

if args.smoke_test:
    input_cov = 0.001 * jnp.ones_like(input_cov)

assert input_cov.shape[1:] == (6, 6)
model_name = str(config.pred_model)


mf = lambda x: jnp.atleast_1d(meanf(x).squeeze())
n_features = X_noisy.shape[1]
mc_samples = config.mc_samples
covariance = False
import time

t0 = time.time()

assert X_noisy.shape[1] == 6
if model_name.lower() == "exact":

    mu, var = egp_predictions(meanf, varf, Xtest=X_noisy, model="exact")

elif model_name.lower() == "mc":

    mu, var = egp_predictions(
        meanf,
        varf,
        Xtest=X_noisy,
        input_cov=input_cov,
        model="mc",
        n_features=n_features,
        n_samples=mc_samples,
        seed=config.seed,
    )
elif model_name.lower() == "sigma":
    mu, var = egp_predictions(
        meanf,
        varf,
        Xtest=X_noisy,
        input_cov=input_cov,
        model="sigma",
        n_features=n_features,
        alpha=config.alpha,
        beta=config.beta,
        kappa=config.kappa,
    )
elif model_name.lower() == "gh":
    mu, var = egp_predictions(
        meanf,
        varf,
        Xtest=X_noisy,
        input_cov=input_cov,
        model="gh",
        n_features=n_features,
        degree=config.degree,
    )
elif model_name.lower() == "taylor_o1":
    mu, var = egp_predictions(
        meanf,
        varf,
        Xtest=X_noisy,
        input_cov=input_cov,
        model="taylor_o1",
        n_features=n_features,
    )
elif model_name.lower() == "taylor_o2":
    mu, var = egp_predictions(
        meanf,
        varf,
        Xtest=X_noisy,
        input_cov=input_cov,
        model="taylor_o2",
        n_features=n_features,
    )

elif model_name.lower() == "mm_mc":
    mu, var = egp_mm_predictions(
        posterior=posterior,
        params=learned_params,
        training_ds=training_ds,
        Xtest=X_noisy,
        input_cov=input_cov,
        model="mc",
        n_features=n_features,
        n_samples=mc_samples,
        seed=config.seed,
    )
elif model_name.lower() == "mm_sigma":
    mu, var = egp_mm_predictions(
        posterior=posterior,
        params=learned_params,
        training_ds=training_ds,
        Xtest=X_noisy,
        input_cov=input_cov,
        model="sigma",
        n_features=n_features,
        alpha=config.alpha,
        beta=config.beta,
        kappa=config.kappa,
    )
elif model_name.lower() == "mm_gh":
    mu, var = egp_mm_predictions(
        posterior=posterior,
        params=learned_params,
        training_ds=training_ds,
        Xtest=X_noisy,
        input_cov=input_cov,
        model="gh",
        n_features=n_features,
        degree=config.degree,
    )
else:
    raise ValueError(f"Unrecognized model: {model_name.lower()}")

t1 = time.time() - t0
wandb.log({"pred_time": t1})


if args.smoke_test:
    print("Y REAL")
    print(mu.min(), mu.max())
    print(Y_real_scaled.min(), Y_real_scaled.max())
    print("Y Var")
    print(var.min(), var.max())

# rescale data
y_mu_real = y_transformer.inverse_transform(mu)
y_var_real = y_transformer.inverse_transform(var)

assert y_var_real.ndim == 2
assert y_mu_real.ndim == 2
assert Y_real.ndim == 2

if args.smoke_test:
    print("Y REAL (Unscaled)")

    print(Y_real.min(), Y_real.max())
    print(y_mu_real.min(), y_mu_real.max())
    print("Y Var (Unscaled)")
    print(y_var_real.min(), y_var_real.max())

# METRICS
get_uncertainty_stats(y_mu_real, y_var_real, Y_real, name="real", logger=wandb)

# GRAPHS
plot_uncertainty_graphs(
    y_mu_real, y_var_real, Y_real, name="real", n_subset=100, logger=wandb
)
