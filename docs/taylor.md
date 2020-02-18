# Linearization (Taylor Expansions)

- [Linearization (Taylor Expansions)](#linearization-taylor-expansions)
  - [Gaussian Processes](#gaussian-processes)
    - [Model](#model)
    - [Predictions with stochastic inputs](#predictions-with-stochastic-inputs)
    - [Analytical Moments](#analytical-moments)
    - [Taylor Approximation](#taylor-approximation)
    - [Linearized Predictive Mean and Variance](#linearized-predictive-mean-and-variance)
      - [Practical Equations](#practical-equations)
  - [Taylor Expansion](#taylor-expansion)
  - [Approximate the Model](#approximate-the-model)
  - [Approximate The Posterior](#approximate-the-posterior)
    - [Expectation](#expectation)
    - [Variance](#variance)
    - [I: Additive Noise Model ($x,f$)](#i-additive-noise-model-mathsemanticsmrowmixmimo-separator%22true%22momifmimrowannotation-encoding%22applicationx-tex%22xfannotationsemanticsmathxf)
        - [Other GP Methods](#other-gp-methods)
    - [II: Non-Additive Noise Model](#ii-non-additive-noise-model)
    - [III: Quadratic Approximation](#iii-quadratic-approximation)
  - [Parallels to the Kalman Filter](#parallels-to-the-kalman-filter)
  - [Connections](#connections)
    - [KL-Divergence](#kl-divergence)
  - [Code](#code)
  - [Literature](#literature)
  - [Supplementary](#supplementary)
      - [Error Propagation](#error-propagation)
      - [Fubini's Theorem](#fubinis-theorem)
      - [Law of Iterated Expecations](#law-of-iterated-expecations)
      - [Conditional Variance](#conditional-variance)

## Gaussian Processes

---

### Model

Let's assume we have inputs with an additive noise term $\epsilon_x$ and let's assume that it is Gaussian distributed. We can write some expressions which are very similar to the GP model equations specified above. So we have a standard GP model.
$$
\begin{aligned}
y &= f(x) + \epsilon_y\\
\epsilon_y &\sim \mathcal{N}(0, \sigma_y^2) \\
\end{aligned}
$$

* **GP Prior**: $p(f|X)\sim\mathcal{N}(m(X), \mathbf{K})$
* **Gaussian Likelihood**: $p(y|f, X)=\mathcal{N}(y|f(x), \sigma_y^2\mathbf{I})$
* **Posterior**: $f \sim \mathcal{GP}(f|\mu_{GP}, \nu^2_{GP})$ 

And the predictive functions $\mu_{GP}$ and $\nu^2_{GP}$ are:

$$
\begin{aligned}
    \mu_\text{GP}(\mathbf{x_*}) &= k(\mathbf{x_*}) \, \mathbf{K}_{GP}^{-1}y=k(\mathbf{x_*}) \, \alpha \\
    \nu^2_\text{GP}(\mathbf{x_*}) &= k(\mathbf{x_*}, \mathbf{x_*}) - k(\mathbf{x_*}) \,\mathbf{K}_{GP}^{-1} \, k(\mathbf{x_*})^{\top}
\end{aligned}
$$

where $\mathbf{K}_\text{GP}=k(\mathbf{x,x}) + \sigma_y^2 \mathbf{I}$.

### Predictions with stochastic inputs


---

### Analytical Moments

We can compute the analytical Gaussian approximation by only computing the mean and the variance of the 

**Mean Function**

$$
\begin{aligned}
m(\mu_\mathbf{x}, \Sigma_\mathbf{x})
&=
\mathbb{E}_\mathbf{f_*}
\left[ f_* \, \mathbb{E}_\mathbf{x_*} \left[ p(f_*|\mathbf{x}_*) \right] \right] \\
&=
\mathbb{E}_\mathbf{x_*}
\left[ \mathbb{E}_{f_*} \left[ f_* \,p(f_* | \mathbf{x_*}) \right]\right]\\
&=
\mathbb{E}_{x_*}\left[ \mu_\text{GP}(\mathbf{x_*}) \right]
\end{aligned}
$$

**Variance Function**

$$
\begin{aligned}
v(\mu_\mathbf{x}, \Sigma_\mathbf{x})
&=
\mathbb{E}_\mathbf{f_*}
\left[ f_*^2 \, \mathbb{E}_\mathbf{x_*} \left[ p(f_*|\mathbf{x}_*) \right] \right] -
\left(\mathbb{E}_\mathbf{f_*}
\left[ f_* \, \mathbb{E}_\mathbf{x_*} \left[ p(f_*|\mathbf{x}_*) \right] \right]\right)^2\\
&=
\mathbb{E}_\mathbf{x_*}
\left[ \mathbb{E}_\mathbf{x_*} \left[ f_*^2 \, p(f_*|\mathbf{x}_*) \right] \right] -
\left(\mathbb{E}_\mathbf{x_*}
\left[ \mathbb{E}_\mathbf{x_*} \left[ f_* \, p(f_*|\mathbf{x}_*) \right] \right]\right)^2\\
&=
\mathbb{E}_\mathbf{x_*}
\left[  \sigma_\text{GP}^2(\mathbf{x}_*) + \mu_\text{GP}^2(\mathbf{x}_*) \right] -
\mathbb{E}_{x_*}\left[ \mu_\text{GP}(\mathbf{x_*}) \right]^2 \\
&=
\mathbb{E}_\mathbf{x_*}
\left[  \sigma_\text{GP}^2(\mathbf{x}_*) \right] + \mathbb{E}_\mathbf{x_*} \left[ \mu_\text{GP}^2(\mathbf{x}_*) \right] -
\mathbb{E}_{x_*}\left[ \mu_\text{GP}(\mathbf{x_*}) \right]^2\\
&=
\mathbb{E}_\mathbf{x_*} \left[  \sigma_\text{GP}^2(\mathbf{x}_*) \right] +
\mathbb{V}_\mathbf{x_*} \left[\mu_\text{GP}(\mathbf{x}_*) \right]
\end{aligned}
$$

---

### Taylor Approximation

We will approximate our mean and variance function via a Taylor Expansion. First the mean function:

$$
\begin{aligned}
\mathbf{z}_\mu =
\mu_\text{GP}(\mathbf{x_*})=
\mu_\text{GP}(\mu_\mathbf{x_*}) +
\nabla \mu_\text{GP}\bigg\vert_{\mathbf{x}_* = \mu_\mathbf{x}}
(\mathbf{x}_* - \mu_\mathbf{x_*})
+ \mathcal{O} (\mathbf{x_*}^2)
\end{aligned}
$$

and then the variance function:

$$
\begin{aligned}
\mathbf{z}_\sigma =
\nu^2_\text{GP}(\mathbf{x_*})=
\nu^2_\text{GP}(\mu_\mathbf{x_*}) +
\nabla \nu^2_\text{GP}\bigg\vert_{\mathbf{x}_* = \mu_\mathbf{x}}
(\mathbf{x}_* - \mu_\mathbf{x_*})
+ \mathcal{O} (\mathbf{x_*}^2)
\end{aligned}
$$

---

### Linearized Predictive Mean and Variance

$$\begin{aligned}
m(\mu_\mathbf{x_*}, \Sigma_\mathbf{x_*})
&=
\mu_\text{GP}(\mu_\mathbf{x_*})\\
v(\mu_\mathbf{x_*}, \Sigma_\mathbf{x_*})
&= \nu^2_\text{GP}(\mu_{x_*}) +
\nabla_\mathbf{x_*} \mu_\text{GP}(\mu_{x_*})^\top
\Sigma_{x_*}
\nabla_\mathbf{x_*} \mu_\text{GP}(\mu_{x_*}) +
\frac{1}{2} \text{Tr}\left\{ \frac{\partial^2 \nu^2(\mu_{x_*})}{\partial x_* \partial x_*^\top}  \Sigma_{x_*}\right\}
\end{aligned}
$$

#### Practical Equations


$$
\begin{aligned}
\mu_\text{GP}(\mathbf{x_*}) &= k(\mathbf{x_*}) \, \mathbf{K}_{GP}^{-1}y=k(\mathbf{x_*}) \, \alpha  \\
\nu_{GP}^2(\mathbf{x_*}) &= \sigma_y^2 + {\color{red}{\nabla_{\mu_\text{GP}}\,\Sigma_\mathbf{x_*} \,\nabla_{\mu_\text{GP}}^\top} }+ k_{**}- {\bf k}_* ({\bf K}+\sigma_y^2 \mathbf{I}_N )^{-1} {\bf k}_{*}^{\top}
\end{aligned}
$$

As seen above, the only extra term we need to include is the derivative of the mean function.


## Taylor Expansion


$$
\begin{aligned}
y &= f(x) + \epsilon_y \\
x &= \mu_x + \epsilon_x \\
\epsilon_y &\sim \mathcal{N} (0, \sigma_y^2) \\
\epsilon_x &\sim \mathcal{N} (0, \Sigma_x)
\end{aligned}
$$
This is the transformation of a Gaussian random variable $x$ through another r.v. $y$ where we have some additive noise $\epsilon_y$. The biggest difference is that the GP model assumes that $x$ is deterministic whereas we assume here that $x$ is a random variable itself. Because we know that integrating out the $x$'s is quite difficult to do in practice (because of the nonlinear Kernel functions), we can make an approximation of $f(\cdot)$ via the Taylor expansion. We can take the a 2nd order Taylor expansion of $f$ to be:
$$
\begin{aligned}
f(x) &\approx f(\mu_x + \epsilon_x) \\
     &\approx f(\mu_x) + \nabla f(\mu_x) \epsilon_x 
     + \sum_{i}\frac{1}{2} \epsilon_x^\top \nabla^2 f(\mu_x) \epsilon_x 
\end{aligned}
$$
where $\nabla_x$ is the gradient of the function $f(\mu_x)$ w.r.t. $x$ and $\nabla_x^2 f(\mu_x)$ is the second derivative (the Hessian) of the function $f(\mu_x)$ w.r.t. $x$. This is a second-order approximation which has that expensive Hessian term. There have have been studies that have shown that that term tends to be neglible in practice and a first-order approximation is typically enough. Now the question is: where to put use the Taylor expansion within the GP model? There are two options: the model or the posterior. We will outline the two approaches below.



## Approximate the Model



## Approximate The Posterior





We can compute the expectation $\mathbb{E}[\cdot]$ and variance $\mathbb{V}[\cdot]$ of this Taylor expansion to come up with an approximate mean and variance function for our posterior.

### Expectation

This calculation is straight-forward because we are taking the expected value of a mean function $f(\mu_x)$, the derivative of a mean function $f(\mu_x)$ and a Gaussian distribution noise term $\epsilon_x$ with mean 0. 
$$
\begin{aligned}
\mathbb{E}[f(x)] &\approx \mathbb{E}[f(\mu_x) + \nabla f(\mu_x) \epsilon_x] \\
								 &= f(\mu_x) + \nabla f(\mu_x) \mathbb{\epsilon_x} \\
								 &= f(\mu_x)
\end{aligned}
$$

### Variance

The variance term is a bit more complex.
$$
\begin{aligned}
\mathbb{E}\left[(f(x) - \mathbb{E}[f(x)])^\top(f(x) - \mathbb{E}[f(x)])\right] 
			&\approx \mathbb{E}\left[(f(x) - f(\mu_x))^\top(f(x) - f(\mu_x))\right] \\
			&\approx \mathbb{E} \left[ \left(f(\mu_x) + \nabla f(\mu_x)\epsilon_x \right) 
			\left( f(\mu_x) + \nabla f(\mu_x)\epsilon_x\right)^\top\right] \\
			&= \mathbb{E} \left[ \left(\nabla f(\mu_x)\: \epsilon_x  \right)^\top
			\left( \nabla f(\mu_x)\: \epsilon_x \right) \right] \\
			&= \nabla f(\mu_x) \mathbb{E}[\epsilon_x\epsilon_x^\top]\nabla f(\mu_x) \\
			&= \nabla f(\mu_x)\: \Sigma_x \:\nabla f(\mu_x)
\end{aligned}
$$

### I: Additive Noise Model ($x,f$)

This is the noise
$$
\begin{bmatrix}
    x \\
    y
    \end{bmatrix}
    \sim \mathcal{N} \left( 
    \begin{bmatrix}
    \mu_{x} \\ 
    \mu_{y} 
    \end{bmatrix}, 
    \begin{bmatrix}
    \Sigma_x & C \\
    C^\top & \Pi
    \end{bmatrix}
    \right)
$$
where
$$
\begin{aligned}
\mu_y &= f(\mu_x) \\
\Pi &= \nabla_x f(\mu_x) \: \Sigma_x \: \nabla_x f(\mu_x)^\top + \nu^2(x) \\
C &= \Sigma_x \: \nabla_x^\top f(\mu_x)
\end{aligned}
$$
So if we want to make predictions with our new model, we will have the final equation as:
$$
\begin{aligned}
f &\sim \mathcal{N}(f|\mu_{GP}, \nu^2_{GP}) \\
    \mu_{GP} &= K_{*} K_{GP}^{-1}y=K_{*} \alpha \\
    \nu^2_{GP} &= K_{**} - K_{*} K_{GP}^{-1}K_{*}^{\top} + \tilde{\Sigma}_x
\end{aligned}
$$
where $\tilde{\Sigma}_x = \nabla_x \mu_{GP} \Sigma_x \nabla \mu_{GP}^\top$.



##### Other GP Methods

We can extend this method to other GP algorithms including sparse GP models. The only thing that changes are the original $\mu_{GP}$ and $\nu^2_{GP}$ equations. In a sparse GP we have the following predictive functions
$$
\begin{aligned}
    \mu_{SGP} &= K_{*z}K_{zz}^{-1}m \\
    \nu^2_{SGP} &= K_{**} 
    - K_{*z}\left[ K_{zz}^{-1} - K_{zz}^{-1}SK_{zz}^{-1} \right]K_{*z}^{\top}
\end{aligned}
$$
So the new predictive functions will be:
$$
\begin{aligned}
    \mu_{SGP} &= K_{*z}K_{zz}^{-1}m \\
    \nu^2_{SGP} &= K_{**} 
    - K_{*z}\left[ K_{zz}^{-1} - K_{zz}^{-1}SK_{zz}^{-1} \right]K_{*z}^{\top} 
    + \tilde{\Sigma}_x
\end{aligned}
$$
As shown above, this is a fairly extensible method that offers a cheap improved predictive variance estimates on an already trained GP model. Some future work could be evaluating how other GP models, e.g. Sparse Spectrum GP, Multi-Output GPs, e.t.c.

### II: Non-Additive Noise Model



### III: Quadratic Approximation





## Parallels to the Kalman Filter

The Kalman Filter (KF) community use this exact formulation to motivate the Extended Kalman Filter (EKF) algorithm and some variants.




$$
\begin{bmatrix}
    x \\
    y
    \end{bmatrix}
    \sim \mathcal{N} \left( 
    \begin{bmatrix}
    \mu_{x} \\ 
    \mu_y 
    \end{bmatrix}, 
    \begin{bmatrix}
    \Sigma_x & C \\
    C^\top & \Pi
    \end{bmatrix}
    \right)
$$





## Connections


### KL-Divergence


## Code


## Literature

* Gaussian Process Priors with Uncertain Inputs: Multiple-Step-Ahead Prediction - Girard et. al. (2002) - Technical Report
  > Does the derivation for taking the expectation and variance for the  Taylor series expansion of the predictive mean and variance. 
* Expectation Propagation in Gaussian Process Dynamical Systems: Extended Version - Deisenroth & Mohamed (2012) - NeuRIPS
  > First time the moment matching **and** linearized version appears in the GP literature.
* Learning with Uncertainty-Gaussian Processes and Relevance Vector Machines - Candela (2004) - Thesis
  > Full law of iterated expectations and conditional variance.

---

## Supplementary



#### Error Propagation

#### Fubini's Theorem

#### Law of Iterated Expecations

#### Conditional Variance
