# Input Uncertainty for Gaussian Processes

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Documentation: [jejjohnson.github.io/uncertain_gps](https://jejjohnson.github.io/uncertain_gps)
* Repo: [github.com/jejjohnson/uncertain_gps](https://github.com/jejjohnson/uncertain_gps)

---

<p>
<center>
<img src="docs/pics/gp_plot.png" width="400"/>

<b>Caption</b>: A graphical model of a GP algorithm with the addition of uncertainty component for the input.
</center>
</p>

This repository is home to my studies on uncertain inputs for Gaussian processes. Gaussian processes are a kernel Bayesian framework that is known to generalize well for small datasets and also offers predictive mean and predictive variance estimates. It is one of the most complete models that model uncertainty.

<p align="center">
<img src="docs/pics/gp_demo.png" />

<b>Caption</b>: Demo showing the error bars for a standard GP predictive variance and an augmented predictive variance estimate using Taylor expansions.
</center>
</p>


In this repository, I am interested in exploring the capabilities and limits with Gaussian process regression algorithms when handling noisy inputs. Input uncertainty is often not talked about in the machine learning literature, so I will be exploring this in great detail for my thesis.

---

### Methods


#### Taylor Approximation

<p align="center">
<center>
<img src="docs/pics/gp_taylor_funcs.png" width="500"/>

<b>Caption</b>: The predictive mean and variance equations with the correction term in red to augment the predictive variance to account for uncertain inputs with a known variance.
</center>
</p>

We can approximate the predictive mean and variance equations via a Taylor expansion which adds a corrective term w.r.t. the derivative of the function and the known variance. This assumes we know the variance and we don't modify the predictive mean of the learned GP function.

---

#### Exact Moment Matching

**TODO**

---

#### Variational Inference

**TODO**

---

#### Monte Carlo Estimation

**TODO**

---

## My Resources

#### [Literature Review](https://jejjohnson.github.io/uncertain_gps/#/literature)

I have gone through most of the relevant works related to noisy inputs in the context of Gaussian processes. It is very extensive and it also offers some literature that is relevant but may not explicitly mentioned uncertain inputs in the paper.

#### [Documentation](https://jejjohnson.github.io/uncertain_gps/)

I have some documentation which has all of my personal notes and derivations related to GPs and noisy inputs. Some highlights include the Taylor approximation, moment matching and variational inference.

#### [GP Model Zoo](https://jejjohnson.github.io/gp_model_zoo/#/literature)

I have documented and try to keep up with some of the latest Gaussian process literature in my repository.

