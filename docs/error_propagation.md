# Error Propagation


## Taylor Series Expansion

> A Taylor series is representation of a function as an infinite sum of terms that are calculated from the values of the functions derivatives at a single point - Wiki

Often times we come across functions that are very difficult to compute analytically. Below we have the simple first-order Taylor series approximation.

Let's take some function $f(\mathbf x)$ where $\mathbf{x} \sim \mathcal{N}(\mu_\mathbf{x}, \Sigma_\mathbf{x})$ described by a mean $\mu_\mathbf{x}$ and covariance $\Sigma_\mathbf{x}$. The Taylor series expansion around the function $f(\mathbf x)$ is:

$$\mathbf z = f(\mathbf x) \approx f(\mu_{\mathbf x}) +   \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\left(  \mathbf x - \mu_{\mathbf x} \right) $$

### Law of Error Propagation


This results in a mean and error covariance of the new distribution $\mathbf z$ defined by:

$$\mu_{\mathbf z} = f(\mu_{\mathbf x})$$
$$\Sigma_\mathbf{z} = \nabla_\mathbf{x} f(\mu_{\mathbf x}) \; \Sigma_\mathbf{x} \; \nabla_\mathbf{x} f(\mu_{\mathbf x})^{\top}$$


<details>
<summary><font color="red">Proof:</font> Mean Function</summary>

Given the mean function:

$$\mathbb{E}[\mathbf{x}] = \frac{1}{N} \sum_{i=1} x_i$$

We can simply apply this to the first-order Taylor series function.

$$
\begin{aligned}
\mu_\mathbf{z} &= 
\mathbb{E}_{\mathbf{x}} \left[  f(\mu_{\mathbf x}) +   \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\left(  \mathbf x - \mu_{\mathbf x} \right) \right] \\
&= \mathbb{E}_{\mathbf{x}} \left[  f(\mu_{\mathbf x}) \right] +   \mathbb{E}_{\mathbf{x}} \left[  \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\left(  \mathbf x - \mu_{\mathbf x} \right) \right] \\
&= f(\mu_{\mathbf x}) + 
\mathbb{E}_{\mathbf{x}} \left[  \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}  \mathbf x  \right]- \mathbb{E}_{\mathbf{x}} \left[ \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\mu_{\mathbf x} \right] \\
&= f(\mu_{\mathbf x}) +
 \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}  \mu_\mathbf{x} -  \frac{\partial f}{\partial \mathbf x} \bigg\vert_{\mathbf{x} = \mu_\mathbf{x}}\mu_{\mathbf x}  \\
&= f(\mu_{\mathbf x}) \\
\end{aligned}
$$

</details>

<details>
<summary><font color="red">Proof:</font> Variance Function</summary>

Given the variance function 

$$\mathbb{V}[\mathbf{x}] = \mathbb{E}\left[ \mathbf{x} - \mu_\mathbf{x} \right]^2$$

$$
\begin{aligned}
\sigma_\mathbf{z}^2
&=
\mathbb{E} \left[ f(\mu_\mathbf{x}) - \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}} (\mathbf{x} - \mu_\mathbf{x}) - \mu_\mathbf{x} \right] \\
&=
\mathbb{E} \left[ \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}}  (\mathbf{x} - \mu_\mathbf{x})\right]^2 \\
&=
\left( \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}} \right)^2 \mathbb{E}\left[  \mathbf{x} - \mu_\mathbf{x}\right]^2\\
&= \left( \frac{\partial f}{\partial \mathbf{x}} \bigg\vert_{\mathbf{x}=\mu_\mathbf{x}} \right)^2 \Sigma_\mathbf{x}
\end{aligned}
$$

</details>



I've linked a nice tutorial for propagating variances below if you would like to go through the derivations yourself. We can relate the above formula to the logic of the NIGP by thinking in terms of the derivatives (slopes) and the input error. We can actually calculate how much the slope contributes to the noise in the error in our inputs because the derivative of a GP is still a GP. Like above, assume that our noise $\epsilon_x$ comes from a normal distribution with variance $\Sigma_x$, $\epsilon_x \sim \mathcal{N}(0, \Sigma_x)$. We also assume that the slope of our function is given by $\frac{\partial f}{\partial x}$. At every infinitesimal point we have a tangent line to the slope, so multiplying the derivative by the error will give us an estimate of how much our variance estimate should change, $\epsilon_x\frac{\partial f}{\partial x}$. We've assumed a constant slope so we will have a mean of 0, 

$$\mathbb{E}\left[ \epsilon_x\frac{\partial f}{\partial x} \right]=m(\mathbf x)=0$$ 

Now we just need to calculate the variance which is given by:


$$\mathbb{E}\left[ \left( \frac{\partial f}{\partial x} \epsilon_x\right)^2\right] = \mathbb{E}\left[ \left( \frac{\partial f}{\partial x} \right)\epsilon_x \epsilon_x^{\top}\left( \frac{\partial f}{\partial x} \right)^{\top} \right] = \frac{\partial f}{\partial x}\Sigma_x \left( \frac{\partial f}{\partial x}\right)^{\top}$$

So we can replace the $\epsilon_y^2$ with a new estimate for the output noise:

$$\epsilon_y^2 \approx \epsilon_y^2 + \frac{\partial f}{\partial x}\Sigma_x \left( \frac{\partial f}{\partial x}\right)^{\top}$$

And we can add this to our formulation:

$$\begin{aligned}
y &= f(\mathbf x) + \frac{\partial f(\mathbf x)}{\partial x}\Sigma_x \left( \frac{\partial f(\mathbf x)}{\partial x}\right)^{\top} + \epsilon_y \\
\end{aligned}$$


---

#### Resources

* Essence of Calculus, Chapter 11 | Taylor Series - 3Blue1Brown - [youtube](https://youtu.be/3d6DsjIBzJ4)
* Introduction to Error Propagation: Derivation, Meaning and Examples - [PDF](http://srl.informatik.uni-freiburg.de/papers/arrasTR98.pdf)
* Statistical uncertainty and error propagation - Vermeer - [PDF](https://users.aalto.fi/~mvermeer/uncertainty.pdf)

\begin{itemize}
    \item \href{http://srl.informatik.uni-freiburg.de/papers/arrasTR98.pdf}{ Introduction to Error Propagation: Derivation, Meaning and Examples}
    \item \href{http://irtfweb.ifa.hawaii.edu/~cushing/downloads/mcc_errorprop.pdf#page5}{Short Summary}
    \item \href{https://users.aalto.fi/~mvermeer/uncertainty.pdf}{Statistical uncertainty and error propagation}
\end{itemize}