# Error Propagation


### Law of Error Propagation

Another explanation of the rational of the Taylor series for the NIGP stems from the \textbf{error propagation law}. Let's take some function $f(\mathbf x)$ where $x \sim \mathcal{P}$ described by a mean $\mu_\mathbf{x}$ and covariance $\Sigma_\mathbf{x}$. The Taylor series expansion around the function $f(\mathbf x)$ is:

$$\mathbf z = f(\mathbf x) \approx f(\mu_{\mathbf x}) +   \frac{\partial f(\mu_{\mathbf x})}{\partial \mathbf x} \left(  \mathbf x - \mu_{\mathbf x} \right) $$

This results in a mean and error covariance of the new distribution $\mathbf z$ defined by:

$$\mu_{\mathbf z} = f(\mu_{\mathbf x})$$
$$\Sigma_\mathbf{z} = \nabla_\mathbf{x} f(\mu_{\mathbf x}) \cdot \Sigma_\mathbf{x} \cdot\nabla_\mathbf{x} f(\mu_{\mathbf x})^{\top}$$

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




\begin{itemize}
    \item \href{http://srl.informatik.uni-freiburg.de/papers/arrasTR98.pdf}{ Introduction to Error Propagation: Derivation, Meaning and Examples}
    \item \href{http://irtfweb.ifa.hawaii.edu/~cushing/downloads/mcc_errorprop.pdf#page5}{Short Summary}
    \item \href{https://users.aalto.fi/~mvermeer/uncertainty.pdf}{Statistical uncertainty and error propagation}
\end{itemize}