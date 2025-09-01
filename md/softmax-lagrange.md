---
title: 'Softmax and Lagrange Multipliers'
---
My understanding from a Machine Learning perspective is that Softmax happens to be _a_ function that can satisfy the requirements of a multi-class probability distribution function,

1. `f([x1, x2]) -> [y1, y2, ...]` where y1 and y2 are positive numbers.
2. Values in `[y1, y2, ...]` sum to 1.

I didn't realize you could analytically derive the Softmax function using Entropy and Lagrange Multipliers. This helps me understand how a Softmax at the end of a DNN converts the numerical output of a DNN into a probability distribution.

## Background

Given a probability distribution for a set of outcomes (classes), the Shannon Entropy for this distribution quantifies how uncertain *a* distribution is if we were to observe them. It just so happens that `log(1/p)` is the only function that satisfies the constraints of the problem Shannon was trying to solve [1], and thus the formula for Shanon Entropy is,

$$ H(p) = - \Sigma_i \ p_i * log(p_i) $$

Why do we care to maximize the entropy of the learned probability distribution? [2] Let's say we have the following discrete probability function `[0.25, 0.25, 0.25, 0.25]`. That is, there are four classes and each of them are equally likely. Here, `H = 1.3`, we have maximum entropy because we make no assumptions about the likelihood of any class. A function `[0.8, 0.1, 0.0009, 0.0001]`, with `H = 0.4`, has some assumptions baked in because we expect the first class to show up 80% of the time.

This setup is called the Maximum Entropy Principle[2] and we will use it in the derivation below.

## Derivation

### Main Optimization Problem

Given $z_i$ as the raw output of a DNN, and $p_i = F(z_i)$, Find $F$ that *maximizes* the given function,

$$ H(p) = - \Sigma_i \ p_i * log(p_i) $$
subject to,
$$ \Sigma_i \ p_i = 1 $$
$$ \Sigma_i \ p_i * z_i = \mu $$

The first constraint is to ensure the probability values sum to 1. 

The second constraint is a bit harder to parse from an ML pov. Derived from a Physics pov [3], this constraint is simply a prior on what the probability distribution must look like. If I'm working on this problem for the first time it's not clear why I should pick this constraint specifically, until you solve a slighly simpler problem which we will do first below.

### Simpler Optimization Problem

Given $z_i$ as the raw output of a DNN, and $p_i = F(z_i)$, Find $F$ that *maximizes* the given function,

$$ H(p) = - \Sigma_i \ p_i * log(p_i) $$
subject to,
$$ \Sigma_i \ p_i = 1 $$

The Lagrange optimization problem is,

$$ L = - \Sigma_i\  p_i log(p_i) - \lambda_1 (\Sigma_i \ p_i -1) $$

Taking the derivative and setting to 0 you get,
$$
∂L/∂p_i = -log(p_i) - 1 - λ₁ = 0
$$ and solving for $p_i$ you get,
$$
\begin{matrix}
-log(p_i) = 1 + λ_1 \\ 
log(p_i) = -1 - λ_1 \\ 
=> p_i = e^{(-1 - λ_1)} \\
\end{matrix}
$$
and applying the second constraint we get,
$$
\begin{align}
\Sigma_i p_i = 1 \\ 
\Sigma_i e^{(-1 - λ_1)} = 1 \\
n * e^{(-1 - λ_1)} = 1 \\
=> p_i = 1/n \\
\end{align}
$$

What this simpler optimization problem is telling us is that we need some additional constraint on the output probability to force the distribution values to be something other than the uniform probability distribution.

The idea with the $\mu$ constraint is that we want the probability distribution to have *some* expected value. In the context of applied ML this just means we find the best fit during training with the assumption that there is *some* expected value and whatever the empirical value we can compute after training is done is the expected value for that training run on the dataset. 

### Solving the Original Problem

Given the constraints we have the following Lagrange optimization problem,
$$
 L = - \Sigma_i\  p_i log(p_i) - \lambda_1 (\Sigma_i \ p_i -1) - \lambda_2(\Sigma_i p_i * z_i - \mu)
$$

Taking the derivative and setting to 0, we get,
$$
\begin{align}
-log(p_i) = 1 + λ₁ + λ₂ * z_i \\
log(p_i) = -1 - λ₁ - λ₂ * z_i \\
p_i = exp(-1 - λ₁ - λ₂ * z_i) \\
=> p_i = exp(-1 - λ₁) * exp(-λ₂ * z_i)
\end{align}
$$

Using the constraint 1 we get,
$$
\begin{align}
\Sigma_i \ e^{(-1 - λ₁)} * e^{(-λ₂ * z_i)} = 1 \\
=> e^{(-1 - λ₁)} = 1 / Σ e^{(-λ₂ * z_i)}
\end{align}
$$

and finally substituting back,
$$
\begin{align}
p_i = exp(-λ₂ * z_i) / Σ exp(-λ₂ * z_j)
\end{align}
$$

and if we set $\lambda_2$ to -1,
$$
\begin{align}
=> p_i = exp(z_i) / Σ exp(z_j)
\end{align}
$$


deriving the softmax formula.

(Note: if you set $\lambda_2$ to $-\infty$ as a limit, we get the _hard max_ function which is $max(1, 2, 4) = 4$)


## References

[1] https://en.wikipedia.org/wiki/A_Mathematical_Theory_of_Communication

[2] https://en.wikipedia.org/wiki/Principle_of_maximum_entropy

[3] https://en.wikipedia.org/wiki/Boltzmann_distribution
