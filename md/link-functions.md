---
title: Logistic Regression == Link(Linear Regression) ??
---

## Linear Regression

For Linear Regression, given a Dependent variable `y` that is a Real number, we want to use the Independent variable `x`, also a Real number, to predict `y`. The formulation for Linear Regression is,

$$ y = w*x + b $$

Here, `w` and `b` are both real numbers as well. On a 2 dimensional graph this equation will form a line and thus the name "linear". The linear regression model makes sense to many real world problems because we know that there is "linearity" in the problem. If the problem is not linear we can still try and fit a linear regression model on the data, it just won't work well when we try to use the model in practice.

If we have a list of independent variables, we can rewrite this line to a multi-dimensional object using a system of equations, using matrices,

$$ Y = w^T*X + B $$

Here, `w` is the learnable parameter. We estimate the value of `w` for a given dataset `D` of size `n` typically with Ordinary Least Squares which is a closed form solution and is the Maximum Likelihood estimate [1].  The key assumptions here are that both `Y` and `X` are continuous (Real numbers). Essentially Linear Regressions helps us model a function that maps real numbers to real numbers (ie. $R \to R$). What if we want to model binary or multi-label outputs?

## Step Function

One way to do this is to use a step function. The extension is fairly simple in that we take the Linear Regression model and apply a 'sign' [2] function to it,
$$ Y = sign\ \ (\ w^T*X + B\ \ ) $$
$$ sign(x) = \begin{cases} 1,\ x \ge 0 \\ 0,\ x < 0 \end{cases} $$
The sign function here captures the idea that if we output a positive value then we match that to one of the _two_ classes. For a 0 or negative value we assign it to the opposite class.

The problem with this model is that the formulation might make sense but it not work in practice. Linear regression works because it assume linearity, but this is tied to the dependent variable. Some problems happen to be linearly separable but most aren't,

![[nonseparble-data.png]]

Non-linear regression can help but using the sign function limits what we can do. We can however take the idea of using an intermediate transformation on the regression and with the right choice of transformation solve this problem. The idea of "link" functions are the way to solve this, using logits.

## Binary Logistic Regression

Let's first look at learning a model that can predict a binary value (ie 0 or 1). We know that we want a function that can map $R \to [0,1]$. The standard normal distribution function is a good candidate and this idea is called the Probit model [3], and around the same time the idea of using log odds showed up [4].

For probability calculations `odds ratio` is an easy measure to compute and communicate the chances of an event occurring against it not occurring. For example, `the odds of it raining today are 3 to 1`, which is to be read as `it is 3 times more likely to rain than not`.

Why would you use this? Let's say out of the past 4 days it rained 3 times versus 1. Here `p(rain) = 3/4` and `p(~rain) = 1/4`. From a lay person's perspective, it's much harder to communicate that the probability of rain is `0.75` or `3/4`. It is simpler to just use `3:1 or, 3 to 1`. The probability `0.75` is just `3/(3+1)` calculated using the odds value, but much harder to visualize. `3 to 1 odds in favour of raining` clearly communicates that it rains 3 times more than it does not. Note, gambling uses the term `odds` a little differently but it's similar in idea.

So given this context, the odds ratio helps use find the function we care about. It is defined in terms of the probability of an event as,

$$ odds = p / (1-p) $$

The odds ratio maps probability values to $(0, +\infty)$. If we use the `log` function on the odds we get a function that maps probability values to $(-\infty, +\infty)$. Log just happens to be the right function to extend the probability mapping. Logits is a portmanteau of "log" and "digits". It just means "the log of numbers", which comes from taking the logs of the odds ratio.

To tie this all together, let's call the linear regression on the inputs `f(x)` and the dependent variable `y` which represents the binary random variable.

For linear regression we are trying to learn params that will help us model the dependent variable `y'` using the regression using the following form,

$$ y' = f(x) \ \ \ \ \ \ \ \ where \ f(x) = w*x + b$$

For a binary variable, `y`, we first apply the log odds link function to y and then model the log odds using the regression. We have a model of the binary dependent variable `y` in terms of `x` just like we tried to with the `sign` function above except we formulate it using an intermediate log odds step with a rearrangement of terms,

$$
\begin{aligned}
log(\frac{y}{1-y}) = f(x) \\ \\
=> \frac{y}{1-y} = e ^ {f(x)} \\
=> y = \frac{e^{f(x)}}{(1 + e^{f(x)})} \\
=> y = \frac{1}{(1 + e^{-f(x)})}
\end{aligned}
$$

Note, multi-label is not the same as multi-class. They sound similar but are different. Multi-class means the dependent class can only be 1 of N classes. It cannot be two classes at the same time. This might seem confusing because many real world objects are typically a combination of multiple labels. To be clear, think of multi-class as predicting "an orange or an apple" for a given image of a fruit. This means that it _has_ to be one or the other and the probability value is modeled using log odds to reflect this. The reason we have $log \ (p / (1-p))$ is to capture this relationship. That is "it is either X or it is not X".

Multi-label on the other hand specifically means the model outputs are independent of each other and think of the classes as being "true" at the same time. Instead of asking "is this an orange or an apple" in the case of multi-class, we are asking "is this an apple?, is this an orange?" in the multi-label setup. For multi-label regression, the formulation is different and you can think of the setup as building a regression classifier for each of the different labels (think of it as `N` binary logistic regressions).

So, how do we extend Binary Logistic Regression to Multinomial (or Multi-class Regression)?

## Multi-class Logistic Regression

For multi-class logistic regression we have to look back at what the logit formulation is actually doing. When we start with $log(y/(1-y)) = f(x)$ we are essentially saying the "boundary line" between the two classes is `f(x)`. In the discussion above `f(x)` is linear for discussion's sake but it can be a non-linear function as well which would help with in cases like the circle dataset above. After rearranging the terms we get a "logistic" separation boundary as we saw above.

So, if we think of the logits as representing the boundary between two sets of data points, additional labels are simply pair-wise logits as well with a caveat. The idea is that if we have `K` classes, we pick one of them (say the last class for convenience), and build logits between that class and every other class. This works because we hold one class out and build multiple "logistic" separation boundaries against it. Assuming `K=3` and using class `3` as boundary, we  have,

$$
\begin{aligned}
log \frac{P(y = i)}{P(y = K)} = w_i * x \\
=> P_{y=1} = P_{y=3} * e^{w_1 * x}\ and \\
P_{y=2} = P_{y=3} * e^{w_2 * x}
\end{aligned}
$$

and by rearranging the terms,

$$
\begin{aligned}
P_{y=1} + P_{y=2} + P_{y=3} = 1 \\
=> P_{y=3} = \frac{1}{1+e^{w_1 * x}+e^{w_2 * x}} \\
\end{aligned}
$$

Note here that class `3` doesn't have a boundary with itself, like the other classes do with the logits formulation. This means that since it doesn't exist we can set $w_3 = 0$  and thus $e^{w_3*x} = 1$. So now we can rewrite $P_{y=2}$ and the other probabilities to,

$$
\begin{aligned}
P_{y=1} = \frac{e^{w_1 * x}}{e^{w_3 * x} + e^{w_1 * x}+e^{w_2 * x}} \\
\\
P_{y=2} = \frac{e^{w_2 * x}}{e^{w_3 * x}+e^{w_1 * x}+e^{w_2 * x}} \\
\\
P_{y=3} = \frac{e^{w_3 * x}}{e^{w_3 * x}+e^{w_1 * x}+e^{w_2 * x}} \\
\\
\\
\equiv P_{y=k} = \frac{e^{w_k * x}}{\Sigma^K_i e^{w_i * x}}
\end{aligned}
$$


Which is the `softmax` function as defined in ML literature[5]

## References

[1] <https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote08.html>

[2] <https://en.wikipedia.org/wiki/Heaviside_step_function>

[3] The origins and development of the logit model - J.S. Cramer, University of Amsterdam and Tinbergen Institute, Amsterdam

[4] <https://en.wikipedia.org/wiki/Probit_model>

[5] Training Stochastic Model Recognition Algorithms as Networks can lead to Maximum Mutual Information Estimation of Parameters - John S. Bridle
