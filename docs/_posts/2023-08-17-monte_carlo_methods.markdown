---
layout: post
title: "Intro to Monte Carlo methods"
date: 2023-08-16
categories: derivative_pricing
---

### Introduction

In this post, we’ll delve into the exciting world of **Monte Carlo methods**, a powerful technique that plays a critical role in modern quantitative finance. We'll begin with a basic introduction, aiming to build a solid foundation that will be valuable when tackling more complex problems later in the series.

At its core, the Monte Carlo method is a statistical technique that allows us to approximate complex mathematical problems using random sampling. Here’s the basic idea:

- Simulate a large number of random samples from the same distribution as our underlying process.
- Calculate the option payoff for these simulated values and discount them.
- Take the average of these discounted payoffs. As the number of simulations increases, this average converges to the option’s true value.

Now let’s start by importing the numpy library:

{% highlight python %}
import numpy as np
{% endhighlight %}

We will develop a function, call_option_mc, to illustrate the power of Monte Carlo methods:

{% highlight python %}
def call_option_mc(S_ini, K, T, r, sigma, N, M):
    dt = T / N  # Define time step
    u = np.exp(sigma * np.sqrt(dt))  # Define u
    d = np.exp(-sigma * np.sqrt(dt))  # Define d
    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probs
    C = np.zeros([M])  # call prices
    S = np.zeros([M, N + 1])  # underlying price
    S[:, 0] = S_ini

    for j in range(0, M):
        random = np.random.binomial(
            1, p, N + 1
        )  # We sample random realizations for the paths of the tree under a binomial distribution
        for i in range(1, N + 1):
            if random[i] == 1:
                S[j, i] = S[j, i - 1] * u
            else:
                S[j, i] = S[j, i - 1] * d

        C[j] = np.exp(-r * T) * max(S[j, N] - K, 0)

    return S, C
{% endhighlight %}

### European Call Option Payoff (Monte Carlo)

In this example, we're focusing on a European call option. We have already defined a function call_option_mc that prices this option using Monte Carlo simulations.

Given the option characteristics:

`S=100`
`K=90`
`T=1`
`r=0%`
`σ=0.3`

Let's see how our Monte Carlo algorithm performs:

{% highlight python %}
S, C = call_option_mc(
    100, 90, 1, 0, 0.3, 2500, 15000
)  # Prices 15000 different simulations

print(np.mean(C))
16.99442814458002

{% endhighlight %}

As you can see, the value is pretty close to the one we got from the workout earlier on the full binomial model of `$17.01`.

### Asian Option Payoff (Monte Carlo)
Unlike European or American options, Asian options have a payoff that depends on the average price of the underlying asset over a certain period, not just the price at a specific point in time.

For our Asian option, we'll use the same parameters we used for the European option and will build a function asian_option_mc:

{% highlight python %}
def asian_option_mc(S_ini, K, T, r, sigma, N, M):
    dt = T / N  # Define time step
    u = np.exp(sigma * np.sqrt(dt))  # Define u
    d = np.exp(-sigma * np.sqrt(dt))  # Define d
    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probs
    Asian = np.zeros([M])  # Asian prices
    S = np.zeros([M, N + 1])  # underlying price
    S[:, 0] = S_ini

    for j in range(0, M):
        random = np.random.binomial(1, p, N + 1)
        Total = S_ini
        for i in range(1, N + 1):
            if random[i] == 1:
                S[j, i] = S[j, i - 1] * u
                Total = Total + S[j, i]
            else:
                S[j, i] = S[j, i - 1] * d
                Total = Total + S[j, i]

        Asian[j] = np.exp(-r * T) * max(Total / (N + 1) - K, 0)

    return S, Asian
{% endhighlight %}

{% highlight python %}
S, Asian = asian_option_mc(100, 90, 1, 0, 0.3, 2500, 10000)
{% endhighlight %}

And average them to get a final estimate:

{% highlight python %}
print(np.mean(Asian))
12.342288405943835
{% endhighlight %}

Let's also study the convergence of the methods in a similar fashion as before:

{% highlight python %}
M = np.arange(1000, 16000, 1000)
asian_price = []

for i in range(len(M)):
    S, Asian = asian_option_mc(100, 90, 1, 0, 0.3, 2500, M[i])
    asian_price.append(np.mean(Asian))
import matplotlib.pyplot as plt
​
plt.plot(M, asian_price)
plt.ylim([10, 15])
plt.title("MC Estimates for different # simulations")
plt.xlabel("Number of Simulations")
plt.ylabel("Asian Call Option Price")
plt.grid(True)
plt.show()

{% endhighlight %}

![MC Asian Model](/images/mc_estimates_asian.PNG)

### Conclusion
By now, you’ve witnessed firsthand how Monte Carlo methods can be a formidable tool when pricing complex derivatives. This technique is especially useful when calculating all the potential payoffs of an option is impractical due to computational constraints.

While this post introduces Monte Carlo from an informal perspective, we aimed to simplify your understanding of it. We plan to explore this topic in further depth in future posts.