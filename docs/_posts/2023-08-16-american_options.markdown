---
layout: post
title: "American Options and Dynamic Delta Hedging"
date: 2023-08-16
categories: derivative_pricing
---

### Introduction
An **American option** gives its holder the right to exercise the option **at any point in time, up until its expiration**. This contrasts with **European options**, which **can only be exercised at expiration**. This early exercise feature significantly impacts not only the option's price, but also the strategy we use to hedge the option.

In the context of European options, we can delta-hedge our portfolio by looking at terminal prices. This, however, is no longer an option for American derivatives, which require dynamic delta hedging for every step in the tree. In this post, we will take a closer look at how to perform dynamic hedging for American options using a binomial tree methodology.

### Delta Hedging in the Binomial Tree: American Options

Let’s start by importing the necessary libraries:


{% highlight python %}
import numpy as np
{% endhighlight %}

The following Python function allows us to calculate the price of an American option.

{% highlight python %}
def american_option(S_ini, K, T, r, u, d, N, opttype):
    dt = T / N
    p = (np.exp(r * dt) - d) / (u - d)
    C = np.zeros([N + 1, N + 1])
    S = np.zeros([N + 1, N + 1])
    Delta = np.zeros([N, N])
    
    for i in range(0, N + 1):
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
        if opttype == "C":
            C[N, i] = max(S[N, i] - K, 0)
        else:
            C[N, i] = max(K - S[N, i], 0)

    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
            if opttype == "C":
                C[j, i] = max(C[j, i], S[j, i] - K)
            else:
                C[j, i] = max(C[j, i], K - S[j, i])
            Delta[j, i] = (C[j + 1, i + 1] - C[j + 1, i]) / (S[j + 1, i + 1] - S[j + 1, i])
    
    return C[0, 0], C, S, Delta
{% endhighlight %}

Let's compute the price and delta for an example American call option:

{% highlight python %}
price, C, S, delta = american_option(45, 100, 5, 0, 1.5, 1 / 1.5, 5, "C")
price = 6.459200000000002

delta = 
array([[0.34208, 0., 0., 0., 0.],
       [0.1328, 0.4816, 0., 0., 0.],
       [0., 0.22133333, 0.65511111, 0., 0.],
       [0., 0., 0.36888889, 0.84592593, 0.],
       [0., 0., 0., 0.61481481, 1.]])

{% endhighlight %}

To understand the effects of early exercise, let's compare this with the delta hedging of a European option with the same characteristics. Let’s look at the delta matrix for a European call option:


{% highlight python %}
def european_option(S_ini, K, T, r, u, d, N, opttype):
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probs
    C = np.zeros([N + 1, N + 1])  # call prices
    S = np.zeros([N + 1, N + 1])  # underlying price
    Delta = np.zeros([N, N])  # delta

    for i in range(0, N + 1):
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
        if opttype == "C":
            C[N, i] = max(S[N, i] - K, 0)
        else:
            C[N, i] = max(K - S[N, i], 0)

    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            C[j, i] = np.exp(-r * dt) * (
                p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i]
            )  # Computing the European option prices
            S[j, i] = (
                S_ini * (u ** (i)) * (d ** (j - i))
            )  # Underlying evolution for each node

            Delta[j, i] = (C[j + 1, i + 1] - C[j + 1, i]) / (
                S[j + 1, i + 1] - S[j + 1, i]
            )  # Computing the delta for each node

    return C[0, 0], C, S, Delta
{% endhighlight %}  

{% highlight python %}
price_euro, C_euro, S_euro, delta_euro = european_option(50, 52, 5, 0.05, 1.2, 0.8, 5, "C")

delta_euro = 
array([[0.72579669, 0., 0., 0., 0.],
       [0.44672292, 0.83592034, 0., 0., 0.],
       [0.09194177, 0.58672101, 0.93425542, 0., 0.],
       [0., 0.12822237, 0.76764645, 1., 0.],
       [0., 0., 0.17881944, 1., 1.]])

price_euro = 13.37283461304853
{% endhighlight %}


### Conclusion

In this lesson, we have seen the importance of dynamic delta hedging in the context of American derivatives. 
The ability to exercise an option early adds complexity to the problem of hedging. This concept is important not only for American options but also for other path-dependent options, which we will explore in future lessons. 

The key takeaway is that the delta of an American option can vary significantly from the delta of a European option, underscoring the necessity for a different hedging strategy.