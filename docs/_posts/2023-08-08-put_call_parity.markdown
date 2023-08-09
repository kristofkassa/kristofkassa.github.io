---
layout: post
title:  "Put-Call Parity in the Binomial Model"
date:   2023-08-07 22:26:00 +0100
categories: [derivative_pricing]
---

### Put-Call Parity in the Binomial Model

Today, we'll navigate through the binomial model for option pricing, focusing particularly on the concept of put-call parity.

### Foundational Groundwork

To provide some context, options are financial instruments that derive their value from an underlying asset, such as stocks. They give holders the right, but not the obligation, to buy or sell the asset at a specified price within a certain timeframe. The binomial model is a widely used technique to price these options, leveraging a tree-like structure that maps out possible future stock prices.

In our last lesson, we began our journey by outlining the binomial model's basics. Today, we'll build upon that foundation, with an emphasis on the pivotal concept of put-call parity.

The Put-Call Parity concept is a fundamental principle in the world of options trading.

### Intuitive Explanation:

Imagine you are considering two different financial strategies:

1. **Strategy A**: You buy a call option (giving you the right to buy a stock at a certain price, called the strike price) and, at the same time, you invest an amount of money at the risk-free rate that will grow to be exactly the strike price when the option expires.

2. **Strategy B**: You buy the stock outright and also buy a put option (giving you the right to sell that stock at the strike price).

Now, under the principle of no-arbitrage (meaning there's no way to make a guaranteed profit without risk), both Strategy A and Strategy B should have the same cost today. Why? Because both strategies guarantee you the same thing: ownership of the stock at the expiration time of the options.

If they didn't cost the same, savvy traders could exploit the difference (or "arbitrage") to make a risk-free profit, which is not sustainable in an efficient market.

### The Math Behind It:

Using the strategies mentioned:

- Cost of **Strategy A** = Cost of the call option + Present value of the strike price (i.e., the amount you'd need to invest today at the risk-free rate to get to the strike price at expiration).
- Cost of **Strategy B** = Stock price today + Cost of the put option.

Put-Call Parity states that the cost of Strategy A = Cost of Strategy B. This is mathematically represented by:

$C_0 + Ke^{-rT} = S_0 + P_0$

Where:

$C_0$ = Price of the call option today.
$P_0$ = Price of the put option today.
$S_0$ = Current stock price.
$K$ = Strike price.
$r$ = Risk-free rate.
$T$ = Time to expiration.
$e$ = Base of the natural logarithm.

To conclude, Put-Call Parity ensures balance in the options market. If the relationship wasn't in balance, it would offer arbitrage opportunities which are essentially "free money" strategies - something markets tend to correct quickly.

### Setting the Technical Environment

Before any computational task, it's prudent to set up our environment. We rely on the versatile NumPy library:

{% highlight python %}
import numpy as np
{% endhighlight %}

The binomial model operates by creating a tree where each node represents a possible future stock price. In our previous lesson, we created a function to evaluate call options, which provide the right to buy an asset:

{% highlight python %}
def binomial_call_full(S_ini, K, T, r, u, d, N):
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # Risk neutral probs
    C = np.zeros([N + 1, N + 1])  # Call prices
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        C[N, i] = max(S_ini * (u ** (i)) * (d ** (N - i)) - K, 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return C[0, 0], C, S
{% endhighlight %}

Given our understanding of call options, it’s a logical progression to consider their counterparts - put options, which offer the right to sell an asset. The transition from call to put in our code primarily hinges on the manner we calculate payoffs.

The put option pricing looks as follows:

{% highlight python %}
def binomial_put_full(S_ini, K, T, r, u, d, N):
    dt = T / N  # Define time step
    p = (np.exp(r * dt) - d) / (u - d)  # Risk neutral probs
    P = np.zeros([N + 1, N + 1])  # Call prices
    S = np.zeros([N + 1, N + 1])  # Underlying price
    for i in range(0, N + 1):
        P[N, i] = max(K - (S_ini * (u ** (i)) * (d ** (N - i))), 0)
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            P[j, i] = np.exp(-r * dt) * (p * P[j + 1, i + 1] + (1 - p) * P[j + 1, i])
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
    return P[0, 0], P, S
{% endhighlight %}

In the realm of option pricing, it's the subtleties that matter. While call options consider the difference between the stock price and strike price (only if the former is higher), put options flip this relationship.

### Testing Our Implementation

In theoretical constructs, testing with real or hypothetical numbers adds immense value. By doing so, we ensure that our theoretical constructs translate effectively into actionable insights:

{% highlight python %}
put_price, P, S = binomial_put_full(100, 90, 1, 0, 1.2, 0.8, 2)
print("Price at t=0 for Put option is $", "{:.2f}".format(put_price))
call_price, C, S = binomial_call_full(100, 90, 1, 0, 1.2, 0.8, 2)
print("Price at t=0 for Call option is $", "{:.2f}".format(call_price))
{% endhighlight %}

`Price at t=0 for Put option is $ 6.50`
`Price at t=0 for Call option is $ 16.50`

For completeness, let's start verifying Put-Call parity at t=0:

{% highlight python %}
round(call_price + 90 * np.exp(-0 * 1), 2) == round(S[0, 0] + put_price, 2)
{% endhighlight %}

`True`

Now, let's check the same thing for some other node of the two-step tree. Remember that this is the evolution of underlying prices and option payoffs:

{% highlight python %}
print("Underlying Price Evolution:\n", S)
print("Call Option Payoff:\n", C)
print("Put Option Payoff:\n", P)
{% endhighlight %}

{% highlight python %}
Underlying Price Evolution:
 [[100.   0.   0.]
 [ 80. 120.   0.]
 [ 64.  96. 144.]]
Call Option Payoff:
 [[16.5  0.   0. ]
 [ 3.  30.   0. ]
 [ 0.   6.  54. ]]
Put Option Payoff:
 [[ 6.5  0.   0. ]
 [13.   0.   0. ]
 [26.   0.   0. ]]
{% endhighlight %}

Let's check put-call parity for the node following path 'd' (underlying price S_d = 80), which we can index as [1,0] in the matrix S:

{% highlight python %}
round(C[1, 0] + 90 * np.exp(-0 * 0.5), 2) == round(S[1, 0] + P[1, 0], 2)
{% endhighlight %}

`True`

Put-Call Parity ensures that no arbitrage opportunities exist in the market. Essentially, it describes a financial relationship between the price of a European call option and European put option, both with the same strike prices and expiration dates.