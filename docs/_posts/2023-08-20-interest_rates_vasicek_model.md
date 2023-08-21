---
layout: post
title: "Simulating Interest Rates: Vasicek Model"
date: 2023-08-20
categories: derivative_pricing
---

We will tackle the use of stochastic differential equations (SDE) for simulating the behavior of other types of assets. Specifically, we will focus on simulating interest rates using the SDE associated with the Vasicek (1977) model.

As always, let's first import some libraries we will need down the road:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as pdr
```

## 1. Vasicek (1977) Model
​
The Vasicek interest rate model can be described as:
​
\begin{equation}
dr = k(\theta - r) \, dt + \sigma \, dz
\end{equation}

- r is the short interest rate.
- k is the speed of reversion to the mean. It dictates how rapidly the interest rate r.
- r reverts to its long-term mean value θ.
- θ is the long-term mean level of the interest rate. The rate r.
- r will tend to revert to this value over time.
- σ is the volatility of the interest rate, defining the standard deviation of changes in the interest rate.
- dz is the change in a Wiener process or standard Brownian motion.

These parameters would typically be estimated or calibrated using market data when applying the model.

Let's build a function for this SDE:

```python

# Vasicek (1977) Model
# Vasicek's risk-neutral process for interest rates is given by:
# dr = k(θ-r)dt + σdz, where dz = √dt * z, with z ~ N(0,1)

def vasicek(r0, K, theta, sigma, T, N, M):
    dt = T / N
    rates = np.zeros((N, M))
    rates[0, :] = r0
    for j in range(M):
        for i in range(1, N):
            dr = (
                K * (theta - rates[i - 1, j]) * dt
                + sigma * np.sqrt(dt) * np.random.normal()
            )
            rates[i, j] = rates[i - 1, j] + dr
    return rates
```

Now, we can use the same intuition of the Monte Carlo methods we used before for the Black-Scholes SDE to simulate interest rates given a current level, $r_0$. 

```python
# Downloading historical 3-month Treasury bill rates from FRED as a representation of short-term interest rates
data = pdr.data.DataReader('TB3MS', 'fred', start='2000-01-01')  # TB3MS is the FRED code for 3-month T-bill secondary market rate
latest_rate = data.iloc[-1, 0] / 100  # Convert the rate from percentage to decimal
r0 = latest_rate
```

We are fetching historical 3-month Treasury bill rates as a representative of short-term interest rates. The most recent value from this data is then used as in the Vasicek model simulation. Adjust the start parameter in the DataReader function if you wish to fetch data from a different starting date.

```python
M = 100  # Number of paths for MC
N = 100  # Number of steps
T = 2.0  # Maturity
K = 0.40
theta = 0.01
sigma = 0.012
t = np.linspace(0, T, N)

rates = vasicek(r0, K, theta, sigma, T, N, M)
rates.shape
```

We have simulated interest rates for 100 paths and 100 steps.

Let's see how these look:

```python
plt.figure(figsize=(12, 8))
for j in range(M):
    plt.plot(t, rates[:, j])

plt.xlabel("Time $t$", fontsize=14)
plt.ylabel("$r(t)$", fontsize=14)
plt.title("Vasicek Paths", fontsize=14)
axes = plt.gca()
axes.set_xlim([0, T])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
```

![vasicek_paths](/images/vasicek_paths.png)

You have seen even more of the power of Monte Carlo techniques. As long as we have an expression (e.g., an SDE) that models the behavior of the underlying asset, we can apply this technique to all different kinds of assets, not just equities!