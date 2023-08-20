---
layout: post
title: "Simulating Interest Rates: Vasicek Model"
date: 2023-08-19
categories: derivative_pricing
---

We will tackle the use of stochastic differential equations (SDE) for simulating the behavior of other types of assets. Specifically, we will focus on simulating interest rates using the SDE associated with the Vasicek (1977) model.

As always, let's first import some libraries we will need down the road:

```python
import matplotlib.pyplot as plt
import numpy as np
```

## 1. Vasicek (1977) Model
​
In Vasicek's model, the risk-neutral process for interest rates is:
​
$ dr = k(θ-r)dt + \sigma dz$
​
where $dz = \sqrt{dt} z$, with $z \sim \mathcal{N}(0,1)$. 
​
Let's build a function for this SDE:

```python
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