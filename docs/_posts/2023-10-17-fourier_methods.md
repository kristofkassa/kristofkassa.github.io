---
layout: post
title: "Fourier-Based Option Pricing"
date: 2023-10-17
categories: derivative_pricing
---

We will check the usefulness of Fourier transform methods for option pricing. We will do so in a setting that is already very familiar to us: the Black-Scholes model. Most of the code in this post is based on and adapted from Hilpisch.

To start, let's import the necessary libraries:

```python
import numpy as np
from numpy.fft import fft
from scipy import stats
from scipy.integrate import quad
```

We are going to check the different European call option prices we obtain under the different methods. This will allow us to compare not only the final outcome but also the speed of the algorithms, which will be key in a real-life setting.

For this pricing exercise, let's think of a European call option with the following parameters.

```python
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
```

## **1. Analytical Solution to BS Model**
​
We will start with the already known analytical solution to the BS model:

```python
def BS_analytic_call(S0, K, T, r, sigma):
    """This function will provide the closed-form solution
    to the European Call Option price based on BS formula
    """
​
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
​
    bs_call = S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * stats.norm.cdf(
        d2, 0.0, 1.0
    )
​
    return bs_call
print(
    " BS Analytical Call Option price will be $",
    BS_analytic_call(S0, K, T, r, sigma).round(4),
)
 BS Analytical Call Option price will be $ 10.4506
```

## **2. Fourier Transform as in Lewis**
​
Next, we will deal with Lewis's approach to obtaining a semi-analytical solution to the price of the option via Fourier methods. As you know already, this method requires a bunch of things to know ex-ante:
​
### **2.1 Black-Scholes Characteristic Function**
​
Fourier pricing methods require that we know the characteristic function of the process $S_t$ (or some form of it like $s_t = log S_t$). In the case of a BS model (without dividends), the characteristic function is given by:
​
$$
\begin{equation*}
  \varphi^{BS} (u, T) = e^{((r-\frac{\sigma^2}{2})iu - \frac{\sigma^2}{2}u^2)T}
\end{equation*}
$$
​
Let's implement a function for it:

```python
def BS_characteristic_func(v, x0, T, r, sigma):
    """Computes general Black-Scholes model characteristic function
    to be used in Fourier pricing methods like Lewis (2001) and Carr-Madan (1999)
    """
​
    cf_value = np.exp(
        ((x0 / T + r - 0.5 * sigma**2) * 1j * v - 0.5 * sigma**2 * v**2) * T
    )
​
    return cf_value
```

### **2.2 Integral Value in Lewis**
​
We also need to get a value for the integral in Lewis:
​
$$
\begin{equation*}
    C_0 = S_0 - \frac{\sqrt{S_0 K} e^{-rT}}{\pi} \int_{0}^{\infty} \mathbf{Re}[e^{izk} \varphi(z-i/2)] \frac{dz}{z^2+1/4}
\end{equation*}
$$
​
We will next compute a function for what's inside the integral:

```python
def BS_integral(u, S0, K, T, r, sigma):
    """Expression for the integral in Lewis (2001)"""
​
    cf_value = BS_characteristic_func(u - 1j * 0.5, 0.0, T, r, sigma)
​
    int_value = (
        1 / (u**2 + 0.25) * (np.exp(1j * u * (np.log(S0 / K))) * cf_value).real
    )
​
    return int_value
```

### **2.3 Calculating the Value of the Integral and Putting It All Together**
​
Now we can put everything together. Note, that we will need to numerically compute the value of the aforementioned integral. There are different methods to do that. For now, we will use the quadrature method (*quad*) included in the scipy package:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
​
```python
def BS_call_Lewis(S0, K, T, r, sigma):
    """European Call option price in BS under Lewis (2001)"""
​
    int_value = quad(lambda u: BS_integral(u, S0, K, T, r, sigma), 0, 100)[0]
​
    call_value = max(0, S0 - np.exp(-r * T) * (np.sqrt(S0 * K)) / np.pi * int_value)
​
    return call_value
print(
    " Fourier Call Option price under Lewis (2001) is $",
    BS_call_Lewis(S0, K, T, r, sigma).round(4),
)
Fourier Call Option price under Lewis (2001) is $ 10.4506
```

## **3. Fast Fourier Transform (FFT) for Option Pricing**
​
Next, we are going to see how Fast Fourier Transform (FFT) could also be implemented in this context. To perform FFT, we will use the 'fft' function included in the numpy library:
https://numpy.org/doc/stable/reference/routines.fft.html
​
```python
def BS_call_FFT(S0, K, T, r, sigma):
    """European Call option price in BS under FFT"""
​
    k = np.log(K / S0)
    x0 = np.log(S0 / S0)
    g = 1  # Factor to increase accuracy
    N = g * 4096
    eps = (g * 150) ** -1
    eta = 2 * np.pi / (N * eps)
    b = 0.5 * N * eps - k
    u = np.arange(1, N + 1, 1)
    vo = eta * (u - 1)
​
    # Modifications to ensure integrability
    if S0 >= 0.95 * K:  # ITM Case
        alpha = 1.5
        v = vo - (alpha + 1) * 1j
        modcharFunc = np.exp(-r * T) * (
            BS_characteristic_func(v, x0, T, r, sigma)
            / (alpha**2 + alpha - vo**2 + 1j * (2 * alpha + 1) * vo)
        )
​
    else:
        alpha = 1.1
        v = (vo - 1j * alpha) - 1j
        modcharFunc1 = np.exp(-r * T) * (
            1 / (1 + 1j * (vo - 1j * alpha))
            - np.exp(r * T) / (1j * (vo - 1j * alpha))
            - BS_characteristic_func(v, x0, T, r, sigma)
            / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha))
        )
​
        v = (vo + 1j * alpha) - 1j
​
        modcharFunc2 = np.exp(-r * T) * (
            1 / (1 + 1j * (vo + 1j * alpha))
            - np.exp(r * T) / (1j * (vo + 1j * alpha))
            - BS_characteristic_func(v, x0, T, r, sigma)
            / ((vo + 1j * alpha) ** 2 - 1j * (vo + 1j * alpha))
        )
​
    # Numerical FFT Routine
    delt = np.zeros(N)
    delt[0] = 1
    j = np.arange(1, N + 1, 1)
    SimpsonW = (3 + (-1) ** j - delt) / 3
    if S0 >= 0.95 * K:
        FFTFunc = np.exp(1j * b * vo) * modcharFunc * eta * SimpsonW
        payoff = (fft(FFTFunc)).real
        CallValueM = np.exp(-alpha * k) / np.pi * payoff
    else:
        FFTFunc = (
            np.exp(1j * b * vo) * (modcharFunc1 - modcharFunc2) * 0.5 * eta * SimpsonW
        )
        payoff = (fft(FFTFunc)).real
        CallValueM = payoff / (np.sinh(alpha * k) * np.pi)
​
    pos = int((k + b) / eps)
    CallValue = CallValueM[pos] * S0
​
    return CallValue
print(
    " Fourier Call Option price via FFT is $", BS_call_FFT(S0, K, T, r, sigma).round(4)
)
 Fourier Call Option price via FFT is $ 10.4506
```

## **4. Execution Time**
​
As you have seen, the three methods yield the same European option price. There are, however, differences in the execution time related to each. Let's explore this:

```python
import datetime
# BS Closed-form
begin = datetime.datetime.now()
price = BS_analytic_call(S0, K, T, r, sigma)
print(
    "BS closed-from price is $",
    price.round(4),
    ".Code took ",
    datetime.datetime.now() - begin,
)
BS closed-from price is $ 10.4506 .Code took  0:00:00.000549
```

Now, check for yourself how much option pricing under both Fourier methods (Lewis and FFT) performs better in terms of time. Obviously, you may realize in advance that a closed-form solution for BS is going to be the faster way. Unfortunately, as we already know, there are not always straightforward closed-form solutions.