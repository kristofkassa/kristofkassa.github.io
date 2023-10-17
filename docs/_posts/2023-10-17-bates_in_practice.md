---
layout: post
title: "Bates (1996) in practice"
date: 2023-10-17
categories: derivative_pricing
---

We will look at how to use previously learned methods like Lewis (2001) and FFT on the Bates (1996) model. As was the case in the previous post, most of the code in this post is based on and adapted from Hilpisch.

To start with, let's import the necessary libraries:

```python
import numpy as np
from scipy.integrate import quad
```

## **1. Lewis (2001) Approach**
​
As we already know, the value of a call option under Lewis (2001) is determined by:
​
$$
\begin{equation*}
    C_0 = S_0 - \frac{\sqrt{S_0 K} e^{-rT}}{\pi} \int_{0}^{\infty} \mathbf{Re}[e^{izk} \varphi^{B96}(z-i/2)] \frac{dz}{z^2+1/4}
\end{equation*}
$$
where $\varphi^{B96}( )$ is the characteristic function of the model. In this case, the characteristic function of Bates (1996) was given by:
​
$$
\begin{equation*}
        \varphi^{B96}_0 (u, T) = \varphi^{H93}_0 \varphi^{M76J}_0 (u, T)
\end{equation*}
$$
​
which is essentially the product of two characteristic functions. Let's first define each of these characteristic functions:
​
​
### **1.1. Characteristic Functions** 
​
### **1.1.1. Heston (1993) Characteristic Function**
​
The characteristic function of Heston (1993), as we saw in Module 1, is given by:
​
$$
\
\begin{equation*}
  \varphi^{H93} (u, T) = e^{H_1(u, T)+H_2(u,T)\nu_0}
\end{equation*}
$$
\
where
$$
\
\begin{equation*}
  H_1 (u, T) \equiv r_0 uiT + \frac{c_1}{\sigma_\nu^2}\Biggl\{ (\kappa_\nu - \rho \sigma_\nu ui+c_2) T - 2 log \left[ \frac{1-c_3e^{c_2T}}{1-c_3} \right] \Biggl\}
\end{equation*}
$$
$$
\
\begin{equation*}
  H_2 (u, T) \equiv \frac{\kappa_\nu - \rho \sigma_\nu ui + c_2}{\sigma_\nu^2} \left[ \frac{1-e^{c_2T}}{1-c_3e^{c_2T}} \right]
\end{equation*}
$$
$$
\
\begin{equation*}
  c_1 \equiv \kappa_\nu \theta_\nu
\end{equation*}
$$
$$
\
\begin{equation*}
  c_2 \equiv - \sqrt{(\rho \sigma_\nu ui - \kappa_\nu)^2 - \sigma_\nu^2(-ui-u^2) }
\end{equation*}
$$
$$
\
\begin{equation*}
  c_3 \equiv \frac{\kappa_\nu - \rho \sigma_\nu ui + c_2}{\kappa_\nu - \rho \sigma_\nu ui - c_2}
\end{equation*}
$$
​
So let's first code this into a function to use later:
​
```python
def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach: characteristic function.
    Parameter definitions see function BCC_call_value."""
    c1 = kappa_v * theta_v
    c2 = -np.sqrt(
        (rho * sigma_v * u * 1j - kappa_v) ** 2 - sigma_v**2 * (-u * 1j - u**2)
    )
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) / (
        kappa_v - rho * sigma_v * u * 1j - c2
    )
    H1 = r * u * 1j * T + (c1 / sigma_v**2) * (
        (kappa_v - rho * sigma_v * u * 1j + c2) * T
        - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))
    )
    H2 = (
        (kappa_v - rho * sigma_v * u * 1j + c2)
        / sigma_v**2
        * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T)))
    )
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value
```

### **1.1.2. Merton (1976) *Adjusted* Characteristic Function (Only Jump Component)**
​
Remember that in order to produce a stochastic volatility jump-diffusion model, we need to incorporate **only** the jump component of the Merton (1976) characteristic function. 
​
The adjusted (only jump) characteristic function of Merton (1976) is therefore given by:
$$
\begin{equation*}
    \varphi^{M76J}_0 (u, T) = e^{\left( \left( i u \omega + \lambda ( e^{i u \mu_j - u^2 \delta^2/2}-1) \right) T \right)}
\end{equation*}
$$
​
where,
$$
\begin{equation*}
    \omega = - \lambda \left( e^{\mu_j + \delta^2/2}-1 \right)
\end{equation*}
$$

Let's then code in this characteristic function:

```python
def M76J_char_func(u, T, lamb, mu, delta):
    """
    Adjusted Characteristic function for Merton '76 model: Only jump component
    """
​
    omega = -lamb * (np.exp(mu + 0.5 * delta**2) - 1)
    char_func_value = np.exp(
        (1j * u * omega + lamb * (np.exp(1j * u * mu - u**2 * delta**2 * 0.5) - 1))
        * T
    )
    return char_func_value
```

### **1.1.3. Bates (1996) Characteristic Function**
​
Now we can combine both previous characteristic functions to produce the one we are interested in:

```python
def B96_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """
    Bates (1996) characteristic function
    """
    H93 = H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    M76J = M76J_char_func(u, T, lamb, mu, delta)
    return H93 * M76J
```

### **1.2. Call and Integral Value in Bates (1996)**
​
The next step is to calculate the value of the Lewis (2001) integral for the specific case of the Bates (1996) characteristic function:

```python
def B96_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """
    Lewis (2001) integral value for Bates (1996) characteristic function
    """
    char_func_value = B96_char_func(
        u - 1j * 0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
    )
    int_func_value = (
        1 / (u**2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    )
    return int_func_value
```

And, finally, we also need a function to compute the overall call option value once we have all the ingredients:

```python
def B96_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """
    Valuation of European call option in B96 Model via Lewis (2001)
    Parameters:
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial level of variance
    lamb: float
        jump intensity
    mu: float
        expected jump size
    delta: float
        standard deviation of jump
    ==========
    """
    int_value = quad(
        lambda u: B96_int_func(
            u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
        ),
        0,
        np.inf,
        limit=250,
    )[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value)
    return call_value
```

### **1.3. Pricing via Lewis (2001) with Bates (1996)**
​
Now, let's assign some values to the different parameters so we can perform some pricing exercises and compare them later on to the Carr and Madan (1999) approach.

```python
# General Parameters
S0 = 100
K = 100
T = 1
r = 0.05
​
# Heston'93 Parameters
kappa_v = 1.5
theta_v = 0.02
sigma_v = 0.15
rho = 0.1
v0 = 0.01
​
# Merton'76 Parameters
lamb = 0.25
mu = -0.2
delta = 0.1
sigma = np.sqrt(v0)
```

Let's see what is the price of the call option then:

```python
print(
    "B96 Call option price via Lewis(2001): $%10.4f"
    % B96_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
)
B96 Call option price via Lewis(2001): $    8.9047
```
​
## **2. FFT Approach - Carr and Madan (1999)**
​
As an alternative to Lewis (2001), we could also implement the FFT algorithm. Essentially, we can apply FFT to the integral in the call option price derived by Carr and Madan (1999):
$$
\begin{equation*}
    C_0 = \frac{e^{-\alpha \kappa}}{\pi} \int_{0}^{\infty} e^{-i\nu \kappa} \frac{e^{-rT} \varphi^{B96} (\nu - (\alpha + 1)i, T)}{\alpha^2 + \alpha - \nu^2 + i(2\alpha + 1)\nu} d\nu
\end{equation*}
$$

Here we are going to use the same numerical routine we implemented in Module 1; please go there for more information. As was the case with the Lewis (2001) approach, we basically have to adapt the characteristic function we are considering to be the Bates (1996) one.

```python
def B96_call_FFT(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """
    Call option price in Bates (1996) under FFT
    """
​
    k = np.log(K / S0)
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
            B96_char_func(v, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
            / (alpha**2 + alpha - vo**2 + 1j * (2 * alpha + 1) * vo)
        )
​
    else:
        alpha = 1.1
        v = (vo - 1j * alpha) - 1j
        modcharFunc1 = np.exp(-r * T) * (
            1 / (1 + 1j * (vo - 1j * alpha))
            - np.exp(r * T) / (1j * (vo - 1j * alpha))
            - B96_char_func(
                v, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
            )
            / ((vo - 1j * alpha) ** 2 - 1j * (vo - 1j * alpha))
        )
​
        v = (vo + 1j * alpha) - 1j
​
        modcharFunc2 = np.exp(-r * T) * (
            1 / (1 + 1j * (vo + 1j * alpha))
            - np.exp(r * T) / (1j * (vo + 1j * alpha))
            - B96_char_func(
                v, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
            )
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
        payoff = (np.fft.fft(FFTFunc)).real
        CallValueM = np.exp(-alpha * k) / np.pi * payoff
    else:
        FFTFunc = (
            np.exp(1j * b * vo) * (modcharFunc1 - modcharFunc2) * 0.5 * eta * SimpsonW
        )
        payoff = (np.fft.fft(FFTFunc)).real
        CallValueM = payoff / (np.sinh(alpha * k) * np.pi)
​
    pos = int((k + b) / eps)
    CallValue = CallValueM[pos] * S0
​
    return CallValue
```

Now, let's see how the FFT algorithm performs in pricing an option with the same given parameters as the Lewis (2001) example:

```python
print(
    "B96 Call option price via FFT: $%10.4f"
    % B96_call_FFT(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta)
)
B96 Call option price via FFT: $    8.9047
```

As you can see, we get exactly the same number for the call option price, with up to 4 decimal places. If you increase the number of decimal places to be shown, though, you will see that the prices are slightly different. This is what we should expect from two different methodologies.

## **3. Conclusion**
​
In this post, we have applied both Fourier-based techniques learned in Module 1--Lewis (2001) and Carr and Madan (1999) FFT procedure--to the stochastic volatility jump-diffusion model of Bates (1996). As you can see, once we are familiar with Fourier pricing methods, it is just a matter of adapting the code to the characteristic function of the underlying asset process.
​
We still have not touched upon the most important issue, though: model calibration. That's what we will do in the next lesson.
​
**References**
​
- Hilpisch, Yves. *Derivatives Analytics with Python: Data Analysis, Models, Simulation, Calibration and Hedging.* John Wiley & Sons, 2015.
​
- Bates, David S. "Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options." *The Review of Financial Studies*, vol. 9, no. 1, 1996, pp. 69-107.
