---
layout: post
title: "Fourier Methods for Heston Model"
date: 2023-10-17
categories: derivative_pricing
---

We saw the performance of Fourier-based methods and Lewis's approach for option pricing under the Black-Scholes model. In this lesson, we will revisit these methods in the context of the Heston (1993) model. First, we will focus on pricing via the Heston model under these methods. Then, we will use them to calibrate the model to observed market prices.

## **1. Fourier-Based Pricing for Heston (1993) Model**
​
First, we are going to see how the Fourier-based approaches perform for Heston (1993) with some pre-defined model parameters. This will require that we define several things, most importantly Heston's characteristic function. 
​
To start with, let's import the necessary libraries for now:

```python
import numpy as np
from scipy.integrate import quad
```

We will go over the process for a standard European call option. Then, you can adapt the code for other options. We will specifically go over the process for following the Lewis (2001) approach. Hopefully, after this pricing process, you will be able to implement FFT by yourselves.
​
Now, there are a few things we need before going over the pure pricing process. Let's go over these while defining an appropriate function to be used later on.
​
### **1.1. Heston (1993) Characteristic Function**
​
Probably the most important ingredient for Fourier transform methods such as Lewis (2001) is **knowledge of the characteristic function for the underlying process**. Deriving the characteristic function of Heston (1993) is not as easy and straightforward as in the case of Black-Scholes. We will present here the closed-form expression; you can check the original Heston (1993) paper or Gatheral (2006) to see the derivation of this characteristic function.
​
The characteristic function of the Heston (1993) model is given by:
​
$$
\
\begin{equation*}
  \varphi^{H} (u, T) = e^{H_1(u, T)+H_2(u,T)\nu_0}
\end{equation*}
$$
\
where
​
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
\
As you can see, the derivation and closed-form expression for the characteristic function of the Heston model is not simple at all. Luckily for us, we can create a function in Python that simplifies its calculations every time:

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

Now that we have our characteristic function, let's move on to another important step in the pricing process.
​
### **1.2 Integral Value in Lewis (2001)**
​
We also need to get a value for the integral in Lewis (2001):
​
$$
\begin{equation*}
    C_0 = S_0 - \frac{\sqrt{S_0 K} e^{-rT}}{\pi} \int_{0}^{\infty} \mathbf{Re}[e^{izk} \varphi(z-i/2)] \frac{dz}{z^2+1/4}
\end{equation*}
$$


Obviously, the expression for the integral is the same one we used for Black-Scholes, but note that the expression for the characteristic function has changed.

```python
def H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """
    Fourier-based approach for Lewis (2001): Integration function.
    """
    char_func_value = H93_char_func(
        u - 1j * 0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0
    )
    int_func_value = (
        1 / (u**2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    )
    return int_func_value
```

### **1.3 Calculating the Value of the Integral and Call Value**
​
Finally, we will need to numerically compute the value of the aforementioned integral. As before, we will use the quadrature method (*quad*) included in the scipy package (https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
​
```python
def H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """Valuation of European call option in H93 model via Lewis (2001)
​
    Parameter definition:
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
    Returns
    =======
    call_value: float
        present value of European call option
    """
    int_value = quad(
        lambda u: H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0),
        0,
        np.inf,
        limit=250,
    )[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value)
    return call_value
```

## **2. Pricing with Heston (1993) via Lewis (2001)**
​
Now that we have all the necessary functions, let's price! 
​
We will do so for a standard European call option with the following parameters:

```python
# Option Parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.02
```

Also, for the purpose of checking whether everything works, we will assume the following parameters for the Heston model. Remember that to obtain these parameters we will have to calibrate the model to market prices. We will do that by the end of this module.

```python
# Heston(1993) Parameters
kappa_v = 1.5
theta_v = 0.02
sigma_v = 0.15
rho = 0.1
v0 = 0.01
```

Now, if we implement the whole pricing process described above, we can get to a Call option price with the mentioned characteristics and model parameters:

```python
print(
    "Heston (1993) Call Option Value:   $%10.4f "
    % H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0)
)
Heston (1993) Call Option Value:   $    5.7578 
```

So far, you have learned how to perform pricing with the Heston (1993) stochastic volatility model under the approach by Lewis (2001). One important advantage of Fourier-based methods is that they require very little information (basically, the characteristic function of the process followed by the underlying) to arrive at a semi-analytical solution for the price of the option. 
​
As usual, most problems related with these methods arise in the market calibration process, the most important tool we have to extract the values for the different model parameters. Now, we will go over the full calibration process of the Heston model with real market data.
​
​
## **3. Heston Model Calibration**
​
At this point, we are going to guide you through the full process of model calibration for the Heston model. We will do this calibration by looking at market option prices. Hence, the first thing we need is options' market data to work with. 
​
Unlike in other occasions, where we directly downloaded data from Yahoo finance, due to the higher complexity in the process involved here, we will work with data in a local file. Specifically, we are going to calibrate our Heston model using market data for options on the EuroStoxx 50 index (Europe's 50 largest firms). We will take the data for just one day, September 30th, 2014. 
​
Let's start by importing some additional libraries needed:
​
```python
import pandas as pd
from scipy.optimize import brute, fmin
```

### **3.1. Gather Options' Market Data**
​
Now, in order to load the mentioned option market data, you need to load the file provided and place it in the same directory we are working on:

```python
# Market Data from www.eurexchange.com
# as of September 30, 2014
​
h5 = pd.HDFStore(
    "option_dataset.h5", "r"
)  # Place this file in the same directory before running the code
data = h5["data"]  # European call & put option data (3 maturities)
h5.close()
S0 = 3225.93  # EURO STOXX 50 level September 30, 2014
```

Once you have the market data loaded, we are going to select the options that we want to be part of the calibration process. We will select near ATM options:

```python
# Option Selection
​
tol = 0.02  # Tolerance level to select ATM options (percent around ITM/OTM options)
options = data[(np.abs(data["Strike"] - S0) / S0) < tol]
options["Date"] = pd.DatetimeIndex(options["Date"])
options["Maturity"] = pd.DatetimeIndex(options["Maturity"])
```

Then, we add time left until maturity and a constant risk-free rate:

```python
# Adding Time-to-Maturity and constant short-rates
​
for row, option in options.iterrows():
    T = (option["Maturity"] - option["Date"]).days / 365.0
    options.loc[row, "T"] = T
    options.loc[row, "r"] = 0.02

data.head()
Date	Strike	Call	Maturity	Put
0	1412035200000000000	1850.0	1373.6	1418947200000000000	0.5
1	1412035200000000000	1900.0	1323.7	1418947200000000000	0.6
2	1412035200000000000	1950.0	1273.8	1418947200000000000	0.8
3	1412035200000000000	2000.0	1223.9	1418947200000000000	0.9
4	1412035200000000000	2050.0	1174.1	1418947200000000000	1.1
```

## **3.2. Calibration Process**
​
Now that we have the data, let's begin our calibration process. Apart from the previously defined functions (or, better said, building on those), we will need to define some additional functions to optimize our model parameters so that they match observed market data.
​
First, we will introduce a function that will evaluate the error the model makes with respect to observed data given certain parameters. As usual, we will rely on a **mean squared error (MSE) function**. We will also define some initial values for the calibration parameters:

```python
i = 0
min_MSE = 500
def H93_error_function(p0):
    """Error function for parameter calibration via
    Lewis (2001) Fourier approach for Heston (1993).
    Parameters
    ==========
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial, instantaneous variance
    Returns
    =======
    MSE: float
        mean squared error
    """
    global i, min_MSE
    kappa_v, theta_v, sigma_v, rho, v0 = p0
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or rho < -1.0 or rho > 1.0:
        return 500.0
    if 2 * kappa_v * theta_v < sigma_v**2:
        return 500.0
    se = []
    for row, option in options.iterrows():
        model_value = H93_call_value(
            S0,
            option["Strike"],
            option["T"],
            option["r"],
            kappa_v,
            theta_v,
            sigma_v,
            rho,
            v0,
        )
        se.append((model_value - option["Call"]) ** 2)
    MSE = sum(se) / len(se)
    min_MSE = min(min_MSE, MSE)
    if i % 25 == 0:
        print("%4d |" % i, np.array(p0), "| %7.3f | %7.3f" % (MSE, min_MSE))
    i += 1
    return MSE
```

Next, we will need a function that performs the optimization process. In other words, it optimizes the model parameters so as to minimize the error function with respect to market data. We will do this in 2 steps in order to look for faster convergence of the prices to market quotes. First, we will use the brute function of scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html), that allows the calibration to focus on most sensible ranges. Once these are declared, we can dig deeper into the specific regions and get the actual parameters more accurately with the fmin function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html).

```python
def H93_calibration_full():
    """Calibrates Heston (1993) stochastic volatility model to market quotes."""
    # First run with brute force
    # (scan sensible regions, for faster convergence)
    p0 = brute(
        H93_error_function,
        (
            (2.5, 10.6, 5.0),  # kappa_v
            (0.01, 0.041, 0.01),  # theta_v
            (0.05, 0.251, 0.1),  # sigma_v
            (-0.75, 0.01, 0.25),  # rho
            (0.01, 0.031, 0.01),
        ),  # v0
        finish=None,
    )
​
    # Second run with local, convex minimization
    # (we dig deeper where promising results)
    opt = fmin(
        H93_error_function, p0, xtol=0.000001, ftol=0.000001, maxiter=750, maxfun=900
    )
    return opt
```

### 3.3. Results from Calibration
​
Now that we have all the necessary ingredients, let's see how our calibration algorithm performs. For that, given the way we structured things before, we just need to call our *H93_calibration_full()* function. This will give us each of the different outputs from calibration, including the values given to the different parameters in the model. Before running, please be aware of the time this algorithm will take!

```python
H93_calibration_full()
   0 | [ 2.5   0.01  0.05 -0.75  0.01] | 820.892 | 500.000
  25 | [ 2.5   0.02  0.05 -0.75  0.02] |  23.864 |  21.568
  50 | [ 2.5   0.02  0.25 -0.75  0.03] |  89.655 |  21.568
  75 | [ 2.5   0.03  0.15 -0.5   0.01] | 193.283 |  21.568
 100 | [ 2.5   0.04  0.05 -0.5   0.02] | 176.340 |  21.568
 125 | [ 2.5   0.04  0.25 -0.5   0.03] | 486.965 |  21.568
 150 | [ 7.5   0.01  0.15 -0.25  0.01] | 840.337 |  21.568
 175 | [ 7.5   0.02  0.05 -0.25  0.02] |  24.810 |  21.568
 200 | [ 7.5   0.02  0.25 -0.25  0.03] |  24.834 |  21.568
 225 | [7.5  0.03 0.15 0.   0.01] | 110.936 |  21.568
 250 | [7.5  0.04 0.05 0.   0.02] | 540.183 |  21.568
 275 | [7.5  0.04 0.25 0.   0.03] | 783.222 |  21.568
 300 | [ 2.61379559  0.00992657  0.15610448 -0.76361614  0.02778356] |   8.046 |   7.795
 325 | [ 1.9152359   0.01257942  0.16036675 -0.91693167  0.0248233 ] |   6.301 |   6.146
 350 | [ 2.04831069  0.01215428  0.15832201 -0.89057611  0.02532865] |   6.151 |   6.145
 375 | [ 2.0376908   0.01207312  0.16816435 -0.86785979  0.02553311] |   6.097 |   6.075
 400 | [ 1.97316132  0.01247835  0.2032535  -0.83478704  0.0254568 ] |   6.002 |   5.996
 425 | [ 2.07617861  0.01268556  0.21375606 -0.83213139  0.02575716] |   5.948 |   5.948
 450 | [ 2.66904762  0.0147503   0.22891593 -0.87575667  0.02607257] |   5.684 |   5.684
 475 | [ 3.14901508  0.01554998  0.22701309 -0.86349685  0.02615741] |   5.358 |   5.358
 500 | [ 3.75301757  0.0179153   0.33579328 -0.72953664  0.02611118] |   4.603 |   4.499
 525 | [ 5.15416061  0.01857864  0.40457683 -0.45664742  0.0270172 ] |   3.894 |   3.825
 550 | [ 5.14078039  0.01861684  0.42859861 -0.43432012  0.02728158] |   3.769 |   3.749
 575 | [ 5.0507416   0.01868091  0.43385652 -0.44775932  0.02732562] |   3.719 |   3.717
 600 | [ 5.05235394  0.01871275  0.43473462 -0.44693948  0.02730235] |   3.716 |   3.716
 625 | [ 5.05371685  0.01872767  0.43504931 -0.44691718  0.0272971 ] |   3.716 |   3.715
 650 | [ 5.04446488  0.01872907  0.43469053 -0.44848558  0.02728228] |   3.715 |   3.715
 675 | [ 5.04382644  0.0187266   0.43463479 -0.44869357  0.02728557] |   3.715 |   3.715
 700 | [ 5.04426287  0.01872709  0.4346593  -0.44857764  0.02728496] |   3.715 |   3.715
 725 | [ 5.04476043  0.01872427  0.43464808 -0.44841756  0.02728689] |   3.715 |   3.715
 750 | [ 5.04505376  0.01872602  0.43468094 -0.44837446  0.02728584] |   3.715 |   3.715
 775 | [ 5.04707738  0.01872566  0.434764   -0.44801571  0.02728898] |   3.715 |   3.715
 800 | [ 5.04731712  0.01872561  0.43477373 -0.44796137  0.02728913] |   3.715 |   3.715
 825 | [ 5.04735223  0.01872569  0.43477611 -0.44795766  0.02728916] |   3.715 |   3.715
 850 | [ 5.04735429  0.0187257   0.43477637 -0.44795746  0.02728915] |   3.715 |   3.715
 875 | [ 5.04735911  0.01872571  0.43477669 -0.44795607  0.02728914] |   3.715 |   3.715
 900 | [ 5.04735681  0.01872573  0.43477687 -0.44795639  0.02728911] |   3.715 |   3.715
 925 | [ 5.0473585   0.01872573  0.43477688 -0.44795608  0.02728912] |   3.715 |   3.715
Optimization terminated successfully.
         Current function value: 3.715367
         Iterations: 488
         Function evaluations: 814
array([ 5.0473602 ,  0.01872573,  0.43477697, -0.44795577,  0.02728912])

```

Now we have finally calibrated our parameters to market values.
​
The results from this calibration give us the following values for the parameters in the Heston (1993) model:
​
$\kappa_\nu = 5.047$ 
​
$\theta_\nu = 0.018$ 
​
$\sigma_\nu = 0.434$
​
$\rho = -0.447$
​
$\nu_0 = 0.027$
​
The next step will be simply using these parameters to price the option we want.

## **4. Conclusion**

In this lesson, we have, first, used Fourier methods to price options using the Lewis (2001) approach for the Heston (1993) model, and second, developed a full calibration of the Heston (1993) model. If you understood the full process covered in this notebook, you are on the right track to face the next module, where we will introduce a model that combines stochastic volatility with jump diffusion features.

References

- Gatheral, Jim. The Volatility Surface: A Practitioner's Guide. John Wiley & Sons Inc., 2006.

- Heston, Steven L. "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." The Review of Financial Studies, vol. 6, no. 2, 1993, pp. 327-343.

- Hilpisch, Yves. Derivatives Analytics with Python: Data Analysis, Models,Simulation, Calibration and Hedging. John Wiley & Sons, 2015.