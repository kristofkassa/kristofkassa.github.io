---
layout: page
title: Derivative Pricing
permalink: /derivative-pricing/
---

Derivatives are financial contracts that derive their value from an underlying asset. These could be stocks, indices, commodities, currencies, exchange rates, or the rate of interest. These financial instruments help you make profits by betting on the future value of the underlying asset. They are complex financial instruments that are used for various purposes including hedging, access to inaccessible markets or commodities, and generating leverage.

## 1. [Binomial Model](/derivative_pricing/2023/08/07/binomial_model)

When it comes to pricing options, understanding the binomial tree model is crucial. In this article, we'll discuss how this model operates and demonstrate how to implement it using Python.

## 2. [Put-Call parity](/derivative_pricing/2023/08/08/put_call_parity)

Put-Call Parity ensures balance in the options market. If the relationship wasn't in balance, it would offer arbitrage opportunities which are essentially "free money" strategies - something markets tend to correct quickly.

## 3. [Introducing Delta](/derivative_pricing/2023/08/09/introducing_delta)

We'll introduce the concept of delta hedging, a strategy used to hedge against price movements in the underlying asset.

## 4. [Calibrating the Binomial Model](/derivative_pricing/2023/08/15/calibrating_binomial)

Calibrating the Binomial Option Pricing Model (BOPM) using the underlying asset's volatility.

## 5. [Dynamic Delta Hedging with American Options](/derivative_pricing/2023/08/16/american_options)

An American option gives its holder the right to exercise the option at any point in time, up until its expiration. This early exercise feature significantly impacts not only the option's price, but also the strategy we use to hedge the option.

## 6. [Monte Carlo Methods intro](/derivative_pricing/2023/08/16/monte_carlo_methods)

At its core, the Monte Carlo method is a statistical technique that allows us to approximate complex mathematical problems using random sampling.

## 7. [Markov's Property and Geometric Brownian Motion](/derivative_pricing/2023/08/17/markov_property_and_gbm)

Highlighting the principle that future states depend solely on the present state and not on the events that occurred before it, this post explains Markov's Property and its significant role in Geometric Brownian Motion.

## 8. [Ito's Lemma and Black Scholes](/derivative_pricing/2023/08/18/ito_lemma_black_scholes)

This post elucidates the connection between Ito's Lemma and the Black-Scholes model, revealing the underlying stochastic calculus that powers option pricing.

## 9. [Bridging the Black-Scholes with Monte Carlo Simulations](/derivative_pricing/2023/08/19/bs_and_mc)

By melding two powerful financial modeling techniques, this post illustrates how Monte Carlo simulations can enhance the predictive capabilities of the Black-Scholes model.

## 10. [Simulating Interest Rates: Vasicek Model](/derivative_pricing/2023/08/20/interest_rates_vasicek_model)

Delving into the intricacies of interest rate simulations, this post introduces the Vasicek Model's fundamentals and its utility in predicting interest rate movements.

## 11. [The Constraints of the Black-Scholes Model: A Data-Driven Analysis](/derivative_pricing/2023/08/21/bs_assumptions)

Exploring the inherent limitations of the Black-Scholes model, this post offers a data-centric critique of its assumptions and real-world applicability.

## 12. [Exploring the Volatility Smile with Yahoo Finance](/derivative_pricing/2023/08/22/implied_vol_and_vol_smile)

Yahoo Finance offers a treasure trove of data on exchange-traded options, both calls and puts, for an underlying stock. What is particularly interesting for us is that it provides the Black-Scholes implied volatility for varying options and strikes.

## 13. [Local Volatility Models: Dupire](/derivative_pricing/2023/09/28/local_vol_dupire)

Diving deep into the implied volatility surface—a 3D visualization that extends the concept of the volatility smile.

## 14. [Local Volatility Models: CEV (constant elasticity of variance) in Practice](/derivative_pricing/2023/09/28/local_vol_cev)

Constant Elasticity of Variance (CEV) model. Not only will we implement this local volatility model in Python, but we'll also calibrate it to real-world implied volatility data. 

## 15. [Stochastic Volatility: Heston](/derivative_pricing/2023/09/28/stochastic_vol_heston_mc_pricing)

We start dealing with stochastic volatility models. Specifically, we will implement the Monte-Carlo simulation of the Heston (1993) stochastic volatility model.

## 16. [Merton model](/derivative_pricing/2023/10/02/merton)

The Merton model is an extension of the Black-Scholes model, incorporating both stochastic volatility and jumps in asset prices. It provides a more realistic representation of asset price dynamics by capturing sudden, significant changes in asset values that occur in real markets.

## 17. [Fourier-Based Option Pricing](/derivative_pricing/2023/10/17/fourier_methods)

We will check the usefulness of Fourier transform methods for option pricing. We will do so in a setting that is already very familiar to us: the Black-Scholes model.

## 18. [Fourier Methods for Heston Model](/derivative_pricing/2023/10/17/fourier_for_heston)

We saw the performance of Fourier-based methods and Lewis's approach for option pricing under the Black-Scholes model. In this lesson, we will revisit these methods in the context of the Heston (1993) model. First, we will focus on pricing via the Heston model under these methods. Then, we will use them to calibrate the model to observed market prices.

## 19. [Merton Model Calibration](/derivative_pricing/2023/10/17/merton_model_calibration)

We will look at how to use previously learned methods like Lewis (2001) on Merton (1976) model in order to perform model calibration.

## 20. [Bates (1996) in practice](/derivative_pricing/2023/10/17/bates_in_practice)

We will look at how to use previously learned methods like Lewis (2001) and FFT on the Bates (1996) model. 

## 21. [Calibrating Bates (1996)](/derivative_pricing/2023/10/17/calibrating_bates)

We will use Lewis (2001) approach to fully calibrate Bates (1996) model based on real option market quotes for the EuroStoxx 50, as we have done before for other models. Different from the calibrations we have done so far, Bates (1996) requires a sequential procedure, whereby we first calibrate the stochastic volatility component of a Heston (1993) type of model. Then, we use those parameters to calibrate the jump component of Bates (1996). Finally, we calibrate, via local optimization, the general model.