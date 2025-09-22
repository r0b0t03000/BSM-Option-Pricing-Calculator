# BSM-Option-Pricing-Calculator

## Description:
This is a Python program designed to output a **theoretical** price and Greeks of a plain vanilla call/put European-style option using Black-Scholes pricing model. Black-Scholes model is a mathematical model developed by economists Fischer Black and Myron Scholes in 1973 used for pricing European options.

This program does not use a pre-built BSM library for calculating option prices or Greeks, but instead relies on a manual implementation of the formulas.

## Features
- Real-time stock data pulled from Yahoo Finance
- Manual implementation of Black-Scholes pricing formula
- Accommodates three volatility estimation methods: realized (historical) volatility, GARCH model, implied volatility
- Calculation of key Greeks: Delta, Gamma, Vega, Theta, Rho
- User input-driven CLI interface

## Formulas Used

<img width="289" height="44" alt="image" src="https://github.com/user-attachments/assets/a9367b54-7f2c-4664-9345-f55a78838d43" />


### Call Option:
<p><img src="image-3.png" alt="Call formula" width="400"/></p>

### Put Option:
<p><img src="image-2.png" alt="Put formula" width="400"/></p>

Where:
- <img src="image-1.png" width="120"/>
- <img src="image-4.png" width="320"/>
- <img src="image-5.png" width="420"/>
- **C** - discounted call option price
- **P** - discounted put option price
- **S** - latest stock's (underlying) close price
- **K** - option's strike price
- **q** - stock's dividend yield
- **r** - risk-free rate
- **T-t** - option's time to expiration measured in years
- **σ** - stock's volatility
- **N** – cumulative distribution function of the standard normal distribution

### Volatility Calculation:







