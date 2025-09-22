# BSM-Option-Pricing-Calculator

## Video-demo: 

## Description:
This is a Python program designed to output a **theoretical** price and Greeks of a plain vanilla call/put European-style option using Black-Scholes pricing model. Black-Scholes model is a mathematical model developed by economists Fischer Black and Myron Scholes in 1973 used for pricing European options.

This program does not use a pre-built BSM library for calculating option prices or Greeks, but instead relies on a manual implementation of the formulas.

## Features
- Real-time stock data pulled from Yahoo Finance.
- Manual implementation of Black-Scholes pricing formula.
- Accommodates three volatility estimation methods: realized (historical) volatility, GARCH model, implied volatility.
- Calculation of key Greeks: Delta, Gamma, Vega, Theta, Rho.
- User input-driven CLI interface.

## Formulas Used

### Call Option:
<img width="289" height="44" alt="image" src="https://github.com/user-attachments/assets/a9367b54-7f2c-4664-9345-f55a78838d43" />

### Put Option:
<img width="334" height="39" alt="image" src="https://github.com/user-attachments/assets/262c7acb-b56a-474c-b141-19fc43667f4e" />


Where:
- <img width="319" height="77" alt="image" src="https://github.com/user-attachments/assets/ed3d9429-f7a9-4643-8086-3b9e0928d728" />
- <img width="428" height="80" alt="image" src="https://github.com/user-attachments/assets/2be74ea7-dda9-4df5-a9ab-41b62960f9c7" />
- **c** - discounted call option price
- **p** - discounted put option price
- **S<sub>0</sub>** - latest stock's (underlying) close price
- **K** - option's strike price
- **q** - stock's dividend yield
- **r** - risk-free rate
- **T-t** - option's time to expiration measured in years
- **σ** - stock's volatility
- **N** – cumulative distribution function of the standard normal distribution

### Volatility Calculation:
- **Realized volatility (RV) method**:  
Underlying's volatility is calculated based on the fixed lookback period of 3 months or 63 days (under assumption of 21 trading days in the
given month). First, underlying's log returns are calculated and stored. Second, standard deviation of log returns are calculated and
stored. Finally, volatility is obtained by annualising standard deviation using the factor of square root of 252 (under assumption of 252
trading days in a year).

- **GARCH method**:  
The program estimates volatility using a GARCH(1,1) model, a common econometric approach for forecasting financial asset volatility.  
The model assumes that conditional variance depends on three components:
  - the long-run average variance of the asset,
  - the most recent squared return (shock term),
  - the most recent conditional variance (persistence term).  
The GARCH model captures volatility clustering and provides a forward-looking volatility estimate.

- **Implied volatility (IV) method**:  
Underlying's volatility is calculated from the volatility implied by the prices of traded options for the given underlying. Specifically,
this is done by inputting the current market price of an option (typically a call or put) into an options pricing model, and solving for the
volatility value that equates the theoretical price to the observed market price.
If the user selects the IV method, two additional lines of output are shown:
- **Strike price used in IV estimation**: nearest traded strike to the user’s input.
- **Expiry date used in IV estimation**: nearest available expiry date to the user’s input.

## Requirements
- Python 3.8 or higher
- Packages: `yfinance`, `numpy`, `pandas`, `scipy`, `arch`

## Usage Instructions

1. Install required packages as per the below:
```bash
pip install yfinance numpy pandas scipy arch
```
or
```bash
pip install -r requirements.txt
```

2. Run the program:
```bash
python main.py -ticker NVDA
```
The program is **not limited** to the Nvidia stock ticker; **any ticker supported by the yfinance API** can be used. If no ticker provided, 
the default ticker "AAPL" for Apple stock will be used instead.

3. Once the program is initiated, the user is prompted for the following inputs:
- **Option's type**: call or put.
- **Option's time to expiration in months**: any reasonable integer representing the option’s time to expiration in **months** (e.g., 3, 6,
9, 12)
- **Option's strike price**: any reasonable option's strike price.
- **Risk-free rate**: market-aligned risk-free rate reflecting present economic conditions.
- **Volatility estimation preference**: user may choose realized volatility (RV), GARCH, or implied volatility (IV) volatility estimation
method.

4. After inputs are provided, the program will output the option's **theoretical** price and Greeks.
## Example Session (NVDA, as of 22.10.2025)
```text
python main.py -ticker NVDA
Choose option's type: call
Choose option's time to maturity in months: 3
Choose option's strike price: 180
Choose risk-free rate (ex. 3%): 4
Choose volatility estimation method among realized vol, GARCH, or implied vol: iv
Strike price used in IV estimation: 180.0
Expiry date used in IV estimation (nearest to chosen option's expiration in months): 2025-12-19
Price: $13.2283
Volatility: 39.59%
Delta: 0.5219
Gamma: 0.0114
Vega: 0.3519
Theta: -0.123
Rho: 0.1974
```

## Limitations
- Supports only European plain vanilla call and put options.
- Assumes constant volatility and interest rates, which are often violated in real-world markets.
- GARCH implementation is fixed to a (1,1) specification.
