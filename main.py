# Black–Scholes Option Calculator
# Requirements: yfinance, pandas, numpy, scipy, arch

import argparse
import math
import statistics
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from arch import arch_model


parser = argparse.ArgumentParser(
    description="Price a European option on a stock using Black–Scholes model." \
    "Program is suited to price a European option on a stock paying dividend yield of q." \
    "Program's user has an option to choose among three volatility estimation methods: realized volatility," \
    "GARCH volatility estimation model, or implied volatility method"
)
parser.add_argument("-ticker", default="AAPL", type=str, help="Stock ticker (default: AAPL)")


#----------------------------------Main------------------------------------#
def main():
    args = parser.parse_args()
    tk = get_stock_ticker(args.ticker)

    opt_type = get_option_type()
    tenor_months = get_option_tenor()
    T = tenor_months / 12.0 
    K = get_option_strike()

    r = get_risk_free_rate()

    vol_pref = get_vol_preference()

    S0 = get_stock_price(tk)
    q = get_dividend_yield(tk)

    if vol_pref in ["iv", "implied volatility"]:
        sigma, K_IV, expiry_IV = get_vol(tk, vol_pref, T, S0, K, r, q, opt_type)
    else:
        sigma = get_vol(tk, vol_pref, T)

    d1, d2 = compute_d1_d2(S0, K, sigma, r, T, q)
    price = round(black_scholes(opt_type, S0, K, r, T, d1, d2, q), 6)
    greeks = calculate_greeks(opt_type, S0, K, d1, d2, T, q, sigma, r)

    if vol_pref in ["iv", "implied volatility"]:
        print(f"Strike price used in IV estimation: {K_IV}")
        print(f"Expiry date used in IV estimation (nearest to chosen option's expiration in monts): {expiry_IV}")

    print(f"Option price: {round(price, 4)} USD")
    print(f"Volatility: {round(sigma*100, 4)}%")
    for g, v in greeks.items():
        print(f"{g.capitalize()}: {round(v, 4)}")


#-----------------------------Helpers--------------------------------------#
def get_option_type() -> str:
    while True:
        t = input("Choose option's type (call/put): ").strip().lower()
        if t in ("call", "put"):
            return t
        print("Option's type should be either 'call' or 'put'.")


def get_option_tenor() -> int:
    while True:
        s = input("Choose option's time to expiration in months: ").strip()
        if "month" in s:
            s = s.replace("months", "").replace("month", "").strip()
        try:
            return int(s)
        except ValueError:
            print("Tenor should be integer months, e.g., 1/3/6/12.")


def get_option_strike() -> float:
    while True:
        try:
            return round(float(input("Choose option's strike price: ").strip()), 5)
        except ValueError:
            print("Strike's value should be numeric, e.g., 100 or 150.53.")


def get_vol_preference() -> str:
    while True:
        s = input("Choose volatility estimation method among realized vol, GARCH, or implied vol: ").strip().lower()
        try:
            if s in ["garch", "realized vol", "implied vol", "iv", "rv"]:
                return s
        except ValueError:
            print("Volatility estimation method should be provided as either realized vol (rv), GARCH, or implied vol (iv)")


def get_risk_free_rate() -> float: 
    while True: 
        s = input("Choose risk-free rate (e.g. 3%): ").strip() 
        try: 
            if s.endswith("%"): 
                return round(float(s.strip("%")) / 100.0, 5) 
            elif s.lower().endswith("percent"): 
                return round(float(s[:-7]) / 100.0, 5) 
            else: 
                return round(float(s) / 100.0, 5) 
        except ValueError: 
            print("Risk free rate should be provided as follows (examples): 3% / 3 % / 3 / 3 percent.")


def get_stock_ticker(ticker: str) -> yf.Ticker:
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise SystemExit(f"No data found for {ticker}")
    return tk


def get_stock_price(tk: yf.Ticker) -> float:
    latest = tk.history(period="1d")
    return round(float(latest["Close"].iloc[-1]), 5)


def get_dividend_yield(tk: yf.Ticker) -> float:
    q = getattr(tk, "fast_info", {}).get("dividendYield", None)
    if q is None:
        q = tk.info.get("dividendYield", 0) or 0
    return float(q or 0)/100 


"""Realized vol: volatility estimated from realized vol with 3 months (~63 trading days) lookback period"""
"""GARCH: volatilty estimated by implementing GARCH model using 'arch' library with 3y lookback period"""
"""Implied vol: volatility estimated by working out option's implied volatility from the bid/ask prices of the traded option""" 
"""with nearest expiry to T and nearest strike price to K."""
"""Newton-Raphson method was chosen to look for correct value of the option's implied volatility"""
def get_vol(tk: yf.Ticker, method, T, S, K, r, q, option_type) -> float:
    if method == "realized vol" or method == "rv":
        return historical_vol(tk)
    elif method == "garch":
        return garch(tk, T)
    else: #(method == "implied vol" or method == "iv")
        return iv(tk, T, S, K, r, q, option_type)


def historical_vol(tk: yf.Ticker) -> float:
        df = tk.history(period="3mo", auto_adjust = True)
        if df.empty:
            raise SystemExit("No price history data")
        df['log return'] = np.log(df['Close'] / df['Close'].shift(1))
        rets = df['log return'].dropna()
        if len(rets) < 2:
            raise SystemExit("Not enough data for volatility.")
        stdev = statistics.stdev(rets)
        return stdev * math.sqrt(252.0)


def garch(tk: yf.Ticker, T: float) -> float:
        df = tk.history(period="4y", auto_adjust = True)
        if df.empty or 'Close' not in df:
            raise SystemExit("No price history for GARCH model")
        rets_pct = 100.0 * np.log(df['Close'] / df['Close'].shift(1))
        rets_pct = rets_pct.dropna()
        if len(rets_pct) < 252:
            raise SystemExit("Not enough price returns data for GARCH model")
        am = arch_model(rets_pct, mean='Zero', vol='GARCH', p=1, q=1, rescale=False)
        res = am.fit(disp='off')
        h = max(1, int(round(252 * T)))
        f = res.forecast(horizon=h, reindex=False)
        var_path_pct = f.variance.iloc[-1].to_numpy()
        sigma_ann = float(np.sqrt(252.0 * (var_path_pct / 10000).mean()))
        return sigma_ann


def iv(tk: yf.Ticker, T: float, S: float, K:float, r: float, q: float, option_type: str) -> float:
    today = date.today()
    expiries = tk.options
    if not expiries:
        raise SystemExit("Options' expiry data not available")
    
    candidates = []
    for expiry in expiries:
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
        T_i = (expiry_date - today).days / 365.0
        if T_i > 0:
            candidates.append((expiry, T_i, abs(T - T_i)))
    
    if not candidates:
        raise SystemExit("No future expiry dates found")
    
    nearest_exp = min(candidates, key = lambda x: x[2])
    exp, T_used, _ = nearest_exp

    options = tk.option_chain(exp)
    chain = options.calls if option_type == 'call' else options.puts
    row = chain.iloc[(chain['strike'] - K).abs().argmin()]
    #return row['impliedVolatility']

    if pd.notna(row['bid']) and pd.notna(row['ask']) and row['bid'] > 0 and row['ask'] > 0:
        mid_price = (row['bid'] + row['ask']) / 2.0
    else:
        if pd.notna(row['lastPrice']) and row['lastPrice'] > 0:
            mid_price = row['lastPrice']
        else:
            raise SystemExit("No data for option's bid / ask / last price")
    
    #Check if option's price is within bounds
    S_q = S*math.exp(-q*T_used)
    K_r = row['strike']*math.exp(-r*T_used)
    if option_type == "call" and not max(S_q - K_r, 0) <= mid_price <= S_q:
        raise SystemExit("Call mid price is out of bounds")
    elif option_type == "put" and not max(K_r - S_q, 0) <= mid_price <= K_r:
        raise SystemExit("Put mid price is out of bounds")
    
    #Compute black scholes price for given parametres, use historical vol as initial guess for IV
    vol_guess = historical_vol(tk)
    tol = 1e-5
    max_iter = 100

    for _ in range(max_iter):
        d1, d2 = compute_d1_d2(S, row['strike'], vol_guess, r, T_used, q)
        bs_price = black_scholes(option_type, S, row['strike'], r, T_used, d1, d2, q)
        diff = bs_price - mid_price #don't check for abs here, because Newton-Raphson cares about signs
        if abs(diff) < tol: 
            return vol_guess, row['strike'], exp
        
        vega  = S * math.exp(-q*T_used) * norm.pdf(d1) * math.sqrt(T_used)
        if vega < 1e-8:
            break

        vol_guess -= diff/vega #Newton-Raphson method to find IV value
        if vol_guess <= 1e-6 or vol_guess > 5.0: #sanity check break
            break

    raise SystemExit("IV value has not converged")


def compute_d1_d2(S: float, K: float, sigma: float, r: float, t: float, q: float):
    if S <= 0 or K <= 0 or sigma <= 0 or t <= 0:
        raise ValueError("S, K, sigma, t must be positive.")
    denom = sigma * math.sqrt(t)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * t) / denom
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def black_scholes(opt_type: str, S: float, K: float, r: float, t: float, d1: float, d2: float, q: float) -> float:
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    C = disc_q * S * norm.cdf(d1) - disc_r * K * norm.cdf(d2)
    P = disc_r * K * norm.cdf(-d2) - disc_q * S * norm.cdf(-d1)
    return C if opt_type == "call" else P


def calculate_greeks(opt_type: str, S: float, K: float, d1: float, d2: float, t: float, q: float, sigma: float, r: float):
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)

    vega  = S * disc_q * norm.pdf(d1) * math.sqrt(t) * 0.01
    gamma = disc_q * norm.pdf(d1) / (S * sigma * math.sqrt(t))

    if opt_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -(S * disc_q * norm.pdf(d1) * sigma) / (2 * math.sqrt(t))
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        ) / 252.0
        rho = K * t * disc_r * norm.cdf(d2) * 0.01
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -(S * disc_q * norm.pdf(d1) * sigma) / (2 * math.sqrt(t))
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        ) / 252.0
        rho = -K * t * disc_r * norm.cdf(-d2) * 0.01

    return {
        "delta": delta,
        "gamma": gamma,
        "vega":  vega,
        "theta": theta,
        "rho":   rho
    }


if __name__ == "__main__":
    main()
