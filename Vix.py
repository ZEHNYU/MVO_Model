import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm


# Calculate Delta using the Black-Scholes model
def calculate_black_scholes_delta(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0:
        raise ValueError("Time to expiration (T) and volatility (sigma) must be positive")

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = -norm.cdf(-d1)
    return delta


# Select options
def select_out_of_money_options(options, percentage=0.9):#<----change OTM percentage here
    sorted_options = options.sort_values(by='strike', ascending=True)
    index = int(len(sorted_options) * percentage)
    out_of_money_options = sorted_options.iloc[index:]
    return out_of_money_options


# Get options chain
def get_vix_options_chain():
    vix = yf.Ticker('^VIX') #<----change ticker here
    expiration_dates = vix.options
    options = vix.option_chain(expiration_dates[0])  # latest expiration date
    calls = options.calls
    puts = options.puts
    return calls, puts


# Calculate required number of contracts to hedge
def calculate_required_contracts(portfolio_value, portfolio_volatility, delta, contract_value):
    risk_exposure = portfolio_value * portfolio_volatility
    num_contracts = risk_exposure / (contract_value * delta)
    return num_contracts


# Main hedging function
def main():
    portfolio_value = 1000000  # <----change Portfolio value
    portfolio_volatility = 0.1137  # <----Expected portfolio volatility

    calls, puts = get_vix_options_chain()

    # Select out-of-the-money call and put options
    selected_call = select_out_of_money_options(calls).iloc[0]
    selected_put = select_out_of_money_options(puts).iloc[0]

    S = 15  # Current VIX index price
    K = selected_put['strike']  # Use the selected call option's strike price

    # Set T to 1 year
    T = 1

    r = 0.05  # Assumed risk-free rate
    sigma = selected_put['impliedVolatility']  # Use the selected call option's implied volatility

    # Ensure sigma is valid
    if sigma <= 0:
        print("Error: Implied volatility (sigma) is non-positive.")
        return

    # Calculate Delta for the selected call option
    delta = calculate_black_scholes_delta(S, K, T, r, sigma, option_type="put")
    contract_value = 100  # VIX option contract value

    # Calculate the required number of contracts to hedge
    required_contracts = calculate_required_contracts(portfolio_value, portfolio_volatility, delta, contract_value)
    print(f'Options to Hedge: {required_contracts}')


if __name__ == '__main__':
    main()