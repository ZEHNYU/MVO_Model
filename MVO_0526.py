import pandas as pd
from functools import reduce
from scipy.optimize import minimize
import numpy as np
import itertools

# Add your data here
dataframes = {
    'EHFI253': pd.read_excel('EHFI253 Index.xlsx'),
    'PEBUY': pd.read_excel('PEBUY Index.xlsx'),
    'PEGROW': pd.read_excel('PEGROW Index.xlsx'),
    'PEDEBT': pd.read_excel('PEDEBT Index.xlsx'),
    'PEVC': pd.read_excel('PEVC Index.xlsx'),
    'TRVCI': pd.read_excel('TRVCI Index.xlsx')
}


def preprocess_function(df, key):
    try:
        df = df.drop(columns=['Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'])
    except KeyError:
        pass
    df.columns = df.iloc[5]
    df = df.drop(df.index[:6])
    df.reset_index(drop=True, inplace=True)
    df['% Change'] = pd.to_numeric(df['% Change'], errors='coerce')
    df['% Change'] = df['% Change'] / 100
    try:
        df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M')
    except Exception as e:
        print(f"Error converting Date: {e}")
    try:
        df = df.drop(columns=['PX_LAST', 'Change'])
    except KeyError:
        pass
    df = df.add_prefix(f"{key}_")
    df.rename(columns={f"{key}_Date": "Date", f"{key}_% Change": f"{key}_Change"}, inplace=True)

    return df


for key in dataframes:
    dataframes[key] = preprocess_function(dataframes[key], key)
processed_dfs = list(dataframes.values())


def merge_dataframes(dfs, on_column):
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=on_column, how='inner'), dfs)
    return merged_df


merged_df = merge_dataframes(processed_dfs, on_column='Date')
pre_model_data = merged_df.dropna()

# Cleaned data
# print(pre_model_data)

# Uncomment to save the cleaned data
# pre_model_data.to_csv('cleaned_data.csv', index=False, header=True)

# MVO model
mvo_data = pre_model_data.drop(columns=['Date'])
mean_returns = mvo_data.mean()
cov_matrix = mvo_data.cov()
risk_free_rate = 0.02 / 12  # Monthly risk-free rate


def modified_sharpe_ratio(weights, cov_matrix, expected_returns, risk_free_rate):
    annual_return = weights.dot((1 + expected_returns) ** 12 - 1)
    annual_volatility = np.sqrt(weights.dot(cov_matrix).dot(weights)) * np.sqrt(12)
    modified_sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    return -modified_sharpe_ratio


# Function to calculate optimal weights and Sharpe ratio for a given asset combination
def calculate_optimal_sharpe_ratio(assets):
    sub_data = mvo_data[assets]
    sub_mean_returns = sub_data.mean()
    sub_cov_matrix = sub_data.cov()

    num_assets = len(sub_mean_returns)
    initial_weights = np.ones(num_assets) / num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    result = minimize(modified_sharpe_ratio, initial_weights, args=(sub_cov_matrix, sub_mean_returns, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    max_sharpe_ratio = -result.fun

    return optimal_weights, max_sharpe_ratio


# Generate all combinations of assets and calculate Sharpe ratios
def find_best_combination(data, max_assets=None):
    asset_names = data.columns.tolist()
    if max_assets is None:
        max_assets = len(asset_names)

    best_sharpe_ratio = -np.inf
    best_combination = None
    best_weights = None
    best_return = None
    best_volatility = None

    for r in range(1, max_assets + 1):
        for combination in itertools.combinations(asset_names, r):
            weights, sharpe_ratio = calculate_optimal_sharpe_ratio(list(combination))
            if sharpe_ratio > best_sharpe_ratio:
                best_sharpe_ratio = sharpe_ratio
                best_combination = combination
                best_weights = weights

                # Calculate return and volatility for the best combination
                sub_data = mvo_data[list(combination)]
                sub_mean_returns = sub_data.mean()
                sub_cov_matrix = sub_data.cov()

                best_return = weights.dot(sub_mean_returns) * 12  # Annual return
                best_volatility = np.sqrt(weights.dot(sub_cov_matrix).dot(weights)) * np.sqrt(12)  # Annual volatility

    return best_combination, best_weights, best_sharpe_ratio, best_return, best_volatility


best_combination, best_weights, best_sharpe_ratio, best_return, best_volatility = find_best_combination(mvo_data)

print("Best Combination of Assets:", best_combination)
print("Optimal Weights for Best Combination:", best_weights)
print("Maximum Sharpe Ratio:", best_sharpe_ratio)
print("Expected Annual Return:", best_return)
print("Annual Volatility (Risk):", best_volatility)