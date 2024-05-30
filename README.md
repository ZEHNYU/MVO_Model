# MVO_test
# Asset Allocation and Optimization Tool
This repository contains scripts to preprocess financial data, merge dataframes, and perform Mean-Variance Optimization (MVO) to find the optimal asset allocation that maximizes the Sharpe ratio.

Prerequisites
Make sure you have the following Python packages installed:

pandas
numpy
scipy

# Usage
## Data Preparation
Add your data files in the same directory as the script. The data files should be in Excel format. Update the dataframes dictionary with the filenames and their corresponding keys.

## Script Overview
Preprocess Function
preprocess_function(df, key): Preprocesses the dataframes by cleaning and formatting the columns.
Merge Dataframes
merge_dataframes(dfs, on_column): Merges multiple dataframes on a specified column.
## MVO Model
modified_sharpe_ratio(weights, cov_matrix, expected_returns, risk_free_rate): Calculates the modified Sharpe ratio for given weights.
calculate_optimal_sharpe_ratio(assets): Calculates the optimal weights and Sharpe ratio for a given combination of assets.
find_best_combination(data, max_assets=None): Generates all combinations of assets and calculates the Sharpe ratios to find the best combination.

# Output
The script will output the following information:

Best Combination of Assets: The combination of assets that provides the maximum Sharpe ratio.
Optimal Weights for Best Combination: The optimal weights for the best combination of assets.
Maximum Sharpe Ratio: The maximum Sharpe ratio achieved.
Expected Annual Return: The expected annual return for the best combination.
Annual Volatility (Risk): The annual volatility (risk) for the best combination.

# Functions
The scripts contain the following main functions:

## Preprocess Function
preprocess_function(df, key): Preprocesses the dataframes by dropping unnecessary columns, renaming columns, converting date formats, and normalizing the percentage change.
## Merge Dataframes
merge_dataframes(dfs, on_column): Merges multiple dataframes into one based on the 'Date' column.
## MVO Model
modified_sharpe_ratio(weights, cov_matrix, expected_returns, risk_free_rate): Calculates the modified Sharpe ratio, considering both the return and risk of the portfolio.
calculate_optimal_sharpe_ratio(assets): Optimizes the asset weights to maximize the Sharpe ratio for a given combination of assets.
find_best_combination(data, max_assets=None): Finds the best combination of assets that maximizes the Sharpe ratio, considering all possible combinations up to a specified maximum number of assets.
