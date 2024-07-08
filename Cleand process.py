import pandas as pd
from functools import reduce
from scipy.optimize import minimize
import numpy as np
import itertools

# Add your data here
dataframes = {
    'EHFI253': pd.read_excel('EHFI253 Index.xlsx'),
    #'PEBUY': pd.read_excel('PEBUY Index.xlsx'),
    'PEDEBT': pd.read_excel('PEDEBT Index.xlsx'),
    'PEGROW': pd.read_excel('PEGROW Index.xlsx'),
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
pre_model_data.to_csv('cleaned_data.csv', index=False, header=True)

