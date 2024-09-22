import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt


def calculate_time_to_maturity(exdate, date):
    exp_date = pd.to_datetime(exdate, format='%Y%m%d')
    curr_date = pd.to_datetime(date, format='%Y%m%d')

    time_to_maturity = (exp_date - curr_date).days / 365.0
    return max(time_to_maturity, 0)


def displaced_diffusion_price(S, K, T, r, sigma, delta, F, option_type='call'):
    try:
        d1 = (np.log((delta * S + (1 - delta) * F) / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = np.exp(-r * T) * ((delta * S + (1 - delta) * F) * norm.cdf(d1) - K * norm.cdf(d2))
        elif option_type == 'put':
            price = np.exp(-r * T) * (K * norm.cdf(-d2) - (delta * S + (1 - delta) * F) * norm.cdf(-d1))
        else:
            raise ValueError(f"Invalid option type: {option_type}")
    except Exception as e:
        print(f"Error in pricing: {e}")
        price = 0.0

    return price


def calibration_error(params, df, S, r):
    sigma, delta, F = params
    total_error = 0

    for index, row in df.iterrows():
        K = row['strike_price_scaled']
        best_bid = row['best_bid']
        best_offer = row['best_offer']
        market_price = (best_bid + best_offer) / 2
        T = calculate_time_to_maturity(row['exdate'], row['date'])

        option_type = 'call' if row['cp_flag'] == 'C' else 'put'

        model_price = displaced_diffusion_price(S, K, T, r, sigma, delta, F, option_type)

        total_error += (model_price - market_price) ** 2

    return total_error


print("hello")

file1 = "hello"
file2 = "hello"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print("DataFrame 1:")
print(df1.shape)
print(df1.head())

print("DataFrame 2:")
print(df2.shape)
print(df2.head())

df = pd.concat([df1, df2], ignore_index=True)

print("Merged DataFrame:")
print(df.shape)
print(df.head())

if df[['strike_price', 'best_bid', 'best_offer', 'exdate', 'date']].isnull().any().any():
    raise ValueError("DataFrame contains missing values in critical columns.")

df['strike_price_scaled'] = df['strike_price'] / 1000

S = 150
r = 0.01

forward_price_initial = df['strike_price_scaled'].mean()
initial_guess = [0.2, 0.5, forward_price_initial]
bounds = [(0.001, None), (0.001, 1), (0.001, None)]

result = minimize(calibration_error, initial_guess, args=(df, S, r), method='L-BFGS-B', bounds=bounds)

if result.success:
    opt_sigma, opt_delta, opt_F = result.x
    print(f"Calibrated Sigma: {opt_sigma:.4f}, Delta: {opt_delta:.4f}, Forward Price: {opt_F:.4f}")
else:
    print("Optimization failed:", result.message)

df['Model Price'] = df.apply(lambda row: displaced_diffusion_price(
    S,
    row['strike_price_scaled'],
    calculate_time_to_maturity(row['exdate'], row['date']),
    r,
    opt_sigma,
    opt_delta,
    opt_F,
    'call' if row['cp_flag'] == 'C' else 'put'
), axis=1)

df['Error'] = df['Model Price'] - ((df['best_bid'] + df['best_offer']) / 2)

print("hi")
