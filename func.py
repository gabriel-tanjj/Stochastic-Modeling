import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import dill as pickle
import lzma
from scipy import interpolate

def save_pickle(filepath, obj):
    with lzma.open(filepath, "wb") as fp:
        pickle.dump(obj, fp)
        
def load_pickle(filepath):
    with lzma.open(filepath, "rb") as fp:
        data = pickle.load(fp)
    return data

def compute_linear_forward(df_rates):
    df_rates['date'] = pd.to_datetime(df_rates['date'], format='%Y%m%d')
    df_rates['rate_date'] = df_rates['date'] + pd.to_timedelta(df_rates["days"], unit='D')
    df_rates['year_frac'] = df_rates['rate_date'].dt.year + \
                            df_rates['rate_date'].dt.dayofyear / 365.25 - \
                            (df_rates['date'].dt.year + df_rates['date'].dt.dayofyear / 365.25)

    return df_rates

def linear_interpolate(rates, time_to_maturity):
    linear_interpolator = interpolate.interp1d(rates["year_frac"], rates["rate"], fill_value='extrapolate')
    r = linear_interpolator(time_to_maturity) / 100
    return r

def calculate_bsm_price(row, S, r, sigma):
    spot = (row['best_bid'] + row['best_offer']) / 2
    vol = row['mid_price'].rolling(window=30).std()
    strike = row['strike_price'] / 1000 
    t = (row['exdate'] - row['date']) / 365 / 10000 
    
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = (np.log(spot / strike) + (r - 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    
    if row['cp_flag'] == 'C':
        return spot * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
    elif row['cp_flag'] == 'P':
        return strike * np.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
def calculate_bach_price(row):
    spot = (row['best_bid'] + row['best_offer']) / 2
    vol = row['mid_price'].rolling(window=30).std()
    strike = row['strike_price'] / 1000 
    t = (row['exdate'] - row['date']) / 365 / 10000 
    d = (spot - strike) / (vol * math.sqrt(t))
    bach_price = (spot - strike) * norm.cdf(d) + vol * math.sqrt(t) * norm.pdf(d)

    return bach_price

def calculate_black_price(row, r):
    F = (row['best_bid'] + row['best_offer']) / 2
    sigma = row['mid_price'].rolling(window=30).std()
    K = row['strike_price'] / 1000 
    T = (row['exdate'] - row['date']) / 365 / 10000 
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    black_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    return black_price


def calculate_ddm_price(row, r, beta):
    F = (row['best_bid'] + row['best_offer']) / 2
    sigma = row['mid_price'].rolling(window=30).std()
    K = row['strike_price'] / 1000
    T = (row['exdate'] - row['date']) / 365 / 10000

    F_adj = F * beta
    K_adj = K + ((1 - beta) / beta) * F
    sigma_adj = sigma / beta
    d1 = (np.log(F_adj / K_adj) + 0.5 * sigma_adj ** 2 * T) / (sigma_adj * np.sqrt(T))
    d2 = d1 - sigma_adj * np.sqrt(T)
    ddm_price = np.exp(-r * T) * (F_adj * norm.cdf(d1) - K_adj * norm.cdf(d2))

    return ddm_price


