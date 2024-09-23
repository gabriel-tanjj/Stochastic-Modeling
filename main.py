import pandas as pd
import dill as pickle
import lzma
import os
from func import save_pickle, load_pickle, compute_linear_forward, linear_interpolate

valuation_date = '20201201'

def save_local():   
    spx = pd.read_csv("/Users/gabriel/Library/Mobile Documents/com~apple~CloudDocs/SMU Masters\
     /QF620 Stochastic Modelling in Finance/Project/SPX_options.csv")
    spy = pd.read_csv("/Users/gabriel/Library/Mobile Documents/com~apple~CloudDocs/SMU Masters\
     /QF620 Stochastic Modelling in Finance/Project/SPY_options.csv")
    rates = pd.read_csv("/Users/gabriel/Library/Mobile Documents/com~apple~CloudDocs/SMU Masters\
     /QF620 Stochastic Modelling in Finance/Project/zero_rates_20201201.csv")
    save_pickle("spx.obj", spx)
    save_pickle("spy.obj", spy)
    save_pickle("rates.obj", rates)
    
def load_local():
    spx = load_pickle("spx.obj")
    spy = load_pickle("spy.obj")
    rates = load_pickle("rates.obj")
    
    return spx, spy, rates

spx, spy, rates = load_local()

rates = compute_linear_forward(df_rates=rates)
r = linear_interpolate(rates=rates, time_to_maturity=3)
print(spx)