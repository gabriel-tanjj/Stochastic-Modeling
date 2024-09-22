import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import dill as pickle
import lzma

def save_pickle(filepath, obj):
    with lzma.open(filepath, "wb") as fp:
        pickle.dump(obj, fp)
        
def load_pickle(filepath):
    with lzma.open(filepath, "rb") as fp:
        data = pickle.load(fp)
    return data

def bsm(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    N = lambda x: norm.cdf(x) 
    bsm_price = S * N(d1) - K * exp(-r * T) * N(d2)
    
    return bsm_price

def bach(S, K, sigma, T):
    d = (S - K) / (sigma * math.sqrt(T))
    bach_price = (S - K) * norm.cdf(d) + sigma * math.sqrt(T) * norm.pdf(d)
    
    return bach_price

def black(F, K, T, r, sigma):
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    black_price = math.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    
    return black_price

def ddm(F, K, T, r, sigma, beta):
    F_adj = F * beta
    K_adj = K + ((1 - beta) / beta) * F
    sigma_adj = sigma / beta

    d1 = (math.log(F_adj / K_adj) + 0.5 * sigma_adj ** 2 * T) / (sigma_adj * math.sqrt(T))
    d2 = d1 - sigma_adj * math.sqrt(T)

    ddm_price = math.exp(-r * T) * (F_adj * norm.cdf(d1) - K_adj * norm.cdf(d2))
    return ddm_price


