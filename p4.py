import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters per qns
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1 / 12
iter = 50000

def bsm_d(s, k, r, sigma, T, t):
    timedelta = T - t
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * timedelta) / (sigma * np.sqrt(timedelta))
    d2 = d1 - sigma * np.sqrt(timedelta)
    return d1, d2

def stock_path_sim(nowprice, r, sigma, T, N, iter):
    dt = T / N
    time_grid = np.linspace(0, T, N + 1)
    simulation_paths = np.zeros((iter, N + 1))
    simulation_paths[:, 0] = nowprice
    for i in range(1, N + 1):
        Z = np.random.standard_normal(iter)
        simulation_paths[:, i] = simulation_paths[:, i - 1] * \
                                 np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return simulation_paths, time_grid

def delta_hedging(simulation_paths, time_grid, N):
    dt = T / N
    hedging_error = np.zeros(iter)
    for m in range(iter):
        S = simulation_paths[m, :]
        phi = np.zeros(N + 1)
        mm_account = np.zeros(N + 1)
        d1, d2 = bsm_d(s=S[0],\
                       k=K,\
                       r=r,\
                       sigma=sigma,\
                       T=T,\
                       t=0)
        call_price = S[0] * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        phi[0] = norm.cdf(d1)
        mm_account[0] = call_price - phi[0] * S[0]

        for i in range(1, N + 1):
            t = time_grid[i]
            timedelta = T - t
            d1, d2 = bsm_d(s=S[i],\
                           k=K, \
                           r=r, \
                           sigma=sigma,\
                           T=T,\
                           t=t)
            phi[i] = norm.cdf(d1)
            mm_account[i] = mm_account[i - 1] * np.exp(r * dt) - (phi[i] - phi[i - 1]) * S[i]

        V_T = phi[-1] * S[-1] + mm_account[-1] * np.exp(r * dt)
        option_payoff = max(S[-1] - K, 0)
        hedging_error[m] = V_T - option_payoff
    return hedging_error


#run simulation
for N in [21, 84]:
    print("simulating paths")
    S_paths, time_grid = stock_path_sim(nowprice=S0,\
                                        r=r,\
                                        sigma=sigma,\
                                        T=T,\
                                        N=N,\
                                        iter=iter)
    hedging_error = delta_hedging(simulation_paths=S_paths,\
                                  time_grid=time_grid,\
                                  N=N)
    plt.figure()
    plt.hist(hedging_error, bins=50, edgecolor='k')
    plt.title(f'Hedging Error Histogram for N = {N}')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()