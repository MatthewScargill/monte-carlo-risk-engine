import pandas as pd
import numpy as np

from calibration import *
from var_es import *

df = pd.read_csv("data/AAPL_SIM.csv", parse_dates=["Date"]).set_index("Date")
# load in CSVs and run calibrate
prices = df[["Adj Close"]]          # shape (T, 1)
rets = to_returns(prices, method="log")
mu, Sigma = estimate_mean_cov(rets, cov_method="ewma", lam=0.94)

# weights ( so far only one asset so 1 )
w = np.array([1.0])

# Parametric VaR/ES (Normal)
mu_p, sigma_p = portfolio_params(mu.values, Sigma.values, w)
var_p, es_p = parametric_var_es_normal(mu_p, sigma_p, alpha=0.99)
print("Parametric Normal VaR/ES (1d):", var_p, es_p)

# Monte Carlo VaR/ES (Normal)
samps = simulate_mc_portfolio_returns(mu.values, Sigma.values, w,
                                      n_sims=200000, horizon_days=1,
                                      dist="normal", antithetic=True, seed=42)
var_mc, es_mc = var_es_from_samples(samps, alpha=0.99)
print("MC Normal VaR/ES (1d):        ", var_mc, es_mc)

# Monte Carlo with Student-t (fatter tails for better real world performance )
samps_t = simulate_mc_portfolio_returns(mu.values, Sigma.values, w,
                                        n_sims=200000, horizon_days=1,
                                        dist="student", df=7.0, antithetic=True, seed=42)
var_t, es_t = var_es_from_samples(samps_t, alpha=0.99)
print("MC Student-t VaR/ES (1d):     ", var_t, es_t)