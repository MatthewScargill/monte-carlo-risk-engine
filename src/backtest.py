import pandas as pd
import numpy as np
from calibration import *
from var_es import *

import pandas as pd
import numpy as np


weights= np.array([1.0])

# basic var backtest, returns number of exceptions where real life was worse than prediction
def var_backtest():

    # this produces all the returns for the whole dataset 
    df = pd.read_csv("data/18msynthstock.csv", parse_dates=["Date"]).set_index("Date")
    # load in CSVs and run calibrate
    prices = df[["Adj Close"]]          # shape (T, 1)
    returns = to_returns(prices, method="log")

    #print(returns)

    # this is setting up the required data
    train_size = 50
    test_size = len(returns) - train_size
    exceptions = 0

    for t in range(train_size, len(returns)):

        # Training window = first t rows
        train_window = returns.iloc[:t]     # grows each loop

        # Estimate mean & stdev from training window
        mu, sigma = estimate_mean_cov(train_window)

        # MC samples
        samples = simulate_mc_portfolio_returns(mu, sigma, weights,seed=42)
        
        # 99% confidence var and es for next day
        var, es = var_es_from_samples(samples)

        lim = - returns.iloc[t,0]
        if var < lim:
            exceptions += 1
    return exceptions


g = var_backtest()
print(g)

# when you set the training size to low you can get an error
# stays 0 for more, ---> conservative model