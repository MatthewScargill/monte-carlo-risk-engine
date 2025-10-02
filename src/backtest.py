import pandas as pd
import numpy as np
from calibration import *
from var_es import *
from scipy.stats import chi2
import numpy as np


# precomputed stuff

weights= np.array([1.0])
# this produces all the returns for the whole dataset 
df = pd.read_csv("data/18msynthstock.csv", parse_dates=["Date"]).set_index("Date")
# load in CSVs and run calibrate
prices = df[["Adj Close"]]          # shape (T, 1)
returns = to_returns(prices, method="log")


def var_backtest(returns, train_size):
    """
    Trains over *train_size* days and implements rolling 99% var backtest over the remaining days in the
    csv

    - returns: DataFrame of returns (T x N)
    - train_size: number of days the model is trained on before beginning rolling backtest

    Returns: (exceptions, test tize) where:
        - exceptions: number of times where real life was worse than prediction
        - training days: number of days in the rolling backtest
    """

    # initialise number of tested days and number of exceptions
    test_size = len(returns) - train_size
    exceptions = 0

    # check number of training days shorter than available data
    assert train_size < len(returns)

    # loop over remaining days in the data set 
    for t in range(train_size, len(returns)):

        # Training window = first t rows
        train_window = returns.iloc[:t]     # grows each loop

        # Estimate mean & sigma from training window
        mu, sigma = estimate_mean_cov(train_window)

        # MC samples
        samples = simulate_mc_portfolio_returns(mu, sigma, weights,seed=42)
        
        # 99% confidence var and es for next day
        var, es = var_es_from_samples(samples)

        # checking against confidence
        lim = - returns.iloc[t,0]
        if var < lim:
            exceptions += 1

    return exceptions, test_size

# checking
x, N = var_backtest(returns, 50)
print(x)

# when you set the training size to low you can get an error
# stays 0 for more, ---> conservative model
# also at 50 you can see student dist perform better 

# ---------------------------------------------------------------------------------------------------------



# think of exceptions as being binomially diustributed 
# L0 is the ideal distribution, L1 is the data suggested one, with LR being the log of the quotient of these
# if L0 similar to L1, log(L0/L1) -> 0
# if the model is correct then LR is distributed like chi2(1), therefore take 95% confidence interval of
# that and compare with calculated LR
# if 95% CI of chi2(1) is larger than LR, don't reject model as "number of exceptions is statistically consistent
# with the claimed confidence level"
alpha = 0.99
def kupiec_backtest():

    p = 1 - alpha  # expected exception rate
    if N <= 0:
        raise ValueError("N must be positive")

    # observed exception rate
    phat = x / N if N > 0 else 0.0
    # avoid edge cases blowing up logs
    eps = 1e-12
    phat = min(max(phat, eps), 1 - eps)

    # Likelihood ratio
    L0 = (1 - p)**(N - x) * (p**x)
    L1 = (1 - phat)**(N - x) * (phat**x)
    lr_uc = -2 * np.log(L0 / L1 + eps)

    # p-value (needs SciPy)
    try:
        p_value = chi2.sf(lr_uc, df=1)
    except Exception:
        p_value = float("nan")

    return lr_uc, p_value

lr_uc, p_val = kupiec_backtest()
print(f"Kupiec UC test: LR={lr_uc:.3f}, p-value={p_val:.3f}")

