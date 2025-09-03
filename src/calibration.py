import numpy as np
import pandas as pd
from scipy import stats

def to_returns(prices, method="log", dropna=True): # dropna to remove nan returns ( if diff(0))
    """
    Convert prices (£) to returns (%).

    - prices: DataFrame (wide) or Series of prices
    - method: 'log' or 'simple' --- log set as default as 
        continuous symmetric and normally distributed (+ less bias that normal)
        > required for MC 
    - dropna: drop the initial NaNs after differencing --- False preserves structure but True 
        makes for easier data to work with

    Returns: DataFrame of returns
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    if method == "log": # set as default as continuous symmetric and normally distributed (+ less bias that normal) -> required for MC
        rets = np.log(prices).diff()
    elif method == "simple": 
        rets = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'") 

    return rets.dropna(how="all") if dropna else rets 

