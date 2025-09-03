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

def ewma_cov(returns, lam=0.94): 
    """
    Produce 'Exponentially Weighted Moving Average' (EMWA) covariance matrix.

    Cov ≈ (1-λ) * Σ_{i=0}^{T-1} λ^i * (r_{t-i}-μ)(r_{t-i}-μ)^T
    Idea: Put greater importance on newer events (eg. stops incoming crisis data being dampened by long stability phase)

    - returns: DataFrame of returns (T x N)
    - lam: decay parameter in (0,1). Higher = longer memory. 
        lam = 0.94 is industry standard for short terms - range up to 0.97.

    Returns: N x N numpy array covariance.
    """
    x = returns.values 
    T, N = x.shape # rows: T = number of days, columns: N = assets

    mu = np.nanmean(x, axis=0, keepdims=True)  
    xc = x - mu # remove the mean to isolate fluctuations                             

    idx = np.arange(T-1, -1, -1)
    w = (1 - lam) * (lam ** idx) # add weighting scheme - favours newer data
    w = w / w.sum() # normalise!                           

    Xw = xc * w[:, None] # weight scaling                      
    cov = (Xw.T @ xc)                          

    return cov

def diagonal_shrinkage(cov, alpha):
    """ 
    Simple diagonal shrinkage:

    Σ_shrunk = (1-α) Σ + α diag(Σ)
    Idea: Since the off diagonal correlations are most sensitive and likely to be
        wrong with limited data, dampen their influence.

    - cov: N x N numpy array of covariance
    - alpha: shrinkage parameter in [0,1]

    Returns: covariance matrix with off diagonal components proportionally dampened
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("shrinkage alpha must be in [0,1]")
    d = np.diag(np.diag(cov))
    return (1.0 - alpha) * cov + alpha * d

def estimate_mean_cov(returns, cov_method="standard", lam=0.94, shrinkage_alpha=None, ddof=1):
    """
    Estimate mean vector and covariance matrix.
    
    returns: DataFrame of returns (T x N)
    cov_method: 'standard' (sample covariance) or 'ewma' (Exponentially Weighted Moving Average covariance)
    lam: EWMA decay λ if cov_method='ewma'
    shrinkage_alpha: optional diagonal shrinkage α in [0,1]
    ddof           : degrees of freedom for sample covariance (default 1)

    Returns: (mu, Sigma) where
        mu    : pd.Series of mean returns by asset
        Sigma : pd.DataFrame covariance matrix (assets x assets)
    """
    rets = returns.dropna(how="any")
    assets = rets.columns 

    # finding the mean
    mu = rets.mean()

    # picking a covariance method + error 
    if cov_method == "standard":
        Sigma = rets.cov(ddof=ddof)
    elif cov_method == "ewma":
        cov = ewma_cov(rets, lam=lam)
        Sigma = pd.DataFrame(cov, index=assets, columns=assets)
    else:
        raise ValueError("cov_method must be 'standard' or 'ewma'")

    if shrinkage_alpha is not None: # apply the shrinkage if that's your jam
        shrunk = diagonal_shrinkage(Sigma.values, shrinkage_alpha)
        Sigma = pd.DataFrame(shrunk, index=assets, columns=assets)

    return mu, Sigma
