import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from calibration import estimate_mean_cov, to_returns


def ensure_posdef(Sigma, jitter=1e-10, max_tries=6):
    """
    Make covariance PSD for Cholesky decomp by adding jitter to diagonal if needed.
    Cholesky decomp required for Monte Carlo (avoids blow up).

    - Sigma: pd.DataFrame covariance matrix
    - jitter: small correction

    Returns: lower triangular Cholesky factor for MC simulation
    """
    S = np.array(Sigma, dtype=float)
    for k in range(max_tries):
        try:
            L = np.linalg.cholesky(S)
            return L
        except np.linalg.LinAlgError:
            S = S + np.eye(S.shape[0]) * (jitter * (10 ** k))

    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 1e-12, None) # clip just in case 
    S_fixed = (vecs * vals) @ vecs.T

    return np.linalg.cholesky(S_fixed)


def portfolio_params(mu, Sigma, weights):
    """
    Collapse vector mu (N,), matrix Sigma (N,N) and weights (N,) to portfolio μ, σ.

    Returns: (mu_p, sigma_p) where 
        mu_p: float portfolio mean
        sigma_p: float portfolio variance
    """
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    mu_vec = np.asarray(mu, dtype=float).reshape(-1, 1)
    Sigma_mat = np.asarray(Sigma, dtype=float)

    mu_p = float((w.T @ mu_vec)[0, 0])
    var_p = float(w.T @ Sigma_mat @ w)
    sigma_p = np.sqrt(var_p)
    
    return mu_p, sigma_p

