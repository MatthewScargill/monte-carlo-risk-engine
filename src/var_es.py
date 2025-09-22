import numpy as np
from calibration import *
from scipy.stats import norm, t as student_t

def portfolio_params(mu, Sigma, weights):
    """
    Collapse vector mu (N,), matrix Sigma (N,N) and weights (N,) to portfolio μ, σ.
    idea: ultimately we want to know how a portfolio performs, if want a specific stock just
    make the original CSV the history of one stock / add stock picking functionality.

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

# parametric model for comparison (normal)
def parametric_var_es_normal(mu_p, sigma_p, alpha=0.99):
    """
    Closed-form VaR/ES under Normal for *log* return R ~ N(mu_p, sigma_p^2).
    Returns positive numbers for loss metrics (VaR/ES >= 0).

    VaR_alpha = -(mu_p + z_alpha * sigma_p)
    ES_alpha  = -(mu_p + sigma_p * phi(z_alpha)/(1-alpha))
    """
    z = norm.ppf(alpha)
    var = -(mu_p + z * sigma_p)
    es = -(mu_p + sigma_p * norm.pdf(z) / (1.0 - alpha))
    return float(var), float(es)

# parametric model for comparison (student tail)
def parametric_var_es_student(mu_p, sigma_p, df=7.0, alpha=0.99):
    """
    Closed-form VaR/ES under standardized Student-t, scaled to sigma_p.
    We use the known ES formula for t with df>1 and df>2 conditions.

    R = mu_p + sigma_p * T,  T ~ t_df (zero-mean, unit-scale)
    """
    if df <= 2:
        raise ValueError("Student-t ES undefined for df <= 2.")
    q = student_t.ppf(alpha, df)
    # scale so that std(T) = sqrt(df/(df-2)) ; our sigma_p already absorbs that,
    # because we define R = mu + sigma_p * T where sigma_p is the *observed* std.
    # VaR:
    var = -(mu_p + sigma_p * q)
    # ES:
    c = (df + q**2) / ((df - 1) * (1 - alpha))
    es = -(mu_p + sigma_p * student_t.pdf(q, df) * c)
    return float(var), float(es)

def parametric_var_es(mu_p, sigma_p, dist, df=7.0, alpha=0.99):
    """
    Closed-form parametric VaR/ES for portfolio returns.

    mu_p : Mean return.
    sigma_p : Standard deviation of returns.
    dist : Distribution assumption: "normal" or "student".
    df : Degrees of freedom for Student-t (only used if dist="student").
    alpha : Confidence level.

    Returns: (var, es) where
        var : Value-at-Risk
        es : Expected Shortfall
    """
    if dist == "normal":
        return parametric_var_es_normal(mu_p, sigma_p, alpha)
    
    if dist == "student":
        return parametric_var_es_student(mu_p, sigma_p, df, alpha)
    
    else: raise ValueError(f"Unsupported distribution '{dist}'. Use 'normal' or 'student'.")

# mc version 
def simulate_mc_portfolio_returns(
    mu, Sigma, weights,
    n_sims=100_000,
    horizon_days=1,
    dist="normal",    # "normal" or "student"
    df=7.0,           # used if dist == "student"
    antithetic=True,
    seed=None,
):
    """
    Simulate *portfolio* log-returns over a horizon using multivariate draws.

    mu, Sigma   : vector (N,), matrix (N,N) of daily *log-return* parameters
    weights     : (N,) portfolio weights (sum to 1 typically)
    n_sims      : number of Monte Carlo scenarios
    horizon_days: number of trading days to aggregate
    dist        : 'normal' or 'student' (multivariate t via Gaussian mixture)
    df          : degrees of freedom for student-t (typ. 5-10)
    antithetic  : if True, half draws are mirrored to reduce variance
    seed        : RNG seed (int) for reproducibility

    Returns: np.ndarray shape (n_sims,) of portfolio log-returns over horizon
    """
    
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)

    if mu.shape[0] != Sigma.shape[0] or Sigma.shape[0] != Sigma.shape[1] or w.shape[0] != mu.shape[0]:
        raise ValueError("Shapes mismatch: mu (N,), Sigma (N,N), weights (N,) required.")

    L = ensure_posdef(Sigma)

    # Antithetic control: generate m draws and mirror to make ~2m total
    if antithetic:
        m = (n_sims + 1) // 2
    else:
        m = n_sims

    # Function to draw one day's *vector* return
    def draw_day(m):
        z = rng.standard_normal(size=(m, len(mu)))
        n_draws = z @ L.T + mu  # (m, N)
        if dist == "normal":
            x = n_draws
        elif dist == "student":
            if df <= 2:
                raise ValueError("df must be > 2 for finite variance.")
            # Gaussian scale mixture: divide by sqrt(u/df), u ~ ChiSquare(df)
            u = rng.chisquare(df, size=(m, 1))
            scale = np.sqrt(u / df)
            x = n_draws / scale  # broadcast (m,1) over columns
        else:
            raise ValueError("dist must be 'normal' or 'student'")
        return x

    # Aggregate over horizon: sum log-returns across days
    # (Assumes i.i.d daily draws conditional on params)
    total = np.zeros((m, len(mu)))
    for _ in range(int(horizon_days)):
        total += draw_day(m)

    # Collapse to portfolio using weights
    port = total @ w  # (m,)

    # Antithetic augmentation
    if antithetic:
        port_full = np.concatenate([port, -port + 2 * (w @ mu) * horizon_days], axis=0)
        samples = port_full[:n_sims]
    else:
        samples = port[:n_sims]

    return samples


def var_es_from_samples(samples, alpha=0.99):
    """
    Compute empirical VaR and ES from simulated *log-return* samples.

    Returns positive numbers for loss metrics (VaR/ES >= 0):
        VaR_alpha = -quantile_alpha(samples)
        ES_alpha  = -mean(samples[samples <= quantile_alpha])
    """
    samples = np.asarray(samples, dtype=float).reshape(-1)
    q = np.quantile(samples, alpha)  # == quantile at alpha
    var = -float(q)
    tail = samples[samples <= q]
    es = -float(tail.mean()) if tail.size else var
    return var, es
