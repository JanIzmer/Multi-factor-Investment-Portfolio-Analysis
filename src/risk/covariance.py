# src/risk/covariance.py
import numpy as np
import pandas as pd
from database.src.connection import get_engine
from typing import Optional

def load_returns(tickers: Optional[list] = None, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None):
    """
    Load returns from the database.
    Returns a pivot table: index=date, columns=ticker, values=return
    """
    engine = get_engine()
    base_query = """SELECT date, asset_id, `return` FROM returns"""
    conditions = []
    
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")
    if conditions:
        query = base_query + " WHERE " + " AND ".join(conditions)
    else:
        query = base_query
    df = pd.read_sql(query, engine)

    # Load tickers for mapping asset_id → ticker
    tickers_df = pd.read_sql("SELECT asset_id, ticker FROM assets", engine)
    df = df.merge(tickers_df, on="asset_id")
    df = df.pivot(index="date", columns="ticker", values="return")
    if tickers:
        available_tickers = [t for t in tickers if t in df.columns]
        if available_tickers:
            df = df[available_tickers]
        else:
            return pd.DataFrame()
    return df

def align_returns(returns_df):
    """
    Keep only dates where every asset has a return (listwise deletion).

    Pandas computes .cov()/.mean() with pairwise deletion, so each cell of the
    matrix would come from a different subset of dates. That can produce a
    non-PSD covariance matrix and means estimated over different periods, which
    the optimizer then treats as comparable. Estimating on a common date panel
    avoids both problems.
    """
    if returns_df is None or returns_df.empty:
        return returns_df
    return returns_df.dropna(how="any")

def nearest_psd(cov, eps=1e-12):
    """
    Project a symmetric matrix onto the nearest PSD matrix by clipping negative
    eigenvalues to `eps` (Higham's projection step).

    Returns (cov_psd, min_eigenvalue_before). Callers can surface the second
    value to tell the user the input matrix was not a valid covariance matrix.
    """
    if isinstance(cov, pd.DataFrame):
        index, columns, values = cov.index, cov.columns, cov.to_numpy(dtype=float)
    else:
        index = columns = None
        values = np.asarray(cov, dtype=float)

    values = (values + values.T) / 2.0  # kill asymmetry from float error
    eigvals, eigvecs = np.linalg.eigh(values)
    min_eig = float(eigvals.min())
    if min_eig >= 0:
        fixed = values
    else:
        fixed = eigvecs @ np.diag(np.clip(eigvals, eps, None)) @ eigvecs.T
        fixed = (fixed + fixed.T) / 2.0

    if index is not None:
        fixed = pd.DataFrame(fixed, index=index, columns=columns)
    return fixed, min_eig


def ledoit_wolf_shrinkage(returns_df):
    """
    Ledoit-Wolf (2004) shrinkage of the sample covariance towards a
    constant-correlation target, on daily returns.

    The sample covariance of ~p assets from ~T observations is dominated by
    estimation error in its smallest eigenvalues, which is exactly where a
    min-variance optimizer puts its weight. Shrinking pulls those eigenvalues
    up and, unlike a diagonal epsilon, does it with a data-driven intensity.

    Returns (cov_daily, delta) where delta is the shrinkage intensity in [0, 1].
    """
    x = align_returns(returns_df)
    t, n = x.shape
    if t < 2 or n < 2:
        return x.cov(), 0.0

    values = x.to_numpy(dtype=float)
    values = values - values.mean(axis=0)
    # 1/T normalization, as the Ledoit-Wolf formulas assume
    sample = values.T @ values / t

    var = np.diag(sample)
    std = np.sqrt(var)
    outer_std = np.outer(std, std)
    corr = sample / outer_std
    # average off-diagonal correlation
    mean_corr = (corr.sum() - n) / (n * (n - 1))

    target = mean_corr * outer_std
    np.fill_diagonal(target, var)

    # pi: variance of the sample covariance entries
    squares = values ** 2
    pi_mat = (squares.T @ squares) / t - sample ** 2
    pi_hat = pi_mat.sum()

    # rho: covariance between the sample entries and the target entries
    # theta_ii,ij = E[x_i^3 x_j] - var_i * S_ij
    term = ((squares * values).T @ values) / t - var[:, None] * sample
    ratio = outer_std / var[:, None]  # sqrt(s_jj / s_ii)
    rho_off = (mean_corr / 2.0) * (ratio * term + ratio.T * term.T)
    np.fill_diagonal(rho_off, 0.0)
    rho_hat = np.trace(pi_mat) + rho_off.sum()

    gamma_hat = float(((target - sample) ** 2).sum())
    if gamma_hat <= 0:
        delta = 0.0
    else:
        delta = float(np.clip(((pi_hat - rho_hat) / gamma_hat) / t, 0.0, 1.0))

    shrunk = delta * target + (1 - delta) * sample
    # rescale to the unbiased 1/(T-1) convention used elsewhere in the project
    shrunk *= t / (t - 1)
    return pd.DataFrame(shrunk, index=x.columns, columns=x.columns), delta


def compute_sample_cov(returns_df, shrinkage="ledoit_wolf"):
    """
    Compute the covariance matrix (annualized) on a common date panel.

    shrinkage:
        "ledoit_wolf" (default) — data-driven shrinkage intensity
        float in [0, 1]         — fixed intensity towards constant correlation
        None / "none"           — raw sample covariance

    The result is always projected onto the PSD cone, so downstream
    sqrt(w'Σw) can never see a negative variance.
    """
    aligned = align_returns(returns_df)
    if shrinkage in (None, "none", 0, 0.0):
        cov_daily = aligned.cov()
    else:
        cov_daily, delta = ledoit_wolf_shrinkage(aligned)
        if shrinkage != "ledoit_wolf":
            fixed_delta = float(np.clip(float(shrinkage), 0.0, 1.0))
            raw = aligned.cov()
            # re-mix with the caller's intensity: recover the target from the
            # data-driven mix, then apply the requested one
            if delta > 0:
                target = (cov_daily - (1 - delta) * raw) / delta
            else:
                target = raw
            cov_daily = fixed_delta * target + (1 - fixed_delta) * raw

    cov_annual = cov_daily * 252  # assume 252 trading days
    cov_annual, _ = nearest_psd(cov_annual)
    return cov_annual

def compute_expected_returns(returns_df):
    """
    Compute expected returns (annualized mean) on a common date panel.
    """
    mean_daily = align_returns(returns_df).mean()
    mean_annual = mean_daily * 252
    return mean_annual
