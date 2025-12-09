# src/risk/optimization.py
import numpy as np
from scipy.optimize import minimize

def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Compute expected return and volatility of portfolio
    """
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def min_volatility(mean_returns, cov_matrix):
    """
    Minimize portfolio volatility
    """
    n = len(mean_returns)
    args = (cov_matrix,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                      x0=np.array([1/n]*n),
                      bounds=bounds,
                      constraints=constraints)
    return result.x

def max_sharpe(mean_returns, cov_matrix, risk_free=0.02):
    """
    Maximize Sharpe ratio
    """
    n = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    def neg_sharpe(weights):
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(ret - risk_free) / vol

    result = minimize(neg_sharpe, x0=np.array([1/n]*n),
                      bounds=bounds,
                      constraints=constraints)
    return result.x
