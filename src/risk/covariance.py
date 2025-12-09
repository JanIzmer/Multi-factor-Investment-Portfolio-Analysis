# src/risk/covariance.py
import pandas as pd
import numpy as np
from sqlalchemy import text
from database.src.connection import get_engine

def load_returns(tickers=None):
    """
    Load returns from the database.
    Returns a pivot table: index=date, columns=ticker, values=return
    """
    engine = get_engine()
    query = "SELECT date, asset_id, return FROM returns"
    df = pd.read_sql(query, engine)

    # Load tickers for mapping asset_id → ticker
    tickers_df = pd.read_sql("SELECT asset_id, ticker FROM assets", engine)
    df = df.merge(tickers_df, on="asset_id")
    df = df.pivot(index="date", columns="ticker", values="return")

    if tickers:
        df = df[tickers]  # subset if requested

    return df

def compute_sample_cov(returns_df):
    """
    Compute sample covariance matrix (annualized)
    """
    cov_daily = returns_df.cov()
    cov_annual = cov_daily * 252  # assume 252 trading days
    return cov_annual

def compute_expected_returns(returns_df):
    """
    Compute expected returns (annualized mean)
    """
    mean_daily = returns_df.mean()
    mean_annual = mean_daily * 252
    return mean_annual
