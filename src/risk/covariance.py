# src/risk/covariance.py
import pandas as pd
import numpy as np
from sqlalchemy import text
from database.src.connection import get_engine
import datetime
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
