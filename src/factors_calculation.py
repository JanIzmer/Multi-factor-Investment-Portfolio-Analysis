# src/factors_calculation.py
import pandas as pd
from sqlalchemy import text
from database.src.connection import get_engine, setup_engine


MARKET_TICKER = "SPY"
ROLLING_WINDOW = 60  
MIN_PERIODS = 20


def load_returns(engine):
    """
    Load returns + tickers from DB.
    Returns DataFrame: date, asset_id, ticker, return
    """
    q = """
    SELECT r.date, a.asset_id, a.ticker, r.return
    FROM returns r
    JOIN assets a USING(asset_id)
    ORDER BY r.date
    """
    df = pd.read_sql(q, engine, parse_dates=["date"])
    return df


def compute_factors(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling beta and volatility for all tickers (including market).
    Returns DataFrame with columns: asset_id, date, beta, volatility
    """
    if returns_df.empty:
        return pd.DataFrame(columns=["asset_id", "date", "beta", "volatility"])

    # pivot: index=date, columns=ticker
    pivot = returns_df.pivot(index="date", columns="ticker", values="return").sort_index()

    if MARKET_TICKER not in pivot.columns:
        raise ValueError(f"Market ticker {MARKET_TICKER} not found in returns table. Found: {list(pivot.columns[:20])}")

    market = pivot[MARKET_TICKER]

    results = []
    # precompute rolling var of market with min_periods
    market_var = market.rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS).var()
    # Also compute market rolling std for market volatility
    market_vol = market.rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS).std()

    for ticker in pivot.columns:
        series = pivot[ticker]

        # If ticker is market itself: beta = 1, volatility = rolling std of market
        if ticker == MARKET_TICKER:
            beta_series = pd.Series(1.0, index=market.index)
            vol_series = market_vol
        else:
            # rolling covariance between asset and market
            rolling_cov = series.rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS).cov(market)
            # avoid division by zero; where market_var is 0 -> beta = NaN
            beta_series = rolling_cov / market_var
            vol_series = series.rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS).std()

        df_t = pd.DataFrame({
            "date": beta_series.index,
            "ticker": ticker,
            "beta": beta_series.values,
            "volatility": vol_series.values
        })

        df_t = df_t.dropna(subset=["beta", "volatility"], how="all")
        if not df_t.empty:
            results.append(df_t)

    if not results:
        return pd.DataFrame(columns=["asset_id", "date", "beta", "volatility"])

    factors_df = pd.concat(results, ignore_index=True)

    asset_map = returns_df[["asset_id", "ticker"]].drop_duplicates(subset=["ticker"])
    factors_df = factors_df.merge(asset_map, on="ticker", how="left")

    factors_df["date"] = pd.to_datetime(factors_df["date"]).dt.date
    factors_df = factors_df[["asset_id", "date", "beta", "volatility"]]

    factors_df.dropna(inplace = True)

    return factors_df


def upsert_factors(engine, df: pd.DataFrame):
    """
    INSERT ... ON DUPLICATE KEY UPDATE into `factors`
    """
    if df.empty:
        print("No factors to insert.")
        return 0

    rows = []
    for _, r in df.iterrows():
        asset_id = int(r["asset_id"])
        date_iso = r["date"].isoformat()
        beta = float(r["beta"])
        vol = float(r["volatility"])
        rows.append((asset_id, date_iso, beta, vol))

    sql = """
    INSERT INTO factors (asset_id, date, beta, volatility)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        beta = VALUES(beta),
        volatility = VALUES(volatility)
    """

    with engine.begin() as conn:
        raw = conn.connection
        cur = raw.cursor()
        try:
            cur.executemany(sql, rows)
            raw.commit()
        finally:
            cur.close()

    return len(rows)


def run_factors_pipeline():
    engine = get_engine()

    print("Loading returns from DB...")
    df = load_returns(engine)

    print("Computing factors...")
    factors_df = compute_factors(df)

    print(f"Computed {len(factors_df)} factor rows.")

    print("Upserting into DB...")
    n = upsert_factors(engine, factors_df)

    print(f"Done. Inserted/updated {n} rows.")


if __name__ == "__main__":
    setup_engine()
    run_factors_pipeline()
