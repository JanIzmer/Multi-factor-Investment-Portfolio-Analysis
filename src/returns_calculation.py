# src/returns_calculation.py
import pandas as pd
from sqlalchemy import text
from database.src.connection import setup_engine, get_engine


def load_price_table(engine):
    """
    Load historical prices joined with asset tickers.
    Returns DataFrame with columns: date (datetime), asset_id (int), ticker (str), close (float)
    """
    q = """
    SELECT hp.date, a.asset_id, a.ticker, hp.close
    FROM historical_prices hp
    JOIN assets a USING (asset_id)
    ORDER BY a.asset_id, hp.date
    """
    df = pd.read_sql(q, engine, parse_dates=["date"])
    return df

def compute_daily_returns_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given DataFrame with columns (date, asset_id, ticker, close),
    return DataFrame with columns (asset_id, date, return)
    """
    if price_df.empty:
        return pd.DataFrame(columns=["asset_id", "date", "return"])

    # Pivot to have dates as index and tickers as columns
    pivot = price_df.pivot_table(index="date", columns="ticker", values="close")
    # Sort index to ensure chronological order
    pivot = pivot.sort_index()

    # Compute daily returns
    daily_returns = pivot.pct_change().dropna(how="all")

    # Unpivot / melt into long format: date, ticker, return
    ret = daily_returns.reset_index().melt(id_vars="date", var_name="ticker", value_name="return")
    # Drop rows where return is NaN
    ret = ret.dropna(subset=["return"])

    # Map ticker to asset_id
    assets_df = price_df[["asset_id", "ticker"]].drop_duplicates(subset=["ticker"])
    ret = ret.merge(assets_df, on="ticker", how="left")

    # Ensure date is a date (no time)
    ret["date"] = pd.to_datetime(ret["date"]).dt.date

    # Keep only necessary columns and order them
    ret = ret[["asset_id", "date", "return"]]

    return ret

def upsert_returns(engine, returns_df: pd.DataFrame):
    """
    Upsert returns into the `returns` table using INSERT ... ON DUPLICATE KEY UPDATE.
    Expects returns_df with columns: asset_id, date (datetime.date), return (float).
    """
    if returns_df is None or returns_df.empty:
        print("No returns to upsert.")
        return 0

    # Prepare rows: (asset_id, date_iso, return)
    rows = []
    for _, row in returns_df.iterrows():
        asset_id = int(row["asset_id"])
        date_val = row["date"]
        date_iso = date_val.isoformat() if hasattr(date_val, "isoformat") else str(date_val)
        try:
            ret_val = float(row["return"])
        except Exception:
            ret_val = float(pd.to_numeric(row["return"], errors="coerce"))

        rows.append((asset_id, date_iso, ret_val))

    sql = """
    INSERT INTO returns (asset_id, date, `return`)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE `return` = VALUES(`return`)
    """

    # Execute batch
    with engine.begin() as conn:
        raw = conn.connection
        cur = raw.cursor()
        try:
            cur.executemany(sql, rows)
            raw.commit()
        finally:
            cur.close()

    return len(rows)

def compute_and_store_returns():
    # Ensure engine is set
    engine = get_engine()

    print("Loading price table from DB...")
    price_df = load_price_table(engine)
    if price_df.empty:
        print("No price data found in historical_prices. Exiting.")
        return

    print("Computing daily returns...")
    returns_df = compute_daily_returns_df(price_df)
    if returns_df.empty:
        print("No returns computed (maybe not enough data). Exiting.")
        return

    print(f"Prepared {len(returns_df)} return rows to upsert.")
    n = upsert_returns(engine, returns_df)
    print(f"Inserted/updated {n} rows into `returns` table.")

if __name__ == "__main__":
    # initialize engine
    setup_engine()
    compute_and_store_returns()
