# database/src/etl_fetch_prices.py
from datetime import datetime
import os
import pandas as pd
import yfinance as yf
from sqlalchemy import text
from sqlalchemy.dialects.mysql import insert as mysql_insert

from database.src.connection import setup_engine, get_engine

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "JPM", "BAC", "SPY", "VTI"
]
START_DATE = os.getenv("PRICE_START_DATE", "2015-01-01")
END_DATE = os.getenv("PRICE_END_DATE", datetime.today().strftime("%Y-%m-%d"))
# ------------------------------------------------

engine = setup_engine()


def ensure_assets_exist(conn, tickers):
    """
    Ensure that all tickers exist in the `assets` table.
    If a ticker is missing, it will be inserted.

    Returns:
        dict: mapping ticker -> asset_id
    """
    existing = conn.execute(text("SELECT asset_id, ticker FROM assets")).fetchall()
    ticker_to_id = {row[1]: row[0] for row in existing}

    for t in tickers:
        if t not in ticker_to_id:
            conn.execute(
                text("INSERT INTO assets (ticker, name, sector) VALUES (:t, :name, :sector)"),
                {"t": t, "name": t, "sector": None}
            )

    rows = conn.execute(text("SELECT asset_id, ticker FROM assets")).fetchall()
    return {row[1]: row[0] for row in rows}


def fetch_prices_for_ticker(ticker, start, end):
    """
    Download price data for a single ticker via yfinance.

    Works with both single-ticker and multi-ticker yfinance responses.
    Returns DataFrame with columns: date (datetime.date), close (float), volume (int/None).
    """
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        print(f"  no data for {ticker}")
        return pd.DataFrame(columns=["date", "close", "volume"])


    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.levels[1]:
            df = df.xs(ticker, axis=1, level=1)  
        elif ticker in df.columns.levels[0]:
            df = df.xs(ticker, axis=1, level=0)
        else:
 
            flat = ["_".join([str(c) for c in col]).strip() for col in df.columns.values]
            df.columns = flat

    else:
        pass

    # Normalize column names to lowercase strings
    df = df.reset_index()
    # Some dataframes may already have 'Date' as column; ensure normalization works
    df.columns = [str(c).lower() for c in df.columns]

    # Find the close column: prefer 'adj close' then 'close'
    close_col = None
    for candidate in ("adj close", "adj_close", "adjclose", "close"):
        matches = [c for c in df.columns if candidate in c]
        if matches:
            # choose the best match (exact 'adj close' or first match)
            exact = [c for c in matches if c == candidate]
            close_col = exact[0] if exact else matches[0]
            break

    # Find volume column
    vol_col = None
    for candidate in ("volume", "vol"):
        matches = [c for c in df.columns if candidate in c]
        if matches:
            vol_col = matches[0]
            break

    if close_col is None:
        raise ValueError(f"{ticker}: no 'Close' or 'Adj Close' column found. Columns: {df.columns.tolist()}")

    if vol_col is None:
        df["volume_tmp"] = None
        vol_col = "volume_tmp"

    df = df.rename(columns={close_col: "close", vol_col: "volume"})
    df = df[["date", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def upsert_prices(conn, asset_id, df_prices):
    """
    Insert or update price records into `historical_prices`.

    Uses MySQL INSERT ... ON DUPLICATE KEY UPDATE with executemany for speed.
    Requires UNIQUE(asset_id, date).
    Returns number of rows attempted (not strictly number changed).
    """
    import math

    if df_prices.empty:
        return 0

    values = []
    for row in df_prices.itertuples(index=False):
        dt = row.date
        if hasattr(dt, "isoformat"):
            dt = dt.isoformat()
        else:
            dt = str(dt)
        # safe close (float)
        close_val = row.close
        if isinstance(close_val, (list, tuple)) or hasattr(close_val, "item"):
            # defensive: extract scalar
            try:
                close_val = float(close_val.item())
            except Exception:
                close_val = float(close_val[0])
        else:
            close_val = float(close_val)
        # safe volume 
        vol_val = row.volume
        if vol_val is None:
            vol_sql = None
        else:
            try:
                if isinstance(vol_val, float) and math.isnan(vol_val):
                    vol_sql = None
                else:
                    vol_sql = int(vol_val)
            except Exception:
                # fallback: try to coerce
                try:
                    vol_sql = int(vol_val.item())
                except Exception:
                    vol_sql = None

        values.append((dt, int(asset_id), close_val, vol_sql))

    if not values:
        return 0

    sql = """
    INSERT INTO historical_prices (`date`, asset_id, `close`, volume)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        `close` = VALUES(`close`),
        volume = VALUES(volume)
    """

    raw_conn = conn.connection
    cursor = raw_conn.cursor()
    try:
        cursor.executemany(sql, values)
        raw_conn.commit()
    finally:
        cursor.close()

    return len(values)



def main(tickers=TICKERS, start=START_DATE, end=END_DATE):
    engine = get_engine()

    # Step 1 — ensure all assets exist
    with engine.begin() as conn:
        ticker_id_map = ensure_assets_exist(conn, tickers)

    # Step 2 — download prices and upsert
    total = 0
    with engine.begin() as conn:
        for t in tickers:
            print(f"Fetching {t} from {start} to {end} ...")
            df = fetch_prices_for_ticker(t, start, end)

            if df.empty:
                print(f"  No data for {t}")
                continue

            asset_id = ticker_id_map.get(t)

            if asset_id is None:
                conn.execute(
                    text("INSERT INTO assets (ticker, name) VALUES (:t, :n)"),
                    {"t": t, "n": t}
                )
                asset_id = conn.execute(
                    text("SELECT asset_id FROM assets WHERE ticker=:t"),
                    {"t": t}
                ).fetchone()[0]
                ticker_id_map[t] = asset_id

            inserted = upsert_prices(conn, asset_id, df)
            total += inserted
            print(f"  Upserted {inserted} rows for {t}")

    print(f"Done. Total upserted rows: {total}")


if __name__ == "__main__":
    main()
