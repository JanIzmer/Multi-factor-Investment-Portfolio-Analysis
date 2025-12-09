import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from sqlalchemy import text

import sys
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.append(str(root))

from database.src.connection import setup_engine, get_engine
from src.risk.covariance import load_returns, compute_sample_cov, compute_expected_returns
from src.risk.optimization import min_volatility, max_sharpe, portfolio_performance
from src.utils import plot_correlation_heatmap  # optional fallback

# initialize engine (reads .env)
setup_engine()

st.set_page_config(layout="wide", page_title="Portfolio Analysis Dashboard")

st.title("Multi-Asset Portfolio Dashboard (Sample covariance)")
st.markdown("""
Interactive analytics for portfolio construction:
- compute sample covariance and expected returns  
- build Min Volatility / Max Sharpe portfolios  
- visualize correlation matrix, efficient frontier, weights and cumulative returns
""")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    all_returns = load_returns()  # cached function below will be used instead in heavy apps
    if all_returns is None or all_returns.empty:
        st.error("No returns found in DB. Run ETL and returns scripts first.")
        st.stop()

    default_tickers = list(all_returns.columns)
    tickers = st.multiselect("Choose tickers (subset)", options=default_tickers, default=default_tickers[:8])

    col1, col2 = st.columns(2)
    start_date = st.date_input("Start date", value=(all_returns.index.min()) if not all_returns.empty else date(2018,1,1))
    end_date = st.date_input("End date", value=(all_returns.index.max()) if not all_returns.empty else date.today())

    method = st.selectbox("Optimization method", options=["Min Volatility", "Max Sharpe"])
    max_weight = st.slider("Max weight per asset (for diversification)", 0.05, 1.0, 0.6, step=0.05)

    # Portfolio save name (sidebar)
    portfolio_name = st.text_input("Portfolio name (for saving)", value=f"{method}_{date.today().isoformat()}")

    # re-run / refresh controls
    st.markdown("---")
    st.caption("Tip: reduce tickers or increase start date for faster results.")

# Filter returns by date and tickers (safe slicing)
@st.cache_data(ttl=600)
def get_filtered_returns(tickers, start_date, end_date):
    df = load_returns()
    if df is None or df.empty:
        return pd.DataFrame()
    # ensure index is datetime
    df.index = pd.to_datetime(df.index)
    mask = (df.index.date >= pd.to_datetime(start_date).date()) & (df.index.date <= pd.to_datetime(end_date).date())
    filtered = df.loc[mask, tickers].dropna(how="all", axis=0)
    return filtered

returns_df = get_filtered_returns(tickers, start_date, end_date)
if returns_df.empty:
    st.warning("No returns in the selected window / tickers. Adjust filters.")
    st.stop()

st.subheader("Data summary")
c1, c2, c3 = st.columns(3)
c1.metric("Tickers", len(returns_df.columns))
c2.metric("Days", returns_df.shape[0])
c3.metric("Start → End", f"{returns_df.index.min().date()} → {returns_df.index.max().date()}")

# Compute cov and expected returns (cached)
@st.cache_data(ttl=600)
def compute_risk_metrics(returns_df):
    cov = compute_sample_cov(returns_df)            
    mu = compute_expected_returns(returns_df)       
    corr = cov.corr()
    return cov, mu, corr

cov_matrix, mu, corr_matrix = compute_risk_metrics(returns_df)

# Left: correlation heatmap
st.subheader("Correlation matrix")
fig_corr = px.imshow(corr_matrix, 
                     labels=dict(x="Asset", y="Asset", color="Correlation"),
                     x=corr_matrix.columns, y=corr_matrix.index,
                     color_continuous_scale="RdBu", zmin=-1, zmax=1)
fig_corr.update_layout(height=600)
st.plotly_chart(fig_corr, use_container_width=True)

# Utility: performance metrics
def compute_perf_metrics(ts_returns, rf=0.02):
    """
    ts_returns: pd.Series of daily returns (not cumulative)
    returns dict with CAGR, AnnVol, AnnRet, Sharpe, MaxDD
    """
    if ts_returns.empty:
        return {"CAGR": np.nan, "AnnVol": np.nan, "AnnRet": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    ts = ts_returns.dropna()
    cum = (1 + ts).cumprod()
    days = (cum.index[-1] - cum.index[0]).days
    total_years = max(days / 365.25, 1/252)  # avoid zero division
    total_return = cum.iloc[-1]
    cagr = total_return ** (1/total_years) - 1
    ann_vol = ts.std() * np.sqrt(252)
    ann_ret = ts.mean() * 252
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1
    max_dd = drawdown.min()
    return {"CAGR": cagr, "AnnVol": ann_vol, "AnnRet": ann_ret, "Sharpe": sharpe, "MaxDD": max_dd}

# Optimize portfolio
st.subheader("Optimization")
tickers_list = list(returns_df.columns)
mean_returns = mu.loc[tickers_list].values
cov_sub = cov_matrix.loc[tickers_list, tickers_list].values

# Optimization functions that respect max weight
def min_vol_with_maxweight(mean_returns, cov, max_w):
    n = len(mean_returns)
    bounds = tuple((0, float(max_w)) for _ in range(n))
    # modify min_volatility to accept bounds
    from scipy.optimize import minimize
    def vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov, w)))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    x0 = np.array([1.0 / n] * n)
    res = minimize(vol, x0=x0, bounds=bounds, constraints=constraints)
    if not res.success:
        st.warning("Min vol optimization did not converge, returning equal weights")
        return np.array([1.0/n]*n)
    return res.x

def max_sharpe_with_maxweight(mean_returns, cov, max_w, risk_free=0.02):
    n = len(mean_returns)
    bounds = tuple((0, float(max_w)) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    from scipy.optimize import minimize
    def neg_sharpe(w):
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return - (ret - risk_free) / vol if vol>0 else 1e6
    x0 = np.array([1.0 / n] * n)
    res = minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=constraints)
    if not res.success:
        st.warning("MaxSharpe optimization did not converge, returning equal weights")
        return np.array([1.0/n]*n)
    return res.x

# Run optimization with spinner
with st.spinner("Optimizing portfolio..."):
    if method == "Min Volatility":
        weights = min_vol_with_maxweight(mean_returns, cov_sub, max_weight)
    else:
        weights = max_sharpe_with_maxweight(mean_returns, cov_sub, max_weight)

# Show weights
weights_df = pd.DataFrame({
    "ticker": tickers_list,
    "weight": np.round(weights, 6)   # round for nicer display
})

st.subheader("Portfolio weights")
col1, col2 = st.columns([2,1])

with col1:
    fig_w = px.bar(weights_df, x="ticker", y="weight", labels={"weight": "Weight", "ticker": "Ticker"})
    fig_w.update_layout(title_text="Optimized weights", yaxis_title="Weight")
    st.plotly_chart(fig_w, use_container_width=True)

    # pie chart of composition
    fig_pie = px.pie(weights_df[weights_df['weight']!=0], names='ticker', values='weight', title='Portfolio composition')
    st.plotly_chart(fig_pie, use_container_width=True)

    # download button for weights
    csv = weights_df.to_csv(index=False)
    st.download_button("Download weights CSV", data=csv, file_name="portfolio_weights.csv", mime="text/csv")

with col2:
    # show a small table with extra metrics
    metrics_df = pd.DataFrame({
        "ticker": tickers_list,
        "weight": np.round(weights, 6),
        "exp_return_ann": np.round(mean_returns,6),
        "exp_vol_ann": np.round(np.sqrt(np.diag(cov_sub)),6)
    })
    metrics_df = metrics_df.sort_values("weight", ascending=False).reset_index(drop=True)
    # nicer formatting
    metrics_df_display = metrics_df.copy()
    metrics_df_display["weight"] = metrics_df_display["weight"].map("{:.2%}".format)
    metrics_df_display["exp_return_ann"] = metrics_df_display["exp_return_ann"].map("{:.2%}".format)
    metrics_df_display["exp_vol_ann"] = metrics_df_display["exp_vol_ann"].map("{:.2%}".format)
    st.table(metrics_df_display.head(20))

# Compute portfolio performance
port_ret, port_vol = portfolio_performance(weights, mean_returns, cov_sub)
st.metric("Portfolio expected annual return", f"{port_ret:.2%}")
st.metric("Portfolio expected annual volatility", f"{port_vol:.2%}")
st.metric("Approx Sharpe (rf=2%)", f"{(port_ret-0.02)/port_vol:.2f}")

# Compute historical cumulative returns (using the selected weights on historical returns)
st.subheader("Cumulative returns (backtest with historical weights)")
weights_array = weights_df['weight'].values
portfolio_ts = (returns_df.fillna(0) * weights_array).sum(axis=1)
cum = (1 + portfolio_ts).cumprod()
fig = px.line(cum, labels={"index":"Date", 0:"Cumulative Return"})
fig.update_layout(title="Portfolio cumulative returns (using historical weights)")
st.plotly_chart(fig, use_container_width=True)

# Display performance metrics computed from historical portfolio returns
perf = compute_perf_metrics(portfolio_ts)
perf_df = pd.Series(perf).rename_axis("metric").reset_index()
perf_df.columns = ["metric", "value"]
perf_df["value_str"] = perf_df["value"].apply(lambda x: f"{x:.2%}" if np.isfinite(x) else "N/A")
st.write("Performance (historical)")
st.table(perf_df[["metric", "value_str"]])

# Efficient frontier (approx) — sample many target returns and minimize vol
st.subheader("Efficient frontier (approx)")
def efficient_frontier(mean_returns, cov, points=50):
    n = len(mean_returns)
    # bounds & constraint sum=1
    bounds = tuple((0, max_weight) for _ in range(n))
    results = []
    import scipy.optimize as sco
    # target space extended a bit
    min_tr = np.nanmin(mean_returns)
    max_tr = np.nanmax(mean_returns)
    target_returns = np.linspace(min_tr*0.8 if not np.isnan(min_tr) else 0, max_tr*1.2 if not np.isnan(max_tr) else 0.1, points)
    for tr in target_returns:
        # minimize volatility subject to expected return >= tr
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w)-1},
            {'type': 'ineq', 'fun': lambda w, tr=tr: np.dot(w, mean_returns) - tr}
        )
        x0 = np.array([1.0/n]*n)
        fun = lambda w: np.sqrt(np.dot(w.T, np.dot(cov, w)))
        res = sco.minimize(fun, x0=x0, bounds=bounds, constraints=cons)
        if res.success:
            r = np.dot(res.x, mean_returns)
            vol = fun(res.x)
            results.append((vol, r))
    if not results:
        return [], []
    res = np.array(results)
    # sort by risk
    idx = np.argsort(res[:,0])
    return res[idx,0].tolist(), res[idx,1].tolist()

risks, rets = efficient_frontier(mean_returns, cov_sub, points=40)
fig = go.Figure()
if risks:
    fig.add_trace(go.Scatter(x=risks, y=rets, mode='lines', name='Efficient frontier'))
fig.add_trace(go.Scatter(x=[port_vol], y=[port_ret], mode='markers', name='Selected portfolio', marker=dict(size=12)))
fig.update_layout(xaxis_title='Volatility', yaxis_title='Expected return')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Save portfolio button (writes to DB)
def save_portfolio_to_db(name, tickers, weights):
    engine = get_engine()
    with engine.begin() as conn:
        # insert meta row using sqlalchemy.text
        conn.execute(text("INSERT INTO portfolio_weights (name) VALUES (:name)"), {"name": name})
        # get last insert id (works in MySQL)
        portfolio_id = conn.execute(text("SELECT LAST_INSERT_ID()")).fetchone()[0]

        # prepare rows (portfolio_id, asset_id, weight)
        rows = []
        for t, w in zip(tickers, weights):
            aid_row = conn.execute(text("SELECT asset_id FROM assets WHERE ticker = :t"), {"t": t}).fetchone()
            if aid_row:
                asset_id = int(aid_row[0])
                rows.append((portfolio_id, asset_id, float(w)))
        if rows:
            raw = conn.connection 
            cur = raw.cursor()
            try:
                cur.executemany(
                    "INSERT INTO portfolio_weight_rows (portfolio_id, asset_id, weight) VALUES (%s, %s, %s)",
                    rows
                )
                raw.commit()
            finally:
                cur.close()

    return portfolio_id

st.write("## Actions")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Save portfolio to DB"):
        if not portfolio_name:
            st.error("Provide a portfolio name before saving.")
        else:
            with st.spinner("Saving portfolio..."):
                try:
                    pid = save_portfolio_to_db(portfolio_name, tickers_list, weights)
                    st.success(f"Saved portfolio id {pid}")
                except Exception as e:
                    st.error(f"Failed to save portfolio: {e}")




