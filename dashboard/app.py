# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

import sys
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.append(str(root))

from database.src.connection import setup_engine, get_engine
from src.risk.covariance import load_returns, compute_sample_cov, compute_expected_returns
from src.risk.optimization import min_volatility, max_sharpe, portfolio_performance
from src.utils import plot_correlation_heatmap  # optional fallback
# We'll create some plot code in-place to keep the app self-contained

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
c2.metric("Rows (days)", returns_df.shape[0])
c3.metric("Start → End", f"{returns_df.index.min().date()} → {returns_df.index.max().date()}")

# Compute cov and expected returns (cached)
@st.cache_data(ttl=600)
def compute_risk_metrics(returns_df):
    cov = compute_sample_cov(returns_df)            # annualized cov
    mu = compute_expected_returns(returns_df)       # annualized mean
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

with col2:
    # show a small table with extra metrics
    metrics_df = pd.DataFrame({
        "ticker": tickers_list,
        "weight": np.round(weights, 6),
        "exp_return_ann": np.round(mean_returns,6),
        "exp_vol_ann": np.round(np.sqrt(np.diag(cov_sub)),6)
    })
    metrics_df = metrics_df.sort_values("weight", ascending=False).reset_index(drop=True)
    st.table(metrics_df.head(20))

# Compute portfolio performance
port_ret, port_vol = portfolio_performance(weights, mean_returns, cov_sub)
st.metric("Portfolio expected annual return", f"{port_ret:.2%}")
st.metric("Portfolio expected annual volatility", f"{port_vol:.2%}")
st.metric("Approx Sharpe (rf=2%)", f"{(port_ret-0.02)/port_vol:.2f}")

# Efficient frontier (approx) — sample many target returns and minimize vol
st.subheader("Efficient frontier (approx)")
def efficient_frontier(mean_returns, cov, points=50):
    n = len(mean_returns)
    # bounds & constraint sum=1
    bounds = tuple((0, max_weight) for _ in range(n))
    results = []
    import scipy.optimize as sco
    target_returns = np.linspace(min(mean_returns)*0.8, max(mean_returns)*1.2, points)
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
    return res[:,0].tolist(), res[:,1].tolist()

risks, rets = efficient_frontier(mean_returns, cov_sub, points=40)
fig = go.Figure()
if risks:
    fig.add_trace(go.Scatter(x=risks, y=rets, mode='lines', name='Efficient frontier'))
fig.add_trace(go.Scatter(x=[port_vol], y=[port_ret], mode='markers', name='Selected portfolio', marker=dict(size=12)))
fig.update_layout(xaxis_title='Volatility', yaxis_title='Expected return')
st.plotly_chart(fig, use_container_width=True)

# Cumulative returns simulation for the portfolio
st.subheader("Cumulative returns (backtest with historical weights)")
# Using historical returns_df to compute cumulative returns for current weights
weights_array = weights_df['weight'].values
portfolio_ts = (returns_df.fillna(0) * weights_array).sum(axis=1)
cum = (1 + portfolio_ts).cumprod()
fig = px.line(cum, labels={"index":"Date", 0:"Cumulative Return"})
fig.update_layout(title="Portfolio cumulative returns (using historical logics)")
st.plotly_chart(fig, use_container_width=True)


