# src/utils.py
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_heatmap(cov_matrix):
    corr = cov_matrix.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask)
    plt.show()

def plot_efficient_frontier(returns_list, risks_list, weights_list, optimal_weights=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risks_list, y=returns_list, mode='lines', name='Efficient frontier'))
    if optimal_weights is not None:
        opt_ret, opt_vol = optimal_weights
        fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers', name='Optimized portfolio'))
    fig.update_layout(xaxis_title='Volatility', yaxis_title='Expected return')
    fig.show()

def plot_weights_bar(weights, tickers):
    fig = go.Figure([go.Bar(x=tickers, y=weights)])
    fig.update_layout(title='Portfolio Weights', xaxis_title='Asset', yaxis_title='Weight')
    fig.show()
