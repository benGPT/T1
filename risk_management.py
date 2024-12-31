import numpy as np
from scipy.optimize import minimize
import yfinance as yf

def calculate_position_size(balance, risk_per_trade, current_price):
    risk_amount = balance * risk_per_trade
    position_size = risk_amount / current_price
    return position_size

def apply_stop_loss_take_profit(position, balance, current_price, risk_per_trade):
    stop_loss = position['price'] * (1 - risk_per_trade * 2)
    take_profit = position['price'] * (1 + risk_per_trade * 3)
    
    if current_price <= stop_loss or current_price >= take_profit:
        balance += position['amount'] * current_price
        position['amount'] = 0
    
    return position, balance

def optimize_portfolio(holdings):
    def portfolio_volatility(weights, returns):
        return np.sqrt(np.dot(weights.T, np.dot(np.cov(returns, rowvar=False), weights)))
    
    def optimize_func(weights, returns):
        return portfolio_volatility(weights, returns)
    
    assets = list(holdings.keys())
    prices = [yf.Ticker(asset).history(period="1y")['Close'] for asset in assets]
    returns = np.log(prices / prices.shift(1))
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(assets)))
    initial_weights = [1/len(assets)] * len(assets)
    
    optimized = minimize(optimize_func, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    
    return dict(zip(assets, optimized.x))

