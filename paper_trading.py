import yfinance as yf
import pandas as pd
import numpy as np

class PaperTradingEnvironment:
    def __init__(self, symbol, timeframe, initial_balance):
        self.symbol = symbol
        self.timeframe = timeframe
        self.balance = initial_balance
        self.positions = {}
        self.trades = []

    def get_latest_data(self):
        data = yf.download(self.symbol, period="1d", interval=self.timeframe)
        return data

    def execute_trade(self, action, risk_per_trade):
        data = self.get_latest_data()
        current_price = data['Close'].iloc[-1]
        
        if action > 0:  # Buy
            amount = min(self.balance, self.balance * risk_per_trade) / current_price
            self.positions[self.symbol] = self.positions.get(self.symbol, 0) + amount
            self.balance -= amount * current_price
            self.trades.append({"type": "buy", "amount": amount, "price": current_price})
            return f"Bought {amount} {self.symbol} at {current_price}"
        elif action < 0:  # Sell
            amount = min(self.positions.get(self.symbol, 0), abs(action))
            self.positions[self.symbol] = self.positions.get(self.symbol, 0) - amount
            self.balance += amount * current_price
            self.trades.append({"type": "sell", "amount": amount, "price": current_price})
            return f"Sold {amount} {self.symbol} at {current_price}"
        else:
            return "No trade executed"

    def get_portfolio_value(self):
        data = self.get_latest_data()
        current_price = data['Close'].iloc[-1]
        portfolio_value = self.balance
        for symbol, amount in self.positions.items():
            portfolio_value += amount * current_price
        return portfolio_value

