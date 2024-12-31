import yfinance as yf
import pandas as pd
import numpy as np

def incorporate_external_factors(symbol):
    # Fetch economic indicators
    gdp_growth = yf.Ticker("GDP").history(period="1y")['Close'].pct_change().iloc[-1]
    unemployment_rate = yf.Ticker("UNRATE").history(period="1y")['Close'].iloc[-1]
    inflation_rate = yf.Ticker("CPI").history(period="1y")['Close'].pct_change().iloc[-1]
    
    # Fetch market sentiment indicators
    vix = yf.Ticker("^VIX").history(period="1y")['Close'].iloc[-1]
    put_call_ratio = yf.Ticker("^PCR").history(period="1y")['Close'].iloc[-1]
    
    # Create external factors array
    external_factors = np.array([gdp_growth, unemployment_rate, inflation_rate, vix, put_call_ratio])
    
    # Repeat the external factors for each time step in X
    return np.tile(external_factors, (X.shape[0], X.shape[1], 1))

