import pandas as pd
import numpy as np

def create_custom_indicator(data):
    # Example: Custom Momentum Indicator
    # This indicator combines RSI and MACD for a custom momentum signal
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # Combine RSI and MACD for custom indicator
    custom_momentum = (rsi - 50) / 50 + (macd - signal) / data['Close'].std()
    
    return custom_momentum

