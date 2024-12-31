import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import talib

def prepare_data(data, window_size=30):
    # Calculate technical indicators
    data['RSI'] = talib.RSI(data['Close'])
    data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'])
    data['MA_20'] = talib.SMA(data['Close'], timeperiod=20)
    data['MA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
    data['Bollinger_Upper'], data['Bollinger_Middle'], data['Bollinger_Lower'] = talib.BBANDS(data['Close'])
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
    data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'])
    data['MOM'] = talib.MOM(data['Close'])
    
    # Create features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'MA_20', 'MA_50', 'ATR', 'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower', 'OBV', 'ADX', 'CCI', 'MOM']
    X = data[features].values
    y = data[['Close']].shift(-1).values[:-1]  # Predict next day's close
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    for i in range(len(X_scaled) - window_size):
        X_sequences.append(X_scaled[i:i+window_size])
        y_sequences.append(y_scaled[i+window_size])
    
    return np.array(X_sequences), np.array(y_sequences), scaler

def add_sentiment_features(X, sentiment_scores):
    # Ensure sentiment_scores align with X
    aligned_scores = sentiment_scores[-len(X):]
    
    # Calculate moving averages of sentiment scores
    sentiment_ma_3 = np.convolve(aligned_scores, np.ones(3), 'valid') / 3
    sentiment_ma_7 = np.convolve(aligned_scores, np.ones(7), 'valid') / 7
    
    # Pad the moving averages to match X's length
    pad_3 = np.full(2, np.nan)
    pad_7 = np.full(6, np.nan)
    sentiment_ma_3 = np.concatenate((pad_3, sentiment_ma_3))
    sentiment_ma_7 = np.concatenate((pad_7, sentiment_ma_7))
    
    # Add sentiment features to X
    sentiment_features = np.column_stack((aligned_scores, sentiment_ma_3, sentiment_ma_7))
    X_with_sentiment = np.concatenate((X, sentiment_features.reshape(X.shape[0], X.shape[1], -1)), axis=2)
    
    return X_with_sentiment

def add_fundamental_features(symbol):
    stock = yf.Ticker(symbol)
    
    # Get financial data
    balance_sheet = stock.balance_sheet
    income_stmt = stock.financials
    cash_flow = stock.cashflow
    
    # Calculate fundamental ratios
    pe_ratio = stock.info['trailingPE'] if 'trailingPE' in stock.info else np.nan
    pb_ratio = stock.info['priceToBook'] if 'priceToBook' in stock.info else np.nan
    debt_to_equity = balance_sheet.loc['Total Liab'].iloc[0] / balance_sheet.loc['Total Stockholder Equity'].iloc[0]
    roa = income_stmt.loc['Net Income'].iloc[0] / balance_sheet.loc['Total Assets'].iloc[0]
    roe = income_stmt.loc['Net Income'].iloc[0] / balance_sheet.loc['Total Stockholder Equity'].iloc[0]
    
    # Create fundamental features array
    fundamental_features = np.array([pe_ratio, pb_ratio, debt_to_equity, roa, roe])
    
    # Repeat the fundamental features for each time step in X
    return np.tile(fundamental_features, (X.shape[0], X.shape[1], 1))

