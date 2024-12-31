import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib

def run_analysis(data):
    # Calculate technical indicators
    data['RSI'] = talib.RSI(data['Close'])
    data['MACD'], data['MACD_Signal'], _ = talib.MACD(data['Close'])
    data['MA_20'] = talib.SMA(data['Close'], timeperiod=20)
    data['MA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
    data['Bollinger_Upper'], data['Bollinger_Middle'], data['Bollinger_Lower'] = talib.BBANDS(data['Close'])
    
    return data

def plot_price_and_indicators(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], name='20-day MA', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], name='50-day MA', line=dict(color='red')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Upper'], name='Upper BB', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Lower'], name='Lower BB', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(title='Price and Technical Indicators', xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    return fig

def plot_volume_analysis(data):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'))
    
    fig.update_layout(title='Trading Volume Over Time', xaxis_title='Date', yaxis_title='Volume')
    
    return fig

def analyze_correlations(data):
    correlation_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MA_20', 'MA_50', 'ATR']].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(title='Correlation Heatmap of Features')
    
    return fig

