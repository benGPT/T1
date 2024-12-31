from celery_config import app
from model_retraining import retrain_model
from data_preparation import prepare_data
from sentiment_analysis import analyze_sentiment
from alternative_data import fetch_alternative_data
from fundamental_analysis import get_fundamental_data
from risk_management import optimize_portfolio
import yfinance as yf
import pandas as pd

@app.task
def retrain_model_task(symbol, days=30):
    data = yf.download(symbol, period=f"{days}d")
    X, y, scaler = prepare_data(data)
    sentiment_scores = analyze_sentiment(symbol, "1d")
    retrain_model(symbol, X, y, scaler, sentiment_scores)
    print(f"Model for {symbol} has been retrained.")

@app.task
def update_data_task(symbol):
    data = yf.download(symbol, period="1d")
    # Process and store the updated data
    print(f"Updated data for {symbol}")
    return data.to_dict()

@app.task
def analyze_sentiment_task(symbol):
    sentiment = analyze_sentiment(symbol, "1d")
    print(f"Analyzed sentiment for {symbol}: {sentiment.mean()}")
    return sentiment.to_dict()

@app.task
def fetch_alternative_data_task(symbol):
    alt_data = fetch_alternative_data(symbol)
    print(f"Fetched alternative data for {symbol}")
    return alt_data.to_dict()

@app.task
def get_fundamental_data_task(symbol):
    fundamental_data = get_fundamental_data(symbol)
    print(f"Fetched fundamental data for {symbol}")
    return fundamental_data

@app.task
def optimize_portfolio_task(holdings):
    optimized_weights = optimize_portfolio(holdings)
    print("Portfolio optimization completed")
    return optimized_weights

@app.task
def update_all_data():
    from app import INSTRUMENTS
    for instrument_type in INSTRUMENTS:
        for symbol in INSTRUMENTS[instrument_type]:
            update_data_task.delay(symbol)
            analyze_sentiment_task.delay(symbol)
            fetch_alternative_data_task.delay(symbol)
            get_fundamental_data_task.delay(symbol)
    print("All data updates initiated")

@app.task
def daily_model_retraining():
    from app import INSTRUMENTS
    for instrument_type in INSTRUMENTS:
        for symbol in INSTRUMENTS[instrument_type]:
            retrain_model_task.delay(symbol)
    print("Daily model retraining initiated")

