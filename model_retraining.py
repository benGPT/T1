import yfinance as yf
from data_preparation import prepare_data, add_sentiment_features, add_fundamental_features
from external_factors import incorporate_external_factors
from alternative_data import fetch_alternative_data
from adaptive_model import AdaptiveModel
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_environment import TradingEnvironment
import numpy as np
from sentiment_analysis import analyze_sentiment

def retrain_model(symbol, days=30):
    # Fetch new data
    data = yf.download(symbol, period=f"{days}d")
    
    # Prepare data
    X, y, scaler = prepare_data(data)
    sentiment_scores = analyze_sentiment(symbol, "1d")
    fundamental_data = add_fundamental_features(symbol)
    external_factors = incorporate_external_factors(symbol)
    alternative_data = fetch_alternative_data(symbol)
    
    X = add_sentiment_features(X, sentiment_scores)
    X = np.concatenate([X, fundamental_data, external_factors, alternative_data], axis=2)
    
    # Create environment
    env = DummyVecEnv([lambda: TradingEnvironment(X, y, scaler)])
    
    # Load existing model
    model = AdaptiveModel.load(f"trained_model_{symbol}")
    
    # Retrain model
    model.learn(total_timesteps=10000)
    
    # Save updated model
    model.save(f"trained_model_{symbol}")
    
    print(f"Model for {symbol} has been retrained and updated.")

