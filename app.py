import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.model_selection import train_test_split
import optuna
import ccxt
import asyncio
import websockets
import json
from datetime import datetime, timedelta
import time

from data_preparation import prepare_data, add_sentiment_features, add_fundamental_features
from trading_environment import TradingEnvironment
from backtesting import backtest
from model_evaluation import calculate_metrics
from data_analysis import run_analysis, plot_price_and_indicators, plot_volume_analysis, analyze_correlations
from sentiment_analysis import analyze_sentiment
from risk_management import calculate_position_size, apply_stop_loss_take_profit, optimize_portfolio
from custom_indicators import create_custom_indicator
from ensemble_model import EnsembleModel
from explainability import explain_model_decision
from adaptive_model import AdaptiveModel
from external_factors import incorporate_external_factors
from user_auth import authenticate_user, create_user, get_user_portfolio, update_user_portfolio
from error_handling import handle_error
from advanced_portfolio import calculate_risk_adjusted_returns
from order_types import place_advanced_order
from paper_trading import PaperTradingEnvironment
from educational_resources import get_educational_content
from model_retraining import retrain_model
from alternative_data import fetch_alternative_data
from fundamental_analysis import get_fundamental_data

# Import new model files
from qnn_model import train_qnn_model, predict_qnn
from gnn_model import train_gnn_model, predict_gnn
from tensorflow_model import train_tensorflow_model, predict_tensorflow

from celery_tasks import (
    retrain_model_task, update_data_task, analyze_sentiment_task,
    fetch_alternative_data_task, get_fundamental_data_task,
    optimize_portfolio_task, update_all_data, daily_model_retraining
)

st.set_page_config(layout="wide", page_title="Advanced Trading App")

# User Authentication
if 'user' not in st.session_state:
    st.session_state.user = None

def login():
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user = authenticate_user(username, password)
        if user:
            st.session_state.user = user
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password")

def signup():
    username = st.sidebar.text_input("New Username")
    password = st.sidebar.text_input("New Password", type="password")
    if st.sidebar.button("Sign Up"):
        if create_user(username, password):
            st.success("User created successfully! Please log in.")
        else:
            st.error("Username already exists")

if not st.session_state.user:
    st.sidebar.title("Login / Sign Up")
    login()
    st.sidebar.markdown("---")
    signup()
else:
    st.sidebar.title(f"Welcome, {st.session_state.user['username']}!")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.experimental_rerun()

# Main Application (only accessible if logged in)
if st.session_state.user:
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Backtesting", "Live Trading", "Paper Trading", "Portfolio", "QNN Model", "GNN Model", "TensorFlow Model", "Educational Resources"])

    # Financial instruments
    INSTRUMENTS = {
        "Cryptocurrencies": ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD", "BCH/USD", "ADA/USD", "DOT/USD", "LINK/USD", "XLM/USD", "DOGE/USD"],
        "Forex": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP"],
        "Indices": ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^GDAXI"],
        "Stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "FB", "TSLA", "NVDA", "JPM", "V", "PEP"]
    }

    # Home page
    if page == "Home":
        st.title("Advanced Trading Application")
        st.write("Welcome to the Advanced Trading Application. Use the sidebar to navigate through different sections.")
        
        st.subheader("Select Financial Instrument")
        instrument_type = st.selectbox("Instrument Type", list(INSTRUMENTS.keys()))
        symbol = st.selectbox("Symbol", INSTRUMENTS[instrument_type])
        
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m", "1m"])
        
        if st.button("Fetch Data"):
            try:
                if instrument_type == "Cryptocurrencies":
                    exchange = ccxt.binance()
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
                    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                    data.set_index('timestamp', inplace=True)
                else:
                    data = yf.download(symbol, period="1y", interval=timeframe)
                
                st.write(data.tail())
                
                fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                     open=data['Open'],
                                                     high=data['High'],
                                                     low=data['Low'],
                                                     close=data['Close'])])
                st.plotly_chart(fig)

                # Fetch and display fundamental data
                fundamental_data = get_fundamental_data(symbol)
                st.subheader("Fundamental Data")
                st.write(fundamental_data)

            except Exception as e:
                handle_error(e)

    # Data Analysis page
    elif page == "Data Analysis":
        st.title("Data Analysis")
        
        instrument_type = st.selectbox("Instrument Type", list(INSTRUMENTS.keys()))
        symbol = st.selectbox("Symbol", INSTRUMENTS[instrument_type])
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m", "1m"])
        
        if st.button("Run Analysis"):
            try:
                data = yf.download(symbol, period="1y", interval=timeframe)
                analysis_results = run_analysis(data)
                
                st.subheader("Price and Indicators")
                fig_price = plot_price_and_indicators(analysis_results)
                st.plotly_chart(fig_price)
                
                st.subheader("Volume Analysis")
                fig_volume = plot_volume_analysis(analysis_results)
                st.plotly_chart(fig_volume)
                
                st.subheader("Correlation Heatmap")
                fig_corr = analyze_correlations(analysis_results)
                st.plotly_chart(fig_corr)
                
                st.subheader("Sentiment Analysis")
                sentiment_scores = analyze_sentiment(symbol, timeframe)
                fig_sentiment = go.Figure(data=[go.Scatter(x=sentiment_scores.index, y=sentiment_scores.values)])
                fig_sentiment.update_layout(title="Sentiment Analysis", xaxis_title="Date", yaxis_title="Sentiment Score")
                st.plotly_chart(fig_sentiment)
                
                st.subheader("Custom Indicator")
                custom_indicator = create_custom_indicator(data)
                fig_custom = go.Figure(data=[go.Scatter(x=data.index, y=custom_indicator)])
                fig_custom.update_layout(title="Custom Indicator", xaxis_title="Date", yaxis_title="Indicator Value")
                st.plotly_chart(fig_custom)
                
                st.subheader("Fundamental Analysis")
                fundamental_data = get_fundamental_data(symbol)
                st.write(fundamental_data)
                
                st.subheader("Alternative Data")
                alt_data = fetch_alternative_data(symbol)
                st.write(alt_data)
                
                st.subheader("Pandas Profiling Report")
                profile = ProfileReport(analysis_results, title=f"{symbol} Data Profiling Report", explorative=True)
                st_profile_report(profile)

            except Exception as e:
                handle_error(e)

    # Model Training page
    elif page == "Model Training":
        st.title("Model Training")
        
        instrument_type = st.selectbox("Instrument Type", list(INSTRUMENTS.keys()))
        symbol = st.selectbox("Symbol", INSTRUMENTS[instrument_type])
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m", "1m"])
        
        training_days = st.slider("Training Period (days)", 30, 365, 180)
        
        if st.button("Train Model"):
            try:
                data = yf.download(symbol, period=f"{training_days}d", interval=timeframe)
                sentiment_scores = analyze_sentiment(symbol, timeframe)
                fundamental_data = add_fundamental_features(symbol)
                external_factors = incorporate_external_factors(symbol)
                alternative_data = fetch_alternative_data(symbol)
                
                X, y, scaler = prepare_data(data)
                X = add_sentiment_features(X, sentiment_scores)
                X = np.concatenate([X, fundamental_data, external_factors, alternative_data], axis=2)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                env_train = DummyVecEnv([lambda: TradingEnvironment(X_train, y_train, scaler)])
                env_test = DummyVecEnv([lambda: TradingEnvironment(X_test, y_test, scaler)])
                
                def optimize_ppo(trial):
                    return {
                        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
                        "n_steps": trial.suggest_int("n_steps", 16, 2048),
                        "batch_size": trial.suggest_int("batch_size", 8, 128),
                        "n_epochs": trial.suggest_int("n_epochs", 3, 30),
                        "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),
                        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.9, 1.0),
                        "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.4),
                        "ent_coef": trial.suggest_loguniform("ent_coef", 1e-8, 1e-1),
                    }
                
                def optimize_agent(trial):
                    model_params = optimize_ppo(trial)
                    model = PPO("MlpPolicy", env_train, verbose=0, **model_params)
                    model.learn(total_timesteps=10000)
                    mean_reward, _ = evaluate_policy(model, env_test, n_eval_episodes=5)
                    return mean_reward
                
                study = optuna.create_study(direction="maximize")
                study.optimize(optimize_agent, n_trials=20)
                
                best_params = study.best_params
                model = AdaptiveModel(env_train, best_params)
                
                with st.spinner("Training model... This may take a while."):
                    model.learn(total_timesteps=100000)
                
                st.success("Model training completed!")
                model.save(f"trained_model_{symbol}")
                
                # Evaluate the model
                mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=10)
                st.write(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                
                # Explain model decision
                sample_observation = env_test.reset()
                explanation = explain_model_decision(model, sample_observation)
                st.subheader("Model Decision Explanation")
                st.write(explanation)

            except Exception as e:
                handle_error(e)

    # Backtesting page
    elif page == "Backtesting":
        st.title("Backtesting")
        
        instrument_type = st.selectbox("Instrument Type", list(INSTRUMENTS.keys()))
        symbol = st.selectbox("Symbol", INSTRUMENTS[instrument_type])
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m", "1m"])
        
        backtesting_days = st.slider("Backtesting Period (days)", 30, 365, 90)
        
        if st.button("Run Backtest"):
            try:
                data = yf.download(symbol, period=f"{backtesting_days}d", interval=timeframe)
                sentiment_scores = analyze_sentiment(symbol, timeframe)
                fundamental_data = add_fundamental_features(symbol)
                external_factors = incorporate_external_factors(symbol)
                alternative_data = fetch_alternative_data(symbol)
                
                X, y, scaler = prepare_data(data)
                X = add_sentiment_features(X, sentiment_scores)
                X = np.concatenate([X, fundamental_data, external_factors, alternative_data], axis=2)
                
                env = DummyVecEnv([lambda: TradingEnvironment(X, y, scaler)])
                model = AdaptiveModel.load(f"trained_model_{symbol}")
                
                backtest_results = backtest(model, env, X, y)
                metrics = calculate_metrics(backtest_results)
                
                st.subheader("Backtest Results")
                for metric, value in metrics.items():
                    st.write(f"{metric}: {value}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=backtest_results['portfolio_values'], name="Portfolio Value"))
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Asset Price"))
                fig.update_layout(title="Backtest Results", xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig)

                # Calculate and display risk-adjusted returns
                risk_adjusted_returns = calculate_risk_adjusted_returns(backtest_results['returns'])
                st.subheader("Risk-Adjusted Returns")
                st.write(risk_adjusted_returns)

            except Exception as e:
                handle_error(e)

    # Live Trading page
    elif page == "Live Trading":
        st.title("Live Trading")
        
        instrument_type = st.selectbox("Instrument Type", list(INSTRUMENTS.keys()))
        symbol = st.selectbox("Symbol", INSTRUMENTS[instrument_type])
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m", "1m"])
        
        initial_balance = st.number_input("Initial Balance", min_value=1000, value=10000, step=1000)
        risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        
        if st.button("Start Live Trading"):
            try:
                model = AdaptiveModel.load(f"trained_model_{symbol}")
                
                async def trade():
                    exchange = ccxt.binance()
                    while True:
                        try:
                            # Fetch latest data
                            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                            data.set_index('timestamp', inplace=True)
                            
                            # Prepare data for model
                            X, y, scaler = prepare_data(data)
                            sentiment_scores = analyze_sentiment(symbol, timeframe)
                            fundamental_data = add_fundamental_features(symbol)
                            external_factors = incorporate_external_factors(symbol)
                            alternative_data = fetch_alternative_data(symbol)
                            X = add_sentiment_features(X, sentiment_scores)
                            X = np.concatenate([X, fundamental_data, external_factors, alternative_data], axis=2)
                            
                            # Get model prediction
                            action, _ = model.predict(X[-1])
                            
                            # Execute trade
                            current_price = data['close'].iloc[-1]
                            position_size = calculate_position_size(initial_balance, risk_per_trade, current_price)
                            
                            if action > 0:  # Buy
                                buy_amount = min(initial_balance, position_size)
                                order = place_advanced_order(exchange, symbol, 'buy', buy_amount, current_price)
                                st.write(f"Bought {buy_amount / current_price} {symbol} at {current_price}")
                            elif action < 0:  # Sell
                                sell_amount = min(position_size, exchange.fetch_balance()[symbol.split('/')[0]]['free'])
                                order = place_advanced_order(exchange, symbol, 'sell', sell_amount, current_price)
                                st.write(f"Sold {sell_amount} {symbol} at {current_price}")
                            
                            # Apply stop loss and take profit
                            open_positions = exchange.fetch_open_orders(symbol)
                            for position in open_positions:
                                position, initial_balance = apply_stop_loss_take_profit(position, initial_balance, current_price, risk_per_trade)
                            
                            # Update portfolio value
                            portfolio_value = initial_balance + sum([pos['amount'] * current_price for pos in open_positions])
                            st.write(f"Current Portfolio Value: ${portfolio_value:.2f}")
                            
                            # Wait for next candle
                            await asyncio.sleep(exchange.rateLimit / 1000)
                        
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            await asyncio.sleep(exchange.rateLimit / 1000)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(trade())

            except Exception as e:
                handle_error(e)

    # Paper Trading page
    elif page == "Paper Trading":
        st.title("Paper Trading")
        
        instrument_type = st.selectbox("Instrument Type", list(INSTRUMENTS.keys()))
        symbol = st.selectbox("Symbol", INSTRUMENTS[instrument_type])
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m", "1m"])
        
        initial_balance = st.number_input("Initial Balance", min_value=1000, value=10000, step=1000)
        risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        
        if st.button("Start Paper Trading"):
            try:
                model = AdaptiveModel.load(f"trained_model_{symbol}")
                paper_env = PaperTradingEnvironment(symbol, timeframe, initial_balance)
                
                async def paper_trade():
                    while True:
                        try:
                            # Fetch latest data
                            data = paper_env.get_latest_data()
                            
                            # Prepare data for model
                            X, y, scaler = prepare_data(data)
                            sentiment_scores = analyze_sentiment(symbol, timeframe)
                            fundamental_data = add_fundamental_features(symbol)
                            external_factors = incorporate_external_factors(symbol)
                            alternative_data = fetch_alternative_data(symbol)
                            X = add_sentiment_features(X, sentiment_scores)
                            X = np.concatenate([X, fundamental_data, external_factors, alternative_data], axis=2)
                            
                            # Get model prediction
                            action, _ = model.predict(X[-1])
                            
                            # Execute paper trade
                            trade_result = paper_env.execute_trade(action, risk_per_trade)
                            st.write(trade_result)
                            
                            # Update and display portfolio value
                            portfolio_value = paper_env.get_portfolio_value()
                            st.write(f"Current Paper Trading Portfolio Value: ${portfolio_value:.2f}")
                            
                            # Wait for next update
                            await asyncio.sleep(60)  # Update every minute
                        
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            await asyncio.sleep(60)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(paper_trade())

            except Exception as e:
                handle_error(e)

    # Portfolio page
    elif page == "Portfolio":
        st.title("Portfolio")
        
        portfolio = get_user_portfolio(st.session_state.user['username'])
        
        st.subheader("Current Holdings")
        for asset, amount in portfolio['holdings'].items():
            st.write(f"{asset}: {amount}")
        
        st.subheader("Transaction History")
        st.table(portfolio['transactions'])
        
        st.subheader("Portfolio Performance")
        fig = go.Figure(data=[go.Scatter(x=portfolio['performance']['dates'], y=portfolio['performance']['values'])])
        fig.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig)
        
        st.subheader("Risk-Adjusted Returns")
        risk_adjusted_returns = calculate_risk_adjusted_returns(portfolio['performance']['returns'])
        st.write(risk_adjusted_returns)
        
        st.subheader("Portfolio Optimization")
        if st.button("Optimize Portfolio"):
            try:
                optimized_weights = optimize_portfolio(portfolio['holdings'])
                st.write("Optimized Portfolio Weights:")
                for asset, weight in optimized_weights.items():
                    st.write(f"{asset}: {weight:.2%}")
            except Exception as e:
                handle_error(e)

    # QNN Model page
    elif page == "QNN Model":
        st.title("Quantum Neural Network (QNN) Model")
        
        instrument_type = st.selectbox("Instrument Type", list(INSTRUMENTS.keys()))
        symbol = st.selectbox("Symbol", INSTRUMENTS[instrument_type])
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m", "1m"])
        
        training_days = st.slider("Training Period (days)", 30, 365, 180)
        
        if st.button("Train QNN Model"):
            try:
                data = yf.download(symbol, period=f"{training_days}d", interval=timeframe)
                
                with st.spinner("Training QNN model... This may take a while."):
                    model, history = train_qnn_model(data)
                
                st.success("QNN model training completed!")
                
                # Display model parameters
                st.subheader("Model Parameters")
                st.write(model.get_weights())
                
                # Display training history
                st.subheader("Training History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
                st.plotly_chart(fig)
                
                # Make predictions
                predictions = predict_qnn(model, data)
                
                # Visualize predictions
                st.subheader("Price Predictions")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price'))
                fig.add_trace(go.Scatter(x=data.index, y=predictions, name='Predicted Price'))
                fig.update_layout(title='Actual vs Predicted Prices', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

            except Exception as e:
                handle_error(e)

    # GNN Model page
    elif page == "GNN Model":
        st.title("Graph Neural Network (GNN) Model")
        
        instrument_type = st.selectbox("Instrument Type", list(INSTRUMENTS.keys()))
        symbol = st.selectbox("Symbol", INSTRUMENTS[instrument_type])
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m", "1m"])
        
        training_days = st.slider("Training Period (days)", 30, 365, 180)
        
        if st.button("Train GNN Model"):
            try:
                data = yf.download(symbol, period=f"{training_days}d", interval=timeframe)
                
                with st.spinner("Training GNN model... This may take a while."):
                    model, history = train_gnn_model(data)
                
                st.success("GNN model training completed!")
                
                # Display model parameters
                st.subheader("Model Parameters")
                st.write(model.get_weights())
                
                # Display training history
                st.subheader("Training History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
                st.plotly_chart(fig)
                
                # Make predictions
                predictions = predict_gnn(model, data)
                
                # Visualize predictions
                st.subheader("Price Predictions")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price'))
                fig.add_trace(go.Scatter(x=data.index, y=predictions, name='Predicted Price'))
                fig.update_layout(title='Actual vs Predicted Prices', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

            except Exception as e:
                handle_error(e)

    # TensorFlow Model page
    elif page == "TensorFlow Model":
        st.title("TensorFlow Deep Learning Model")
        
        instrument_type = st.selectbox("Instrument Type", list(INSTRUMENTS.keys()))
        symbol = st.selectbox("Symbol", INSTRUMENTS[instrument_type])
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m", "1m"])
        
        training_days = st.slider("Training Period (days)", 30, 365, 180)
        
        if st.button("Train TensorFlow Model"):
            try:
                data = yf.download(symbol, period=f"{training_days}d", interval=timeframe)
                
                with st.spinner("Training TensorFlow model... This may take a while."):
                    model, history = train_tensorflow_model(data)
                
                st.success("TensorFlow model training completed!")
                
                # Display model summary
                st.subheader("Model Summary")
                model.summary(print_fn=lambda x: st.text(x))
                
                # Display training history
                st.subheader("Training History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
                st.plotly_chart(fig)
                
                # Make predictions
                predictions = predict_tensorflow(model, data)
                
                # Visualize predictions
                st.subheader("Price Predictions")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price'))
                fig.add_trace(go.Scatter(x=data.index, y=predictions, name='Predicted Price'))
                fig.update_layout(title='Actual vs Predicted Prices', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

            except Exception as e:
                handle_error(e)

    # Educational Resources page
    elif page == "Educational Resources":
        st.title("Educational Resources")
        
        topic = st.selectbox("Select a topic", ["Trading Basics", "Technical Analysis", "Fundamental Analysis", "Risk Management", "Machine Learning in Trading", "Quantum Computing in Finance"])
        
        if topic:
            content = get_educational_content(topic)
            st.markdown(content)

else:
    st.warning("Please log in to access the application.")

# Automated model retraining
def automated_retraining():
    while True:
        for instrument_type in INSTRUMENTS:
            for symbol in INSTRUMENTS[instrument_type]:
                try:
                    retrain_model(symbol)
                except Exception as e:
                    handle_error(e)
        time.sleep(24 * 60 * 60)  # Wait for 24 hours before next retraining cycle

# Start automated retraining in a separate thread
import threading
#retraining_thread = threading.Thread(target=automated_retraining)
#retraining_thread.start()

def schedule_background_tasks():
    # Schedule immediate data update
    update_all_data.delay()
    
    # Schedule daily tasks
    from celery.schedules import crontab
    
    app.conf.beat_schedule = {
        'update-all-data-every-hour': {
            'task': 'celery_tasks.update_all_data',
            'schedule': crontab(minute=0),  # Run every hour
        },
        'retrain-models-daily': {
            'task': 'celery_tasks.daily_model_retraining',
            'schedule': crontab(hour=0, minute=0),  # Run at midnight
        },
    }

# Call this function when the app starts
schedule_background_tasks()

# Start automated retraining in a separate thread
import threading
retraining_thread = threading.Thread(target=schedule_background_tasks)
retraining_thread.start()

