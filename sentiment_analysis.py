import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import pandas as pd

nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(symbol, timeframe):
    # Fetch news articles
    ticker = yf.Ticker(symbol)
    news = ticker.news
    
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Analyze sentiment for each news article
    sentiments = []
    for article in news:
        sentiment = sia.polarity_scores(article['title'])
        sentiments.append(sentiment['compound'])
    
    # Create a DataFrame with dates and sentiment scores
    df = pd.DataFrame({'date': [article['providerPublishTime'] for article in news],
                       'sentiment': sentiments})
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df.set_index('date', inplace=True)
    
    # Resample to match the trading data timeframe
    if timeframe == '1d':
        df = df.resample('D').mean()
    elif timeframe == '1h':
        df = df.resample('H').mean()
    elif timeframe in ['15m', '5m', '1m']:
        df = df.resample('T').mean()
    
    return df['sentiment']

