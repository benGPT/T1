import pandas as pd
import numpy as np

def fetch_alternative_data(symbol):
    """
    Fetch alternative data for the given symbol.
    This is a placeholder function. In a real-world scenario, you would
    integrate with actual alternative data providers.
    """
    # Generate some dummy data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'social_media_sentiment': np.random.uniform(0, 1, len(dates)),
        'news_sentiment': np.random.uniform(0, 1, len(dates)),
        'web_traffic': np.random.randint(500, 2000, len(dates)),
    }, index=dates)
    
    return data

