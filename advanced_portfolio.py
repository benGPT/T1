import numpy as np
import pandas as pd

def calculate_risk_adjusted_returns(returns):
    """
    Calculate various risk-adjusted return metrics.
    """
    avg_return = np.mean(returns)
    std_dev = np.std(returns)
    
    sharpe_ratio = avg_return / std_dev

    sortino_ratio = avg_return / np.std(returns[returns < 0])
    
    max_drawdown = np.max(np.maximum.accumulate(np.cumprod(1 + returns)) - np.cumprod(1 + returns))
    calmar_ratio = avg_return / max_drawdown
    
    return {
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown": max_drawdown
    }

