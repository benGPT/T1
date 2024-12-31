import numpy as np

def calculate_metrics(backtest_results):
    returns = backtest_results['returns']
    portfolio_values = backtest_results['portfolio_values']
    
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
    sortino_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns[returns < 0])
    
    max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)
    
    win_rate = np.sum(returns > 0) / len(returns)
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Sortino Ratio': f"{sortino_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}"
    }

