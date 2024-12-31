import numpy as np

def backtest(model, env, X, y):
    obs = env.reset()
    done = False
    portfolio_values = [10000]
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(env.portfolio_value)
    
    return {
        'portfolio_values': portfolio_values,
        'returns': np.diff(portfolio_values) / portfolio_values[:-1]
    }

