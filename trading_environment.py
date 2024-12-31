import gym
import numpy as np

class TradingEnvironment(gym.Env):
    def __init__(self, X, y, scaler):
        super(TradingEnvironment, self).__init__()
        self.X = X
        self.y = y
        self.scaler = scaler
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))  # Continuous action space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1] + 3,))
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.btc_held = 0
        self.portfolio_value = self.balance
        return self._next_observation()
    
    def _next_observation(self):
        obs = np.concatenate([
            self.X[self.current_step].flatten(),
            [self.balance, self.btc_held, self.portfolio_value]
        ])
        return obs
    
    def step(self, action):
        current_price = self.y[self.current_step][0]
        next_price = self.y[self.current_step + 1][0] if self.current_step + 1 < len(self.y) else current_price
        
        # Execute trade
        if action[0] > 0:  # Buy
            btc_to_buy = min(self.balance / current_price, action[0])
            self.btc_held += btc_to_buy
            self.balance -= btc_to_buy * current_price
        elif action[0] < 0:  # Sell
            btc_to_sell = min(self.btc_held, -action[0])
            self.balance += btc_to_sell * current_price
            self.btc_held -= btc_to_sell
        
        # Calculate reward
        new_portfolio_value = self.balance + self.btc_held * next_price
        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        self.portfolio_value = new_portfolio_value
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.y) - 1
        
        return self._next_observation(), reward, done, {}

    def render(self):
        # Implement visualization if needed
        pass

