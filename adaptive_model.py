import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class AdaptiveCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose: int = 1):
        super(AdaptiveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.performance_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate current performance
            mean_reward, _ = evaluate_policy(self.model, self.training_env, n_eval_episodes=5)
            self.performance_history.append(mean_reward)
            
            # Check if performance is declining
            if len(self.performance_history) > 5 and np.mean(self.performance_history[-5:]) < np.mean(self.performance_history[-10:-5]):
                # Adjust learning rate
                self.model.learning_rate *= 0.9
                
                # Increase exploration
                self.model.ent_coef *= 1.1
        
        return True

class AdaptiveModel(PPO):
    def __init__(self, env, hyperparameters):
        super().__init__("MlpPolicy", env, **hyperparameters)
        self.adaptive_callback = AdaptiveCallback(check_freq=1000)
    
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="PPO", reset_num_timesteps=True, progress_bar=False):
        return super().learn(total_timesteps, callback=[callback, self.adaptive_callback] if callback else self.adaptive_callback,
                             log_interval=log_interval, tb_log_name=tb_log_name, reset_num_timesteps=reset_num_timesteps, progress_bar=progress_bar)

