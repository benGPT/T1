import numpy as np
from stable_baselines3 import PPO, A2C, SAC

class EnsembleModel:
    def __init__(self, env, hyperparameters):
        self.models = [
            PPO("MlpPolicy", env, **hyperparameters),
            A2C("MlpPolicy", env, **hyperparameters),
            SAC("MlpPolicy", env, **hyperparameters)
        ]
    
    def learn(self, total_timesteps):
        for model in self.models:
            model.learn(total_timesteps)
    
    def predict(self, observation, state=None, mask=None, deterministic=False):
        predictions = [model.predict(observation, state, mask, deterministic) for model in self.models]
        actions = [p[0] for p in predictions]
        states = [p[1] for p in predictions]
        
        # Ensemble decision (e.g., average of actions)
        ensemble_action = np.mean(actions, axis=0)
        
        return ensemble_action, states

    def save(self, path):
        for i, model in enumerate(self.models):
            model.save(f"{path}_model_{i}")
    
    @classmethod
    def load(cls, path, env):
        ensemble = cls(env, {})
        for i in range(len(ensemble.models)):
            ensemble.models[i] = ensemble.models[i].load(f"{path}_model_{i}")
        return ensemble

```python file="explainability.py"
import shap
import numpy as np

def explain_model_decision(model, observation):
    # Create a function that the explainer can call
    def f(x):
        return model.predict(x)[0]
    
    # Create the explainer
    explainer = shap.KernelExplainer(f, np.zeros((1, observation.shape[1])))
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(observation)
    
    # Create explanation
    feature_names = [f"Feature {i}" for i in range(observation.shape[1])]
    explanation = []
    for i in range(len(feature_names)):
        explanation.append(f"{feature_names[i]}: {shap_values[0][i]:.4f}")
    
    return "\n".join(explanation)

