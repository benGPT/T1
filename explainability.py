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

