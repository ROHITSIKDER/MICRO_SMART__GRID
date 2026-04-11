import numpy as np
import os
import json

# Set Keras backend to torch before importing keras
os.environ['KERAS_BACKEND'] = 'torch'

import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model_path, X_test, y_test):
    print(f"\n--- Evaluating Model: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None

    try:
        model = keras.models.load_model(model_path)
        y_pred = model.predict(X_test, verbose=0)
        
        metrics = {}
        # Calculate metrics for each output (wind, solar, biomass)
        targets = ['Wind Speed', 'Solar Energy', 'Biomass Energy']
        for i, target in enumerate(targets):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            print(f"{target}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            metrics[target] = {'mse': float(mse), 'mae': float(mae), 'r2': float(r2)}
            
        # Overall metrics
        mse_overall = mean_squared_error(y_test, y_pred)
        mae_overall = mean_absolute_error(y_test, y_pred)
        r2_overall = r2_score(y_test, y_pred)
        print(f"Overall MSE: {mse_overall:.4f}")
        
        metrics['overall'] = {
            'mse': float(mse_overall),
            'mae': float(mae_overall),
            'r2': float(r2_overall),
            'rmse': float(np.sqrt(mse_overall))
        }
        return metrics

    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def main():
    # Load data
    if not os.path.exists('model/X.npy') or not os.path.exists('model/y.npy'):
        print("Error: Data files not found.")
        return

    X = np.load('model/X.npy')
    y = np.load('model/y.npy')
    
    # Split using the same random state as in training scripts
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    results = {}
    results['lstm'] = evaluate_model('model/saved_models/lstm_model.keras', X_test, y_test)
    results['cnn_lstm'] = evaluate_model('model/saved_models/cnn_lstm_model.keras', X_test, y_test)
    
    # Save results to JSON
    with open('model/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nMetrics saved to 'model/metrics.json'")

if __name__ == "__main__":
    main()
