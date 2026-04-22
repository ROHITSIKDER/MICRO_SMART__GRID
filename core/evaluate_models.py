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
        
        # Avoid division by zero for MAPE
        epsilon = 1e-7 
        
        for i, target in enumerate(targets):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / (y_test[:, i] + epsilon))) * 100
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            print(f"{target}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}")
            metrics[target] = {
                'mse': float(mse), 
                'rmse': float(rmse),
                'mae': float(mae), 
                'mape': float(mape),
                'r2': float(r2)
            }
            
        # Overall metrics
        mse_overall = mean_squared_error(y_test, y_pred)
        mae_overall = mean_absolute_error(y_test, y_pred)
        rmse_overall = np.sqrt(mse_overall)
        mape_overall = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100
        r2_overall = r2_score(y_test, y_pred)
        
        print(f"Overall RMSE: {rmse_overall:.4f}, Overall MAPE: {mape_overall:.2f}%")
        
        metrics['overall'] = {
            'mse': float(mse_overall),
            'rmse': float(rmse_overall),
            'mae': float(mae_overall),
            'mape': float(mape_overall),
            'r2': float(r2_overall)
        }
        return metrics

    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def main():
    # Get Project Root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X_PATH = os.path.join(PROJECT_ROOT, 'model', 'X.npy')
    Y_PATH = os.path.join(PROJECT_ROOT, 'model', 'y.npy')
    METRICS_OUT = os.path.join(PROJECT_ROOT, 'model', 'metrics.json')

    # Load data
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("Error: Data files not found.")
        return

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    # Split using the same random state as in training scripts
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    results = {}
    lstm_m = os.path.join(PROJECT_ROOT, 'model', 'saved_models', 'lstm_model.keras')
    cnn_lstm_m = os.path.join(PROJECT_ROOT, 'model', 'saved_models', 'cnn_lstm_model.keras')
    
    results['lstm'] = evaluate_model(lstm_m, X_test, y_test)
    results['cnn_lstm'] = evaluate_model(cnn_lstm_m, X_test, y_test)
    
    # Save results to JSON
    with open(METRICS_OUT, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nMetrics saved to {METRICS_OUT}")

if __name__ == "__main__":
    main()
