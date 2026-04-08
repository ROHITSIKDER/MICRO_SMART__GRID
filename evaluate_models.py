import numpy as np
import os
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set Keras backend to torch
os.environ['KERAS_BACKEND'] = 'torch'

def evaluate_model(model_path, X_test, y_test):
    print(f"\n--- Evaluating Model: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    try:
        model = keras.models.load_model(model_path)
        y_pred = model.predict(X_test, verbose=0)
        
        # Calculate metrics for each output (wind, solar, biomass)
        targets = ['Wind Speed', 'Solar Energy', 'Biomass Energy']
        for i, target in enumerate(targets):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            print(f"{target}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            
        # Overall metrics
        mse_overall = mean_squared_error(y_test, y_pred)
        print(f"Overall MSE: {mse_overall:.4f}")

    except Exception as e:
        print(f"Error evaluating model: {e}")

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
    
    evaluate_model('model/saved_models/lstm_model.keras', X_test, y_test)
    evaluate_model('model/saved_models/cnn_lstm_model.keras', X_test, y_test)

if __name__ == "__main__":
    main()
