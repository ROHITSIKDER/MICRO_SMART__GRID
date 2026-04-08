import numpy as np
import os
import sys

# Set Keras backend to torch
os.environ['KERAS_BACKEND'] = 'torch'

def train_model():
    print("--- Starting Training Script ---")
    sys.stdout.flush()
    
    try:
        print("Importing modules (this may take a moment)...")
        sys.stdout.flush()
        from sklearn.model_selection import train_test_split
        import keras
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        print(f"Imports successful. Keras version: {keras.__version__} Backend: {keras.backend.backend()}")
        sys.stdout.flush()

        # 1. Load data from numpy files
        print("Loading data from model/X.npy and model/y.npy...")
        sys.stdout.flush()
        if not os.path.exists('model/X.npy') or not os.path.exists('model/y.npy'):
            print("Error: Data files not found. Please run preprocessing first.")
            return

        X = np.load('model/X.npy')
        y = np.load('model/y.npy')
        print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
        sys.stdout.flush()

        # 2. Split into train/test (80% train, 20% test)
        print("Splitting data into train and test sets...")
        sys.stdout.flush()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Split complete.")
        sys.stdout.flush()

        # 3. Build the LSTM model
        print("Building LSTM model...")
        sys.stdout.flush()
        from keras.layers import Input
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(50),
            Dropout(0.2),
            Dense(3)
        ])
        print("Model built.")
        sys.stdout.flush()

        # 4. Compile the model
        print("Compiling model...")
        sys.stdout.flush()
        model.compile(optimizer='adam', loss='mse')
        print("Model compiled.")
        sys.stdout.flush()

        # 5. Train the model
        print("Starting fit process (20 epochs)...")
        sys.stdout.flush()
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        print("Training complete.")
        sys.stdout.flush()

        # 7. Save the model
        print("Saving model...")
        sys.stdout.flush()
        os.makedirs('model/saved_models', exist_ok=True)
        # Using .keras extension as .h5 is legacy
        model_path = 'model/saved_models/lstm_model.keras'
        model.save(model_path)
        print(f"Model saved successfully to {model_path}")
        sys.stdout.flush()

    except Exception as e:
        print(f"AN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

if __name__ == "__main__":
    train_model()
