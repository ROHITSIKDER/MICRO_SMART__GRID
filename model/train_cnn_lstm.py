import numpy as np
import os
import sys

# Set Keras backend to torch for compatibility with Python 3.14
os.environ['KERAS_BACKEND'] = 'torch'

def train_cnn_lstm():
    print("--- Starting CNN-LSTM Training Script ---")
    sys.stdout.flush()
    
    try:
        print("Importing modules...")
        sys.stdout.flush()
        from sklearn.model_selection import train_test_split
        import keras
        from keras.models import Sequential
        from keras.layers import Conv1D, LSTM, Dense, Dropout, Input
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

        # 3. Build the CNN-LSTM model
        print("Building CNN-LSTM model...")
        sys.stdout.flush()
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            # Conv1D (64 filters, kernel size 2, relu)
            Conv1D(filters=64, kernel_size=2, activation='relu'),
            # LSTM (50 units)
            LSTM(50),
            # Dropout (0.2)
            Dropout(0.2),
            # Dense (1 output)
            Dense(1)
        ])
        print("Model built.")
        sys.stdout.flush()

        # 4. Compile the model
        print("Compiling model (optimizer='adam', loss='mse')...")
        sys.stdout.flush()
        model.compile(optimizer='adam', loss='mse')
        print("Model compiled.")
        sys.stdout.flush()

        # 5. Train the model
        print("Starting fit process (20 epochs, batch_size=32)...")
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

        # 6. Save the model
        print("Saving model...")
        sys.stdout.flush()
        os.makedirs('model/saved_models', exist_ok=True)
        model_path = 'model/saved_models/cnn_lstm_model.keras'
        model.save(model_path)
        print(f"Model saved successfully to {model_path}")
        sys.stdout.flush()

    except Exception as e:
        print(f"AN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

if __name__ == "__main__":
    train_cnn_lstm()
