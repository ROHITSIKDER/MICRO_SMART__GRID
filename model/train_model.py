import numpy as np
import os
# Set Keras backend to torch
os.environ['KERAS_BACKEND'] = 'torch'
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input

def train_model():
    # 1. Load data from numpy files
    print("Loading data...")
    if not os.path.exists('model/X.npy') or not os.path.exists('model/y.npy'):
        print("Error: Data files not found. Please run preprocessing first.")
        return

    X = np.load('model/X.npy')
    y = np.load('model/y.npy')

    # 2. Split into train/test (80% train, 20% test)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Build the LSTM model
    print("Building LSTM model...")
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        # LSTM layer with 50 units
        LSTM(50),
        
        # Dropout layer to prevent overfitting
        Dropout(0.2),
        
        # Dense output layer for prediction
        Dense(1)
    ])

    # 4. Compile the model
    model.compile(optimizer='adam', loss='mse')

    # 5. Train the model
    # epochs=20, batch_size=32, validation_split=0.2
    print("Starting training...")
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1  # 6. Print progress
    )

    # 7. Save the model
    os.makedirs('model/saved_models', exist_ok=True)
    model_path = 'model/saved_models/lstm_model.keras'
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()
