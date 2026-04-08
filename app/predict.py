import numpy as np
import pandas as pd
import os
import sys

# Set Keras backend to torch
os.environ['KERAS_BACKEND'] = 'torch'
# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras

def main():
    # Paths
    model_path = 'model/saved_models/cnn_lstm_model.keras'
    data_path = 'Data/final_data.csv'

    # Load the trained model
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load final dataset
    try:
        if not os.path.exists(data_path):
            print(f"Error: Dataset not found at {data_path}")
            return
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Select last 7 rows
    last_7_rows = df.tail(7)

    # Extract features: temperature, wind_speed, solar_irradiance, biomass_energy
    features = last_7_rows[['temperature', 'wind_speed', 'solar_irradiance', 'biomass_energy']]

    # Convert into numpy array
    input_array = features.to_numpy()

    # Reshape input to match model: shape should be (1, 7, 4)
    input_reshaped = input_array.reshape(1, 7, 4)

    # Make prediction
    prediction_array = model.predict(input_reshaped, verbose=0)
    
    # The model outputs [wind, solar, biomass]
    wind_pred = float(prediction_array[0][0])
    solar_pred = float(prediction_array[0][1])
    biomass_pred = float(prediction_array[0][2])

    # Print output clearly
    print("-" * 35)
    print(f"Predicted Wind Speed: {wind_pred:.4f}")
    print(f"Predicted Solar Energy: {solar_pred:.4f}")
    print(f"Predicted Biomass Energy: {biomass_pred:.4f}")

    # Smart Grid Logic
    # IF solar > 0.7 → "High Solar Generation"
    # IF biomass > 0.5 → "Biomass Contributing"
    # IF both low → "Energy Deficit"
    # ELSE → "Balanced System"
    
    if solar_pred > 0.7:
        grid_status = "High Solar Generation"
    elif biomass_pred > 0.5:
        grid_status = "Biomass Contributing"
    elif wind_pred > 0.5:
        grid_status = "High Wind Contribution"
    elif solar_pred < 0.3 and biomass_pred < 0.3 and wind_pred < 0.3:
        grid_status = "Energy Deficit"
    else:
        grid_status = "Balanced System"

    print(f"Smart Grid Logic: {grid_status}")
    print("-" * 35)

if __name__ == "__main__":
    main()
