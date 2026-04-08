import pandas as pd
import numpy as np
import os

def create_final_sequences(input_path, x_output_path, y_output_path, sequence_length=7):
    print(f"Loading final data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Drop DATE column
    data_df = df.drop(columns=['DATE'])
    
    # Features: temperature, wind_speed, solar_irradiance, biomass_energy
    # Ensure they are in this order
    cols = ['temperature', 'wind_speed', 'solar_irradiance', 'biomass_energy']
    data = data_df[cols].values
    
    X = []
    y = []
    
    print(f"Creating sequences (length={sequence_length})...")
    # solar_irradiance is at index 2, biomass_energy is at index 3
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 1:4]) # wind_speed, solar_irradiance, and biomass_energy
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"X shape: {X.shape}") # (samples, 7, 4)
    print(f"y shape: {y.shape}") # (samples,)
    
    print(f"Saving X to {x_output_path}...")
    np.save(x_output_path, X)
    
    print(f"Saving y to {y_output_path}...")
    np.save(y_output_path, y)
    print("Step 3 completed.")

if __name__ == "__main__":
    input_file = os.path.join('Data', 'final_data.csv')
    x_save_path = os.path.join('model', 'X.npy')
    y_save_path = os.path.join('model', 'y.npy')
    
    create_final_sequences(input_file, x_save_path, y_save_path)
