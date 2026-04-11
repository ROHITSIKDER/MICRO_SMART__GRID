import pandas as pd
import numpy as np
import os

def prepare_sequences(input_path, x_output_path, y_output_path, window_size=7):
    """
    Converts cleaned data into time-series sequences for machine learning.
    """
    try:
        print(f"Loading cleaned data from {input_path}...")
        df = pd.read_csv(input_path)
        
        # 1. Drop DATE column (not needed for numerical training)
        df = df.drop(columns=['DATE'])
        
        # 2. Convert dataframe to numpy array for easier indexing
        data = df.values
        
        X = []
        y = []
        
        # 3. Create sliding window sequences
        # We iterate through the data and take chunks of 'window_size' days
        print(f"Creating sequences with a window size of {window_size} days...")
        for i in range(len(data) - window_size):
            # X: Past 7 days of all features (temp, wind, solar, biomass)
            X.append(data[i:i + window_size])
            
            # y: Next day's features (wind, solar, biomass) - indices 1 to 4
            y.append(data[i + window_size, 1:4])
            
        # 4. Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # 5. Print shapes for verification
        print(f"Shape of X (Input sequences): {X.shape}") # (Samples, Time Steps, Features)
        print(f"Shape of y (Target values): {y.shape}")    # (Samples,)
        
        # 6. Save sequences to files
        print(f"Saving X to {x_output_path}...")
        np.save(x_output_path, X)
        
        print(f"Saving y to {y_output_path}...")
        np.save(y_output_path, y)
        
        print("Sequence preparation completed successfully!")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    # Define paths
    input_file = os.path.join('Data', 'final_data.csv')
    x_save_path = os.path.join('model', 'X.npy')
    y_save_path = os.path.join('model', 'y.npy')
    
    # Ensure model directory exists
    if not os.path.exists('model'):
        os.makedirs('model')
        
    prepare_sequences(input_file, x_save_path, y_save_path)
