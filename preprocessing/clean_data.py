import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def clean_dataset(input_path, output_path):
    """
    Loads, cleans, and normalizes the NASA POWER dataset.
    """
    try:
        print(f"Loading data from {input_path}...")
        
        # 1. Load the dataset (skipping first 11 metadata rows)
        # We skip 11 rows because the data header starts at line 12
        df = pd.read_csv(input_path, skiprows=11)
        
        # 2. Handle missing values
        # NASA POWER uses -999 for missing data
        print("Handling missing values (replacing -999 with NaN)...")
        df.replace(-999, np.nan, inplace=True)
        
        # Drop any rows that have missing or invalid values
        df.dropna(inplace=True)
        
        # 3. Convert YEAR, MO, DY to a single DATE column
        print("Converting dates...")
        df['DATE'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(columns={
            'YEAR': 'year', 
            'MO': 'month', 
            'DY': 'day'
        }))
        
        # 4. Rename columns properly
        print("Renaming columns...")
        df.rename(columns={
            'T2M': 'temperature',
            'WS10M': 'wind_speed',
            'ALLSKY_SFC_SW_DWN': 'solar_irradiance'
        }, inplace=True)
        
        # 5. Keep only the necessary columns
        cols_to_keep = ['DATE', 'temperature', 'wind_speed', 'solar_irradiance']
        df = df[cols_to_keep]
        
        # 6. Normalize the features using MinMaxScaler
        # We only normalize the numerical features (not the DATE column)
        print("Normalizing features...")
        features = ['temperature', 'wind_speed', 'solar_irradiance']
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        
        # 7. Save cleaned data
        print(f"Saving cleaned data to {output_path}...")
        df.to_csv(output_path, index=False)
        
        print("Data processing completed successfully!")
        return True

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return False

if __name__ == "__main__":
    # Define file paths
    # Note: Using Data/ as shown in the directory structure
    raw_data_file = os.path.join('Data', 'raw_data.csv')
    cleaned_data_file = os.path.join('Data', 'cleaned_data.csv')
    
    # Ensure Data directory exists (though it should according to file tree)
    if not os.path.exists('Data'):
        os.makedirs('Data')
        
    clean_dataset(raw_data_file, cleaned_data_file)
