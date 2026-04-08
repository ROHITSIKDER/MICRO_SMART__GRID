import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def clean_biomass_data(input_path, output_path):
    print(f"Loading biomass data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Identify columns and create DATE
    if 'Year' in df.columns and 'Month' in df.columns and 'Day' in df.columns:
        df['DATE'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df = df.drop(columns=['Year', 'Month', 'Day'])
    
    # 2. Rename biomass column
    # Auto-detect: look for 'biogas_production' or 'biomass'
    biomass_col = None
    for col in df.columns:
        if 'biogas_production' in col.lower() or 'biomass' in col.lower():
            biomass_col = col
            break
    
    if biomass_col:
        print(f"Found biomass column: {biomass_col}")
        df = df.rename(columns={biomass_col: 'biomass_energy'})
    else:
        raise ValueError("Could not find biomass column in the dataset.")

    # Aggregate by DATE if multiple entries exist
    df = df.groupby('DATE').mean().reset_index()

    # 3. Handle missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['biomass_energy'])
    
    # 4. Normalize biomass_energy (0 to 1)
    scaler = MinMaxScaler()
    df['biomass_energy'] = scaler.fit_transform(df[['biomass_energy']])
    
    # 5. Keep only DATE and biomass_energy for merging
    df = df[['DATE', 'biomass_energy']]
    
    print(f"Saving cleaned biomass data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Step 1 completed.")

if __name__ == "__main__":
    input_file = os.path.join('Data', 'biogas_dataset.csv')
    output_file = os.path.join('Data', 'cleaned_biomass.csv')
    clean_biomass_data(input_file, output_file)
