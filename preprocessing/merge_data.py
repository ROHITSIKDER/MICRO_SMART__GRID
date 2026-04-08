import pandas as pd
import os

def merge_datasets(nasa_path, biomass_path, output_path):
    print(f"Loading NASA data from {nasa_path}...")
    nasa_df = pd.read_csv(nasa_path)
    
    print(f"Loading biomass data from {biomass_path}...")
    biomass_df = pd.read_csv(biomass_path)
    
    # 1. Ensure DATE format matches
    nasa_df['DATE'] = pd.to_datetime(nasa_df['DATE'])
    biomass_df['DATE'] = pd.to_datetime(biomass_df['DATE'])
    
    # 2. Merge on DATE
    print("Merging datasets on DATE...")
    merged_df = pd.merge(nasa_df, biomass_df, on='DATE', how='inner')
    
    # 3. Keep columns: temperature, wind_speed, solar_irradiance, biomass_energy
    columns_to_keep = ['DATE', 'temperature', 'wind_speed', 'solar_irradiance', 'biomass_energy']
    merged_df = merged_df[columns_to_keep]
    
    # 4. Drop missing rows
    merged_df = merged_df.dropna()
    
    print(f"Final dataset has {len(merged_df)} rows.")
    print(f"Saving final data to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    print("Step 2 completed.")

if __name__ == "__main__":
    nasa_file = os.path.join('Data', 'cleaned_data.csv')
    biomass_file = os.path.join('Data', 'cleaned_biomass.csv')
    output_file = os.path.join('Data', 'final_data.csv')
    merge_datasets(nasa_file, biomass_file, output_file)
