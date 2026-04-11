import os

# 1. SETUP & CONFIGURATION
# Set Keras backend to torch for Python 3.14 compatibility (MUST BE BEFORE KERAS IMPORT)
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import json
from sklearn.model_selection import train_test_split
MODEL_PATH = 'model/saved_models/cnn_lstm_model.keras'
DATA_PATH = 'Data/final_data.csv'

# Optimization Constants
BATTERY_CAPACITY = 500  # kWh
GRID_COST = 8           # ₹ per kWh
BATTERY_COST = 2        # ₹ per kWh (Degradation/Maintenance cost)
RENEWABLE_COST = 0      # ₹ per kWh
INITIAL_BATTERY_LEVEL = 100  # Start with 100 kWh

def load_data_and_predict():
    """Loads the model and generates predictions for a 30-day period."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Error: Model or Data file missing. Please ensure training is complete.")
        return None, None

    try:
        # Load Model
        model = keras.models.load_model(MODEL_PATH)
        
        # Load Data
        df = pd.read_csv(DATA_PATH)
        
        # Prepare sequences for a sample period (last 60 days to ensure enough for 30 predictions)
        # Sequence length is 7
        test_data = df.tail(67).copy()
        features = test_data[['temperature', 'wind_speed', 'solar_irradiance', 'biomass_energy']].values
        
        X = []
        for i in range(len(features) - 7):
            X.append(features[i:i+7])
        X = np.array(X)
        
        # Predict
        predictions = model.predict(X, verbose=0)
        
        # Scaling factor: Convert normalized (0-1) to kWh
        # Assuming peak generation for each source is 200 kWh per day
        scaling_factor = 200 
        predicted_renewable = np.sum(predictions * scaling_factor, axis=1)
        
        # Return only 30 days of data for cleaner visualization
        return predicted_renewable[:30], test_data['DATE'].iloc[7:37].values
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def run_optimization(predicted_energy):
    """Energy Management System Logic."""
    n = len(predicted_energy)
    
    # Simulate realistic demand: Base 250 kWh + random oscillation
    # This represents a small community/microgrid load
    np.random.seed(42)
    demand = 250 + np.random.normal(0, 40, n)
    
    # Tracking variables
    battery_level = INITIAL_BATTERY_LEVEL
    energy_from_renewable = []
    energy_stored_in_battery = []
    energy_from_battery = []
    energy_from_grid = []
    battery_history = []
    
    for i in range(n):
        gen = predicted_energy[i]
        dem = demand[i]
        
        # Step 1: Use Renewables first
        used_ren = min(gen, dem)
        remaining_dem = dem - used_ren
        surplus_ren = max(0, gen - dem)
        
        # Step 2: Handle Surplus (Store in Battery)
        stored = 0
        if surplus_ren > 0:
            space_in_battery = BATTERY_CAPACITY - battery_level
            stored = min(surplus_ren, space_in_battery)
            battery_level += stored
            
        # Step 3: Handle Deficit (Use Battery then Grid)
        used_bat = 0
        used_grid = 0
        if remaining_dem > 0:
            used_bat = min(remaining_dem, battery_level)
            battery_level -= used_bat
            remaining_dem -= used_bat
            
            if remaining_dem > 0:
                used_grid = remaining_dem
        
        # Track steps
        energy_from_renewable.append(used_ren)
        energy_stored_in_battery.append(stored)
        energy_from_battery.append(used_bat)
        energy_from_grid.append(used_grid)
        battery_history.append(battery_level)

    # 4. COST CALCULATIONS
    # With Optimization: Battery usage cost (₹2) + Grid cost (₹8)
    cost_with_opt_list = (np.array(energy_from_battery) * BATTERY_COST) + (np.array(energy_from_grid) * GRID_COST)
    cost_with_opt = cost_with_opt_list.sum()
    
    # Without Optimization: All deficit (Demand - Renewable) comes from Grid at ₹8
    deficit_no_opt = np.maximum(0, demand - np.array(predicted_energy))
    cost_no_opt = (deficit_no_opt * GRID_COST).sum()
    
    results = {
        'demand': demand,
        'predicted_energy': predicted_energy,
        'ren_used': energy_from_renewable,
        'bat_used': energy_from_battery,
        'grid_used': energy_from_grid,
        'bat_level': battery_history,
        'cost_opt_total': float(cost_with_opt),
        'cost_no_opt_total': float(cost_no_opt)
    }
    
    # Add savings and metrics
    savings = cost_no_opt - cost_with_opt
    results['savings'] = float(savings)
    results['savings_pct'] = float((savings / cost_no_opt) * 100) if cost_no_opt > 0 else 0
    results['grid_reduction_pct'] = float((np.sum(deficit_no_opt) - np.sum(energy_from_grid)) / np.sum(deficit_no_opt) * 100) if np.sum(deficit_no_opt) > 0 else 0
    
    return results

def visualize_results(results):
    """Generates and saves research-quality plots."""
    days = np.arange(1, len(results['demand']) + 1)
    
    plt.figure(figsize=(12, 10))

    # Graph 1: Demand vs Predicted Supply
    plt.subplot(2, 1, 1)
    plt.plot(days, results['demand'], 'r--', label='Energy Demand (kWh)', linewidth=2)
    plt.plot(days, results['predicted_energy'], 'g-', label='Predicted Renewable Supply (kWh)', linewidth=2)
    plt.fill_between(days, results['predicted_energy'], results['demand'], 
                     where=(results['predicted_energy'] > results['demand']), 
                     color='green', alpha=0.2, label='Surplus (To Battery)')
    plt.fill_between(days, results['predicted_energy'], results['demand'], 
                     where=(results['predicted_energy'] < results['demand']), 
                     color='red', alpha=0.2, label='Deficit (From Bat/Grid)')
    plt.title('Microgrid Performance: Energy Demand vs Predicted Supply', fontsize=14)
    plt.xlabel('Day Index')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Graph 2: Cost Comparison
    plt.subplot(2, 1, 2)
    categories = ['Without Optimization\n(Direct Grid)', 'With Optimization\n(Battery Mgmt)']
    costs = [results['cost_no_opt_total'], results['cost_opt_total']]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = plt.bar(categories, costs, color=colors, width=0.4)
    plt.title('Economic Impact: Operational Cost Comparison (30 Days)', fontsize=14)
    plt.ylabel('Total Cost (₹)')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 100, f'₹{int(yval):,}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    # Save to microgrid_results.png for consistency
    plt.savefig('microgrid_results.png')
    print("\n[System] Visualization report saved to 'microgrid_results.png'")

def main():
    print("="*60)
    print("       MICROGRID ENERGY OPTIMIZATION & COST ANALYSIS")
    print("="*60)
    
    # Step 1: Load Predictions from CNN-LSTM
    print("\n[Step 1] Initializing CNN-LSTM inference...")
    pred_energy, dates = load_data_and_predict()
    if pred_energy is None: return

    # Step 2: Run Optimization Logic
    print("[Step 2] Simulating Energy Management Strategy...")
    results = run_optimization(pred_energy)
    
    # Step 3: Clear Printed Results
    print("\n" + "*"*60)
    print("                OPTIMIZATION RESULTS SUMMARY")
    print("*"*60)
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-" * 60)
    print(f"{'Total Simulation Period':<30} | {len(pred_energy)} Days")
    print(f"{'Total Demand':<30} | {results['demand'].sum():.2f} kWh")
    print(f"{'Total Renewable Generation':<30} | {results['predicted_energy'].sum():.2f} kWh")
    print(f"{'Grid Dependency (Opt)':<30} | {np.sum(results['grid_used']):.2f} kWh")
    print(f"{'Battery Contribution':<30} | {np.sum(results['bat_used']):.2f} kWh")
    print("-" * 60)
    print(f"{'Estimated Cost WITHOUT Opt':<30} | ₹{results['cost_no_opt_total']:.2f}")
    print(f"{'Estimated Cost WITH Opt':<30} | ₹{results['cost_opt_total']:.2f}")
    print(f"{'NET SAVINGS':<30} | ₹{results['savings']:.2f}")
    print(f"{'PERCENTAGE SAVED':<30} | {results['savings_pct']:.2f}%")
    print("*"*60)

    # Step 4: Save optimization results
    with open('model/optimization_results.json', 'w') as f:
        # We only save JSON serializable part
        json_results = {k: v for k, v in results.items() if not isinstance(v, (np.ndarray, list))}
        json.dump(json_results, f, indent=4)
    print("\nOptimization metrics saved to 'model/optimization_results.json'")

    # Step 5: Final Summary
    print("\n[Step 3] Scientific Conclusion:")
    print(f"By utilizing a 500kWh battery storage system alongside CNN-LSTM predictions,")
    print(f"the microgrid successfully reduced grid reliance by {np.sum(results['bat_used']):.2f} kWh.")
    print(f"The optimization module achieved a cost reduction of {results['savings_pct']:.1f}%, making the")
    print(f"hybrid system significantly more economically viable.")

    # Step 6: Visualization
    print("\n[Step 4] Generating Research Plots...")
    visualize_results(results)

if __name__ == "__main__":
    main()
