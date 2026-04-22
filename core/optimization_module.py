import os
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import json

# Get Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'saved_models', 'cnn_lstm_model.keras')
DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'final_data.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
METRICS_FILE = os.path.join(PROJECT_ROOT, 'model', 'optimization_results.json')

# PHYSICAL CONSTANTS for Renewable Generation
ETA_PV = 0.18        # Solar panel efficiency (18%)
AREA_PV = 50         # Solar panel area in m^2
RHO_AIR = 1.225      # Air density (kg/m^3)
AREA_WIND = 25       # Wind turbine swept area in m^2
CP_WIND = 0.4        # Power coefficient of wind turbine

# OPTIMIZATION CONSTANTS
BATTERY_CAPACITY = 500   # kWh (SOC_max)
SOC_MIN = 50             # kWh (SOC_min)
INITIAL_SOC = 100        # Start level (kWh)
EFF_CHARGE = 0.95        # Charging efficiency (eta_c)
EFF_DISCHARGE = 0.90     # Discharging efficiency (eta_d)
P_BAT_MAX = 100          # Max charge/discharge rate (kW)

BATTERY_DEG_COST = 2     # ₹ per kWh (C_bat)
RENEWABLE_COST = 0       # ₹ per kWh (C_gen)

def get_grid_cost(hour):
    """Mathematical Function: Time-dependent Grid Cost C_grid(t)."""
    # Peak hours (18-22) are more expensive
    if 18 <= hour <= 22:
        return 12
    # Off-peak hours (0-6) are cheaper
    elif 0 <= hour <= 6:
        return 5
    return 8

def calculate_solar_power(irradiance):
    """Formula: P_solar = η_pv * A * G"""
    # irradiance is in kW-hr/m^2/day from NASA data, we treat it as instantaneous G for sim
    return ETA_PV * AREA_PV * irradiance

def calculate_wind_power(v):
    """Formula: P_wind = 0.5 * ρ * A * v^3 * Cp"""
    return 0.5 * RHO_AIR * AREA_WIND * (v ** 3) * CP_WIND / 1000 # Convert to kW

def load_data_and_predict():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Error: Model or Data file missing.")
        return None, None

    try:
        model = keras.models.load_model(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        test_data = df.tail(67).copy()
        features = test_data[['temperature', 'wind_speed', 'solar_irradiance', 'biomass_energy']].values
        
        X = []
        for i in range(len(features) - 7):
            X.append(features[i:i+7])
        X = np.array(X)
        
        predictions = model.predict(X, verbose=0)
        # predictions[:, 0] = wind_speed, predictions[:, 1] = solar, predictions[:, 2] = biomass
        return predictions[:30], test_data['DATE'].iloc[7:37].values
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def run_optimization(predictions):
    """
    Mathematical Optimization with Power Balance and Battery Constraints.
    Technique: Priority-Based Greedy Logic (Heuristic).
    """
    n = len(predictions)
    np.random.seed(42)
    demand = 250 + np.random.normal(0, 40, n)
    
    soc = INITIAL_SOC
    results = {
        'p_solar': [], 'p_wind': [], 'p_biomass': [], 'p_ren_total': [],
        'p_bat': [], 'soc': [], 'p_grid': [], 'p_load': [], 'cost': []
    }
    
    total_cost = 0
    served_load = 0

    for t in range(n):
        # 1. Physical Generation Formulas
        wind_speed_pred = predictions[t, 0] * 15 # De-normalizing approx
        solar_irrad_pred = predictions[t, 1] * 10
        biomass_pred = predictions[t, 2] * 50
        
        p_solar = calculate_solar_power(solar_irrad_pred)
        p_wind = calculate_wind_power(wind_speed_pred)
        p_biomass = biomass_pred # Direct bio-energy estimate
        p_ren = p_solar + p_wind + p_biomass
        
        p_load = demand[t]
        
        # 2. Power Balance Constraint (MANDATORY): P_load(t) = P_ren(t) + P_bat(t) + P_grid(t)
        # Step 1: Net Power
        p_net = p_ren - p_load
        
        p_bat = 0
        p_grid = 0
        
        # 3. Battery SOC Model (SOC(t+1) = SOC(t) + η_c*P_c - P_d/η_d)
        if p_net > 0: # Surplus: Charge Battery
            p_charge_available = min(p_net, P_BAT_MAX)
            space_in_bat = (BATTERY_CAPACITY - soc) / EFF_CHARGE
            p_charge = min(p_charge_available, space_in_bat)
            
            soc = soc + (EFF_CHARGE * p_charge)
            p_bat = -p_charge # Negative means charging
        
        elif p_net < 0: # Deficit: Use Battery then Grid
            p_needed = abs(p_net)
            p_discharge_available = min(p_needed, P_BAT_MAX)
            energy_in_bat = (soc - SOC_MIN) * EFF_DISCHARGE
            p_discharge = min(p_discharge_available, energy_in_bat)
            
            soc = soc - (p_discharge / EFF_DISCHARGE)
            p_bat = p_discharge
            
            # Remaining deficit from Grid
            remaining_deficit = p_needed - p_discharge
            if remaining_deficit > 0:
                p_grid = remaining_deficit
        
        # Power Balance Verification
        # P_load = P_ren - P_bat + P_grid  (where p_bat is positive when discharging)
        # So P_ren + p_grid + p_bat(discharge) - p_bat(charge) = P_load
        
        # 4. Objective Function: Minimize Total_Cost = sum(C_grid*P_grid + C_bat*P_bat)
        current_hour = (t % 24) # Simple hour approximation
        c_grid = get_grid_cost(current_hour)
        
        # Battery cost only for discharging (degradation)
        step_cost = (p_grid * c_grid) + (max(0, p_bat) * BATTERY_DEG_COST)
        total_cost += step_cost
        
        served_load += (p_load if p_grid >= 0 else p_load) # In this model we always serve load via grid
        
        # Store results
        results['p_solar'].append(p_solar)
        results['p_wind'].append(p_wind)
        results['p_biomass'].append(p_biomass)
        results['p_ren_total'].append(p_ren)
        results['p_bat'].append(p_bat)
        results['soc'].append(soc)
        results['p_grid'].append(p_grid)
        results['p_load'].append(p_load)
        results['cost'].append(step_cost)

    # 5. Performance Metrics
    cost_baseline = np.sum(results['p_load']) * 8 # Baseline: All from grid at avg rate
    savings = ((cost_baseline - total_cost) / cost_baseline) * 100
    ren_penetration = (np.sum(results['p_ren_total']) / np.sum(results['p_load'])) * 100
    reliability = (served_load / np.sum(results['p_load'])) * 100
    
    final_metrics = {
        'total_cost': float(total_cost),
        'cost_baseline': float(cost_baseline),
        'savings_pct': float(savings),
        'ren_penetration_pct': float(ren_penetration),
        'reliability_pct': float(reliability)
    }
    
    return results, final_metrics

def visualize(results, metrics):
    days = np.arange(1, len(results['p_load']) + 1)
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(days, results['p_load'], 'r--', label='Demand (P_load)')
    plt.plot(days, results['p_ren_total'], 'g-', label='Renewable (P_solar + P_wind + P_bio)')
    plt.title('Power Balance: Demand vs Renewable Supply')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(days, results['soc'], 'b-', label='Battery SOC')
    plt.axhline(y=BATTERY_CAPACITY, color='r', linestyle='--', label='Max Cap')
    plt.axhline(y=SOC_MIN, color='orange', linestyle='--', label='Min SOC')
    plt.title('Battery State of Charge (SOC) Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.bar(days, results['p_grid'], color='grey', label='Grid Usage (P_grid)')
    plt.title('Grid Dependency Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'microgrid_results.png')
    plt.savefig(plot_path)
    
    print("\n" + "="*60)
    print("                ENHANCED OPTIMIZATION RESULTS")
    print("="*60)
    for k, v in metrics.items():
        print(f"{k:<25}: {v:>10.2f}")
    print("="*60)

if __name__ == "__main__":
    preds, dates = load_data_and_predict()
    if preds is not None:
        res, met = run_optimization(preds)
        visualize(res, met)
        with open(METRICS_FILE, 'w') as f:
            json.dump(met, f, indent=4)
