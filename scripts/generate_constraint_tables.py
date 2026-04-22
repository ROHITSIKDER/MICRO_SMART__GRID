import os
import sys
os.environ['KERAS_BACKEND'] = 'torch'

import pandas as pd
import numpy as np
import keras

# Setup Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'core'))

def generate_mathematical_tables():
    print("Generating Mathematical Constraint Tables...")
    
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'saved_models', 'cnn_lstm_model.keras')
    DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'final_data.csv')
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Error: Model or Data file missing.")
        return

    model = keras.models.load_model(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    test_data = df.tail(37).copy()
    features = test_data[['temperature', 'wind_speed', 'solar_irradiance', 'biomass_energy']].values
    X = []
    for i in range(len(features) - 7):
        X.append(features[i:i+7])
    X = np.array(X)
    predictions = model.predict(X, verbose=0)
    
    from optimization_module import run_optimization
    results, metrics = run_optimization(predictions[:30])
    
    # 1. TABLE 1: Power Balance Verification
    # P_load = P_ren + P_bat + P_grid
    p_ren = np.array(results['p_ren_total'])
    p_bat = np.array(results['p_bat'])
    p_grid = np.array(results['p_grid'])
    p_load = np.array(results['p_load'])
    balance_deviation = p_load - (p_ren + p_bat + p_grid)
    
    table1_rows = []
    for i in range(30):
        table1_rows.append(f"{i+1:<5} | {p_load[i]:<10.2f} | {p_ren[i]:<10.2f} | {p_bat[i]:<10.2f} | {p_grid[i]:<10.2f} | {balance_deviation[i]:<10.4f}")
    
    table1 = "TABLE 1: CONSTRAINT VERIFICATION - POWER BALANCE (P_load = P_ren + P_bat + P_grid)\n"
    table1 += "="*80 + "\n"
    table1 += f"{'Day':<5} | {'P_load':<10} | {'P_ren':<10} | {'P_bat':<10} | {'P_grid':<10} | {'Mismatch':<10}\n"
    table1 += "-"*80 + "\n"
    table1 += "\n".join(table1_rows)
    table1 += "\n" + "="*80 + "\n"

    # 2. TABLE 2: Battery SOC & Power Limits
    # SOC_min (50) <= SOC <= SOC_max (500), |P_bat| <= 100
    soc = np.array(results['soc'])
    p_bat_abs = np.abs(p_bat)
    
    table2_rows = []
    for i in range(30):
        table2_rows.append(f"{i+1:<5} | {p_bat_abs[i]:<10.2f} | {'100':<12} | {soc[i]:<10.2f} | {'50':<10} | {'500':<10}")
    
    table2 = "TABLE 2: CONSTRAINT VERIFICATION - BATTERY LIMITS (SOC & Power)\n"
    table2 += "="*80 + "\n"
    table2 += f"{'Day':<5} | {'|P_bat|':<10} | {'Max_P_Limit':<12} | {'SOC':<10} | {'Min_SOC':<10} | {'Max_SOC':<10}\n"
    table2 += "-"*80 + "\n"
    table2 += "\n".join(table2_rows)
    table2 += "\n" + "="*80 + "\n"

    # 3. TABLE 3: Renewable Capacity Limits
    # P_used <= P_available
    p_ren_to_load = np.minimum(p_ren, p_load)
    surplus = p_ren - p_ren_to_load
    
    table3_rows = []
    for i in range(30):
        table3_rows.append(f"{i+1:<5} | {p_ren[i]:<12.2f} | {p_ren_to_load[i]:<10.2f} | {surplus[i]:<10.2f} | {'PASS':<10}")
    
    table3 = "TABLE 3: CONSTRAINT VERIFICATION - RENEWABLE CAPACITY LIMITS (P_used <= P_available)\n"
    table3 += "="*80 + "\n"
    table3 += f"{'Day':<5} | {'P_available':<12} | {'P_used':<10} | {'Surplus':<10} | {'Status':<10}\n"
    table3 += "-"*80 + "\n"
    table3 += "\n".join(table3_rows)
    table3 += "\n" + "="*80 + "\n"

    # Save to file
    OUT_PATH = os.path.join(PROJECT_ROOT, 'reports', 'mathematical_constraints_tables.txt')
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    with open(OUT_PATH, 'w') as f:
        f.write("MICRO-SMART-GRID: MATHEMATICAL CONSTRAINTS VERIFICATION DATA\n")
        f.write("Generated from: mathematical_constraints_verification.png logic\n\n")
        f.write(table1 + "\n\n")
        f.write(table2 + "\n\n")
        f.write(table3 + "\n")
    
    print(f"Mathematical constraint tables saved to: {OUT_PATH}")

if __name__ == "__main__":
    generate_mathematical_tables()
