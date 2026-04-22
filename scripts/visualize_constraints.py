import os
import sys
os.environ['KERAS_BACKEND'] = 'torch'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

# Setup Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'core'))

def visualize_strict_constraints():
    print("Generating Constraint Verification Graphs...")
    
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'saved_models', 'cnn_lstm_model.keras')
    DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'final_data.csv')
    
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
    
    days = np.arange(1, 31)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # 1. POWER BALANCE CONSTRAINT: P_load = P_ren + P_bat + P_grid
    # We plot the deviation to show it is exactly zero.
    p_ren = np.array(results['p_ren_total'])
    p_bat = np.array(results['p_bat']) # Discharging is positive, Charging is negative
    p_grid = np.array(results['p_grid'])
    p_load = np.array(results['p_load'])
    
    balance_deviation = p_load - (p_ren + p_bat + p_grid)
    
    ax1.plot(days, balance_deviation, 'g-', linewidth=2, label='Power Mismatch ($P_{load} - \sum P_{sources}$)')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Constraint 1: Power Balance Verification (Target = 0)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Deviation (kW)')
    ax1.set_ylim(-1, 1) # Show that it's nearly perfectly zero
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. BATTERY CONSTRAINTS: SOC and Power Limits
    # SOC_min <= SOC <= SOC_max
    # |P_bat| <= P_bat_max
    ax2_soc = ax2.twinx()
    ax2.bar(days, np.abs(p_bat), color='#9b59b6', alpha=0.3, label='|Battery Power| ($P_{bat}$)')
    ax2.axhline(100, color='#8e44ad', linestyle='--', label='Max Power Limit (100kW)')
    
    ax2_soc.plot(days, results['soc'], color='#2980b9', linewidth=2.5, label='SOC(t)')
    ax2_soc.axhline(500, color='#c0392b', linestyle=':', label='Max Cap (500kWh)')
    ax2_soc.axhline(50, color='#e67e22', linestyle=':', label='Min SOC (50kWh)')
    
    ax2.set_ylabel('Power (kW)', fontweight='bold')
    ax2_soc.set_ylabel('SOC (kWh)', fontweight='bold')
    ax2.set_title('Constraint 2: Battery SOC & Power Limits', fontsize=14, fontweight='bold')
    
    # Handle Legend for dual axis
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_soc.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax2.grid(True, alpha=0.3)

    # 3. GENERATOR / RENEWABLE CONSTRAINTS: P_ren_used <= P_ren_available
    # Available renewable is what we predicted. Used is what we actually took.
    ax3.plot(days, p_ren, 'g-', linewidth=2.5, label='Available Renewable ($P_{available}$)')
    # In this greedy model, we always use available first, then store surplus.
    # So used ren = p_ren. Let's visualize the "Load Served by Renewables" vs "Total Available"
    p_ren_to_load = np.minimum(p_ren, p_load)
    ax3.fill_between(days, p_ren, p_ren_to_load, color='#2ecc71', alpha=0.3, label='Surplus (Sent to Battery)')
    ax3.plot(days, p_ren_to_load, color='#27ae60', linestyle='--', label='Used for Load ($P_{used}$)')
    
    ax3.set_title('Constraint 3: Generator/Renewable Capacity Limits', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Power (kW)')
    ax3.set_xlabel('Day Index')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    OUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'math-analysis', 'mathematical_constraints_verification.png')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Constraint verification graph saved to: {OUT_PATH}")

if __name__ == "__main__":
    visualize_strict_constraints()
