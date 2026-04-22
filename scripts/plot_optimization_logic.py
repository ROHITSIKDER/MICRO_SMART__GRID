import os
os.environ['KERAS_BACKEND'] = 'torch'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

def plot_optimization_formulation():
    print("Generating Optimization Problem Formulation Graph...")
    
    # 1. Load Data and Run Optimization to get time-series
    MODEL_PATH = 'model/saved_models/cnn_lstm_model.keras'
    DATA_PATH = 'Data/final_data.csv'
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
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
    
    from optimization_module import run_optimization, get_grid_cost
    results, metrics = run_optimization(predictions[:24]) # 24-hour snapshot for clarity
    
    hours = np.arange(24)
    grid_prices = [get_grid_cost(h % 24) for h in hours]
    
    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Subplot 1: Objective Function Input (Grid Price)
    ax1.step(hours, grid_prices, where='post', color='#e74c3c', linewidth=2.5, label='Grid Price ($C_{grid}$)')
    ax1.set_ylabel('Price (₹/kWh)', fontweight='bold')
    ax1.set_title('Optimization Logic: Mathematical Formulation Components', fontsize=16, fontweight='bold')
    ax1.fill_between(hours, grid_prices, step="post", alpha=0.2, color='#e74c3c')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 15)
    
    # Subplot 2: Decision Variable (Battery Power P_bat)
    # Positive = Discharge (Saving money), Negative = Charge (Using surplus)
    p_bat = results['p_bat']
    ax2.bar(hours, p_bat, color=['#27ae60' if x > 0 else '#2980b9' for x in p_bat], alpha=0.7, label='Battery Power ($P_{bat}$)')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel('Power (kW)', fontweight='bold')
    ax2.text(0.5, 80, 'Discharging (Minimizing Cost)', color='#27ae60', fontweight='bold')
    ax2.text(0.5, -80, 'Charging (Storing Surplus)', color='#2980b9', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Constraint Satisfaction (Power Balance)
    # P_load = P_ren + P_grid + P_bat
    ax3.plot(hours, results['p_load'], 'k--', label='Demand ($P_{load}$)', linewidth=2)
    ax3.plot(hours, results['p_ren_total'], 'g-', label='Renewable ($P_{ren}$)', linewidth=2)
    ax3.fill_between(hours, results['p_load'], results['p_ren_total'], color='grey', alpha=0.1, label='Net Deficit/Surplus')
    ax3.set_ylabel('Energy (kWh)', fontweight='bold')
    ax3.set_xlabel('Hour of the Day', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.xticks(hours)
    plt.tight_layout()
    plt.savefig('optimization_formulation_graph.png', dpi=300)
    print("Graph saved as 'optimization_formulation_graph.png'")

if __name__ == "__main__":
    plot_optimization_formulation()
