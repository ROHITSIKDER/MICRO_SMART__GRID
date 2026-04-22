import os
import sys
os.environ['KERAS_BACKEND'] = 'torch'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import keras

# Get Project Root and add core to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'core'))

def generate_enhanced_visuals():
    print("Generating new graphs using updated mathematical formulas...")
    
    MATH_PLOTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'math-analysis')
    if not os.path.exists(MATH_PLOTS_DIR):
        os.makedirs(MATH_PLOTS_DIR)

    MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'saved_models', 'cnn_lstm_model.keras')
    DATA_PATH = os.path.join(PROJECT_ROOT, 'Data', 'final_data.csv')
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train first.")
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
    
    # Re-run the optimization logic
    from optimization_module import run_optimization
    results, metrics = run_optimization(predictions[:30])
    
    days = np.arange(1, 31)
    
    # GRAPH 1: POWER BALANCE COMPOSITION (STACKED)
    plt.figure(figsize=(12, 6))
    p_bat_discharge = [max(0, p) for p in results['p_bat']]
    
    plt.stackplot(days, 
                  results['p_solar'], 
                  results['p_wind'], 
                  results['p_biomass'], 
                  p_bat_discharge, 
                  results['p_grid'],
                  labels=['Solar', 'Wind', 'Biomass', 'Battery Discharge', 'Grid'],
                  colors=['#f1c40f', '#3498db', '#2ecc71', '#9b59b6', '#95a5a6'],
                  alpha=0.8)
    
    plt.plot(days, results['p_load'], color='black', linewidth=2, label='Demand (P_load)')
    plt.title('HRES Power Balance: Mathematical Composition (30 Days)', fontsize=14)
    plt.xlabel('Day Index')
    plt.ylabel('Power (kW)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(MATH_PLOTS_DIR, 'power_balance_stacked.png'), dpi=300)
    
    # GRAPH 2: BATTERY SOC VS COST EFFICIENCY
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Day Index')
    ax1.set_ylabel('Battery SOC (kWh)', color='#2980b9')
    ax1.plot(days, results['soc'], color='#2980b9', linewidth=3, label='SOC(t)')
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    ax1.axhline(y=500, color='r', linestyle='--', alpha=0.5, label='Max Capacity')
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Min SOC')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Daily Operational Cost (₹)', color='#c0392b')
    ax2.bar(days, results['cost'], color='#c0392b', alpha=0.3, label='Cost(t)')
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    plt.title('Battery Dynamics & Cost Impact Analysis', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(MATH_PLOTS_DIR, 'soc_cost_analysis.png'), dpi=300)

    # GRAPH 3: RENEWABLE PENETRATION RADAR
    plt.figure(figsize=(8, 6))
    total_ren = np.sum(results['p_ren_total'])
    total_load = np.sum(results['p_load'])
    penetration = (total_ren / total_load) * 100
    
    categories = ['Renewable Penetration', 'Grid Dependency', 'Load Reliability']
    values = [penetration, 100-penetration, metrics['reliability_pct']]
    colors = ['#27ae60', '#e67e22', '#2980b9']
    plt.bar(categories, values, color=colors)
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.title('System Reliability & Sustainability Metrics', fontsize=14)
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 110)
    plt.savefig(os.path.join(MATH_PLOTS_DIR, 'sustainability_metrics.png'), dpi=300)

    # CREATE SUMMARY TABLE
    SUMMARY_PATH = os.path.join(PROJECT_ROOT, 'reports', 'math_performance_summary.md')
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write("# Micro-Smart-Grid Mathematical Performance Report\n\n")
        f.write("## 1. Key Optimization Metrics\n")
        f.write("| Metric | Formula | Value |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| Total Cost | $\\sum (C_{{grid}}P_{{grid}} + C_{{bat}}P_{{bat}})$ | ₹{metrics['total_cost']:,.2f} |\n")
        f.write(f"| Cost Savings | $(Cost_{{base}} - Cost_{{opt}}) / Cost_{{base}}$ | {metrics['savings_pct']:.2f}% |\n")
        f.write(f"| Renewable Penetration | $P_{{ren}} / P_{{load}}$ | {metrics['ren_penetration_pct']:.2f}% |\n")
        f.write(f"| Load Reliability | $Served Load / Total Load$ | {metrics['reliability_pct']:.2f}% |\n\n")
        
        f.write("## 2. Forecasting Model Comparison\n")
        METRICS_PATH = os.path.join(PROJECT_ROOT, 'model', 'metrics.json')
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as mfile:
                m = json.load(mfile)
            f.write("| Model | RMSE | MAPE | R² Score |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            f.write(f"| CNN-LSTM | {m['cnn_lstm']['overall']['rmse']:.4f} | {m['cnn_lstm']['overall']['mape']:.2f}% | {m['cnn_lstm']['overall']['r2']:.4f} |\n")
            f.write(f"| LSTM | {m['lstm']['overall']['rmse']:.4f} | {m['lstm']['overall']['mape']:.2f}% | {m['lstm']['overall']['r2']:.4f} |\n")

    print(f"Mathematical plots and report generated in '{MATH_PLOTS_DIR}' and 'reports/'")

if __name__ == "__main__":
    generate_enhanced_visuals()
