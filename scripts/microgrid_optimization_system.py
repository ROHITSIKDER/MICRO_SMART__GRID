import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

def generate_data(n_points=72):
    """Generates realistic renewable energy and demand data."""
    time = np.arange(n_points)
    
    # Renewable Energy: 24-hour cycle (solar-like) with noise
    solar_base = np.maximum(0, 150 * np.sin(2 * np.pi * time / 24 - np.pi/2) + 50)
    renewable_energy = solar_base + np.random.normal(0, 10, n_points)
    renewable_energy = np.maximum(0, renewable_energy)
    
    # Demand: Double peak (morning and evening) with noise
    demand_base = 80 + 40 * np.sin(2 * np.pi * time / 24) + 30 * np.sin(4 * np.pi * time / 24)
    demand = demand_base + np.random.normal(0, 15, n_points)
    demand = np.maximum(20, demand)
    
    return time, renewable_energy, demand

def simulate_microgrid(renewable, demand):
    # Constants
    BATTERY_CAPACITY = 500.0
    INITIAL_BATTERY_LEVEL = 100.0
    GRID_COST_PER_KWH = 8.0
    BATTERY_COST_PER_KWH = 2.0
    RENEWABLE_COST_PER_KWH = 0.0
    CO2_EMISSION_FACTOR = 0.82 # kg CO2 per kWh of grid energy
    
    n = len(renewable)
    sim_days = n / 24.0
    
    # Tracking arrays
    battery_levels = np.zeros(n + 1)
    battery_levels[0] = INITIAL_BATTERY_LEVEL
    
    renewable_used = np.zeros(n)
    battery_discharged = np.zeros(n)
    grid_used = np.zeros(n)
    
    # Logic with Optimization
    for t in range(n):
        current_battery = battery_levels[t]
        
        if renewable[t] >= demand[t]:
            renewable_used[t] = demand[t]
            extra_energy = renewable[t] - demand[t]
            charge_amount = min(extra_energy, BATTERY_CAPACITY - current_battery)
            battery_levels[t+1] = current_battery + charge_amount
            battery_discharged[t] = 0
            grid_used[t] = 0
        else:
            renewable_used[t] = renewable[t]
            deficit = demand[t] - renewable[t]
            discharge_amount = min(deficit, current_battery)
            battery_discharged[t] = discharge_amount
            battery_levels[t+1] = current_battery - discharge_amount
            grid_used[t] = deficit - discharge_amount
            
    # Calculate Results WITH Optimization
    total_renewable = np.sum(renewable_used)
    total_battery = np.sum(battery_discharged)
    total_grid_opt = np.sum(grid_used)
    
    cost_with_opt = (total_renewable * RENEWABLE_COST_PER_KWH + 
                     total_battery * BATTERY_COST_PER_KWH + 
                     total_grid_opt * GRID_COST_PER_KWH)
    
    # Logic WITHOUT Optimization
    grid_used_no_opt = np.maximum(0, demand - renewable)
    total_grid_no_opt = np.sum(grid_used_no_opt)
    cost_without_opt = total_grid_no_opt * GRID_COST_PER_KWH
    
    # Savings Calculations
    total_savings = cost_without_opt - cost_with_opt
    daily_savings = total_savings / sim_days
    monthly_savings = daily_savings * 30
    yearly_savings = daily_savings * 365
    
    # Grid and Efficiency Calculations
    grid_reduction_pct = ((total_grid_no_opt - total_grid_opt) / total_grid_no_opt * 100) if total_grid_no_opt > 0 else 0
    efficiency_improvement = (total_savings / cost_without_opt * 100) if cost_without_opt > 0 else 0
    
    # Environmental Impact
    grid_saved_kwh = total_grid_no_opt - total_grid_opt
    co2_reduction = grid_saved_kwh * CO2_EMISSION_FACTOR
    
    return {
        "battery_levels": battery_levels[:-1],
        "renewable_used": total_renewable,
        "battery_used": total_battery,
        "grid_used_opt": total_grid_opt,
        "grid_used_no_opt": total_grid_no_opt,
        "cost_with_opt": cost_with_opt,
        "cost_without_opt": cost_without_opt,
        "daily_savings": daily_savings,
        "monthly_savings": monthly_savings,
        "yearly_savings": yearly_savings,
        "grid_reduction_pct": grid_reduction_pct,
        "efficiency_improvement": efficiency_improvement,
        "co2_reduction": co2_reduction
    }

def plot_results(time, renewable, demand, results):
    plt.figure(figsize=(12, 16))
    plt.rcParams.update({'font.size': 10})
    
    # Graph 1: Predicted Energy vs Demand
    plt.subplot(3, 1, 1)
    plt.plot(time, renewable, label='Predicted Renewable Energy (kWh)', color='green', linewidth=2)
    plt.plot(time, demand, label='Energy Demand (kWh)', color='red', linestyle='--', linewidth=2)
    plt.fill_between(time, renewable, demand, where=(renewable > demand), color='green', alpha=0.2, label='Excess (Charging)')
    plt.fill_between(time, renewable, demand, where=(renewable < demand), color='red', alpha=0.2, label='Deficit (Discharging/Grid)')
    plt.title('Microgrid Energy Balance: Supply vs Demand', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Hours)', fontweight='bold')
    plt.ylabel('Energy (kWh)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graph 2: Cost Comparison
    plt.subplot(3, 1, 2)
    categories = ['Without Optimization', 'With Optimization']
    costs = [results['cost_without_opt'], results['cost_with_opt']]
    bars = plt.bar(categories, costs, color=['grey', 'royalblue'], width=0.5)
    plt.title('Economic Analysis: Cost Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Total Cost (₹)', fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5, f'₹{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Graph 3: Battery level
    plt.subplot(3, 1, 3)
    plt.plot(time, results['battery_levels'], label='Battery Level (kWh)', color='orange', linewidth=2.5)
    plt.axhline(y=500, color='darkred', linestyle=':', label='Max Capacity (500kWh)')
    plt.fill_between(time, results['battery_levels'], color='orange', alpha=0.1)
    plt.title('Energy Storage System: Battery State of Charge (SoC)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Hours)', fontweight='bold')
    plt.ylabel('Storage Level (kWh)', fontweight='bold')
    plt.ylim(-20, 550)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('microgrid_results.png', dpi=300)

def main():
    # 1. Generate Data
    time, renewable, demand = generate_data(72) # 3 days
    
    # 2. Run Simulation
    results = simulate_microgrid(renewable, demand)
    
    # 3. Clean Final Report
    print("="*60)
    print("           ADVANCED MICROGRID OPTIMIZATION REPORT")
    print("="*60)
    
    print(f"{'COST ANALYSIS':<30}")
    print(f"Cost Without Optimization:    ₹{results['cost_without_opt']:>10.2f}")
    print(f"Cost With Optimization:       ₹{results['cost_with_opt']:>10.2f}")
    print("-" * 60)
    
    print(f"{'SAVINGS PROJECTIONS':<30}")
    print(f"Daily Savings:                ₹{results['daily_savings']:>10.2f}")
    print(f"Monthly Savings (30 days):    ₹{results['monthly_savings']:>10.2f}")
    print(f"Yearly Savings (365 days):    ₹{results['yearly_savings']:>10.2f}")
    print("-" * 60)
    
    print(f"{'PERFORMANCE METRICS':<30}")
    print(f"Efficiency Improvement:       {results['efficiency_improvement']:>10.2f} %")
    print(f"Grid Dependency Reduction:    {results['grid_reduction_pct']:>10.2f} %")
    print("-" * 60)
    
    print(f"{'ENVIRONMENTAL IMPACT':<30}")
    print(f"Total CO2 Reduction:          {results['co2_reduction']:>10.2f} kg")
    print(f"Estimated Yearly CO2 Saved:   {results['co2_reduction'] / 3 * 365:>10.2f} kg")
    print("="*60)
    
    # 4. Generate Graphs
    plot_results(time, renewable, demand, results)
    print("\nVisual analytics generated and saved to 'microgrid_results.png'.")

if __name__ == "__main__":
    main()
