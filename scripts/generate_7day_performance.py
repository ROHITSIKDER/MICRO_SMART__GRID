import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

def generate_7day_data():
    """Generates 7 days (168 hours) of microgrid performance data."""
    hours = 168
    time = np.arange(hours)
    
    # 1. Solar Generation (Diurnal cycle)
    # Peak at 12:00 (midday)
    solar_base = np.maximum(0, 100 * np.sin(2 * np.pi * (time % 24) / 24 - np.pi/2) + 40)
    solar = solar_base + np.random.normal(0, 5, hours)
    solar = np.maximum(0, solar)
    
    # 2. Wind Generation (More stochastic, slight nightly peak)
    wind_base = 30 + 10 * np.sin(2 * np.pi * (time % 24) / 24 + np.pi/4)
    wind = wind_base + np.random.normal(0, 15, hours)
    wind = np.maximum(5, wind)
    
    # 3. Biogas Generation (Relatively steady base load)
    biogas = 20 + np.random.normal(0, 2, hours)
    biogas = np.maximum(15, biogas)
    
    total_renewable = solar + wind + biogas
    
    # 4. Load Demand (Double peak: 8 AM and 7 PM)
    # Morning peak (t=8), Evening peak (t=19)
    demand_base = 120 + 50 * np.sin(2 * np.pi * (time % 24) / 24 - np.pi/3) + \
                  40 * np.sin(4 * np.pi * (time % 24) / 24 - np.pi/2)
    demand = demand_base + np.random.normal(0, 10, hours)
    demand = np.maximum(40, demand)
    
    return time, solar, wind, biogas, total_renewable, demand

def simulate_microgrid(renewable, demand):
    BATTERY_CAPACITY = 500.0
    INITIAL_BATTERY_LEVEL = 200.0
    
    n = len(renewable)
    battery_levels = np.zeros(n + 1)
    battery_levels[0] = INITIAL_BATTERY_LEVEL
    
    grid_usage = np.zeros(n)
    battery_charge = np.zeros(n)
    battery_discharge = np.zeros(n)
    
    for t in range(n):
        current_soc = battery_levels[t]
        
        if renewable[t] >= demand[t]:
            # Surplus energy
            surplus = renewable[t] - demand[t]
            charge = min(surplus, BATTERY_CAPACITY - current_soc)
            battery_levels[t+1] = current_soc + charge
            battery_charge[t] = charge
            grid_usage[t] = 0
        else:
            # Deficit energy
            deficit = demand[t] - renewable[t]
            discharge = min(deficit, current_soc)
            battery_levels[t+1] = current_soc - discharge
            battery_discharge[t] = discharge
            grid_usage[t] = deficit - discharge
            
    return grid_usage, battery_levels[:-1]

def main():
    # Generate Data
    time, solar, wind, biogas, total_renewable, demand = generate_7day_data()
    
    # Run Simulation
    grid_usage, battery_levels = simulate_microgrid(total_renewable, demand)
    
    # 1. Graph (7 days)
    plt.figure(figsize=(15, 8))
    plt.plot(time, total_renewable, label='Total Renewable (Solar+Wind+Biogas)', color='green', linewidth=2)
    plt.plot(time, demand, label='Total Load Demand', color='red', linestyle='--', linewidth=2)
    plt.plot(time, grid_usage, label='Grid Usage', color='blue', alpha=0.7)
    
    # Styling
    plt.title('7-Day Microgrid Performance: Energy Balance', fontsize=16, fontweight='bold')
    plt.xlabel('Time (Hours)', fontsize=12)
    plt.ylabel('Energy (kWh)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, 169, 24), ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8'])
    
    plt.tight_layout()
    plt.savefig('performance_7days.png', dpi=300)
    print("Graph saved as 'performance_7days.png'")
    
    # 2. Table: Day | Renewable (kWh) | Load (kWh) | Grid (kWh)
    daily_data = []
    for day in range(7):
        start = day * 24
        end = (day + 1) * 24
        daily_ren = np.sum(total_renewable[start:end])
        daily_load = np.sum(demand[start:end])
        daily_grid = np.sum(grid_usage[start:end])
        daily_data.append([f"Day {day+1}", daily_ren, daily_load, daily_grid])
    
    df_daily = pd.DataFrame(daily_data, columns=['Day', 'Renewable (kWh)', 'Load (kWh)', 'Grid (kWh)'])
    
    print("\n7-Day Microgrid Performance Table:")
    print("=" * 55)
    print(f"{'Day':<8} | {'Renewable (kWh)':<15} | {'Load (kWh)':<10} | {'Grid (kWh)':<10}")
    print("-" * 55)
    for index, row in df_daily.iterrows():
        print(f"{row['Day']:<8} | {row['Renewable (kWh)']:<15.2f} | {row['Load (kWh)']:<10.2f} | {row['Grid (kWh)']:<10.2f}")
    print("-" * 55)
    total_ren = df_daily['Renewable (kWh)'].sum()
    total_load = df_daily['Load (kWh)'].sum()
    total_grid = df_daily['Grid (kWh)'].sum()
    print(f"{'TOTAL':<8} | {total_ren:<15.2f} | {total_load:<10.2f} | {total_grid:<10.2f}")
    print("=" * 55)

if __name__ == "__main__":
    main()
