import pandas as pd
import numpy as np

# Load last 1 week of data (168 hours)
df = pd.read_csv('Data/final_data.csv').tail(168)

# Recreate renewable generation (consistent with generate_weekly_analysis.py)
solar = df['solar_irradiance'] * 100
wind = df['wind_speed'] * 50
biogas = df['biomass_energy'] * 30
total_ren = solar + wind + biogas

# Recreate load demand (consistent with generate_weekly_analysis.py)
load_base = 120
time_idx = np.arange(168)
load = load_base + 50 * np.sin(2 * np.pi * (time_idx - 6) / 24) + \
       40 * np.sin(4 * np.pi * (time_idx - 17) / 24)
load = np.maximum(40, load)
total_load_kwh = np.sum(load)
cost_grid_only = total_load_kwh * 8

# Sensitivity Scenarios: Battery Capacity
capacities = [250, 500, 750, 1000, 1500]
sensitivity_results = []

for cap in capacities:
    soc = 200 # Starting SOC
    total_grid_purchase = 0
    for i in range(168):
        gen = total_ren.iloc[i]
        dem = load[i]
        net = gen - dem
        
        if net >= 0:
            charge = min(net, cap - soc)
            soc += charge
        else:
            deficit = abs(net)
            discharge = min(deficit, soc)
            soc -= discharge
            total_grid_purchase += (deficit - discharge)
            
    total_cost = total_grid_purchase * 8
    reduction = (1 - total_cost / cost_grid_only) * 100
    sensitivity_results.append({
        'Capacity': cap,
        'Grid_kWh': total_grid_purchase,
        'Cost_INR': total_cost,
        'Reduction_Pct': reduction
    })

# Format Table
header = f"{'Battery Capacity (kWh)':<25} | {'Grid Purchase (kWh)':<20} | {'Total Cost (₹)':<15} | {'Cost Reduction (%)':<20}"
separator = "-" * 88
rows = []
for r in sensitivity_results:
    row = f"{r['Capacity']:<25} | {r['Grid_kWh']:<20.2f} | {r['Cost_INR']:<15.2f} | {r['Reduction_Pct']:<20.2f} %"
    rows.append(row)

# Update report
table_content = f"\nTABLE 4: SENSITIVITY ANALYSIS (BATTERY CAPACITY)\n{separator}\n{header}\n{separator}\n" + "\n".join(rows) + f"\n{separator}\n"

with open('weekly_analysis_report.txt', 'a', encoding='utf-8') as f:
    f.write(table_content)

print("Sensitivity analysis added to weekly_analysis_report.txt")
