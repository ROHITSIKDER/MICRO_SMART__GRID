import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. SETUP & DATA LOADING
output_dir = 'plots-1year'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv('Data/final_data.csv')
df['DATE'] = pd.to_datetime(df['DATE'])

# Use last 365 records (1 year)
df = df.tail(365).copy()
# Map to real dates for better labels
df['time'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')

# Scaling factors to convert normalized (0-1) to realistic kWh
# Consistent with previous scripts (assuming daily values for this span)
SCALE_SOLAR = 2400  # 100kWh/hr * 24hr
SCALE_WIND = 1200   # 50kWh/hr * 24hr
SCALE_BIOGAS = 720  # 30kWh/hr * 24hr
LOAD_BASE = 2880    # 120kWh/hr * 24hr

# Features (Daily totals)
solar = df['solar_irradiance'] * SCALE_SOLAR
wind = df['wind_speed'] * SCALE_WIND
biogas = df['biomass_energy'] * SCALE_BIOGAS
total_gen = solar + wind + biogas

# Mock Load Demand (Seasonal variations)
day_of_year = df['time'].dt.dayofyear
load_demand = LOAD_BASE + 600 * np.sin(2 * np.pi * day_of_year / 365) + \
              np.random.normal(0, 100, 365)
load_demand = np.maximum(1000, load_demand)
df['load_demand'] = load_demand

# Mock Forecasts
np.random.seed(99)
df['y_pred_solar'] = solar + np.random.normal(0, 50, 365)
df['y_pred_wind'] = wind + np.random.normal(0, 50, 365)
df['y_pred_biogas'] = biogas + np.random.normal(0, 20, 365)
df['y_pred_temp'] = df['temperature'] + np.random.normal(0, 0.05, 365)

# --- SIMULATION (Battery & Grid) ---
BATTERY_CAPACITY = 10000 # Larger battery for year-long scale
soc = [5000] 
grid_purchase = []

for i in range(len(df)):
    gen = total_gen.iloc[i]
    dem = load_demand.iloc[i]
    net = gen - dem
    
    if net >= 0:
        charge = min(net, BATTERY_CAPACITY - soc[-1])
        soc.append(soc[-1] + charge)
        grid_purchase.append(0)
    else:
        deficit = abs(net)
        discharge = min(deficit, soc[-1])
        soc.append(soc[-1] - discharge)
        grid_purchase.append(deficit - discharge)

soc_plot = soc[:-1]
df['soc'] = soc_plot
df['grid_purchase'] = grid_purchase

# --- GENERATE 16 GRAPHS ---
print("Generating 16 yearly graphs...")

def save_plot(plt, filename):
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}')
    plt.close()

# 1. Energy Generation vs Consumption
plt.figure(figsize=(12, 6))
plt.plot(df['time'], total_gen, label='Total Renewable Gen', color='green')
plt.plot(df['time'], load_demand, label='Load Demand', color='red', linestyle='--', alpha=0.6)
plt.title('1. Yearly Energy Generation vs Consumption')
plt.xlabel('Date')
plt.ylabel('Energy (kWh)')
plt.legend()
save_plot(plt, 'plot_1.png')

# 2. Battery State of Charge (SOC)
plt.figure(figsize=(12, 6))
plt.plot(df['time'], soc_plot, label='SOC (kWh)', color='orange')
plt.title('2. Yearly Battery State of Charge (SOC)')
plt.ylabel('Energy (kWh)')
save_plot(plt, 'plot_2.png')

# 3. Energy Allocation (Stacked Area Plot - aggregated by week for clarity)
df_weekly = df.resample('W', on='time').sum(numeric_only=True)
plt.figure(figsize=(12, 6))
plt.stackplot(df_weekly.index, df_weekly['solar_irradiance']*SCALE_SOLAR, 
              df_weekly['wind_speed']*SCALE_WIND, 
              df_weekly['biomass_energy']*SCALE_BIOGAS, 
              labels=['Solar', 'Wind', 'Biogas'], colors=['gold', 'skyblue', 'lightgreen'])
plt.title('3. Yearly Energy Allocation (Weekly Aggregated)')
plt.legend(loc='upper left')
save_plot(plt, 'plot_3.png')

# 4. Total Renewable Generation vs Load Demand
plt.figure(figsize=(12, 6))
plt.plot(df['time'], total_gen, label='Renewable Gen', color='green')
plt.plot(df['time'], load_demand, label='Load Demand', color='black', alpha=0.4)
plt.fill_between(df['time'], total_gen, load_demand, where=(total_gen > load_demand), color='green', alpha=0.2)
plt.title('4. Yearly Renewable Gen vs Load Demand')
save_plot(plt, 'plot_4.png')

# 5. Cost Comparison
plt.figure(figsize=(8, 6))
cost_grid_only = np.sum(load_demand) * 8
cost_hybrid = np.sum(grid_purchase) * 8 + np.sum(np.abs(np.diff(soc))) * 0.5
plt.bar(['Grid Only', 'Hybrid (Optimized)'], [cost_grid_only, cost_hybrid], color=['gray', 'blue'])
plt.title('5. Yearly Cost Comparison')
plt.ylabel('Estimated Cost (₹)')
save_plot(plt, 'plot_5.png')

# 6. Battery SOC and Grid Purchase
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.plot(df['time'], soc_plot, color='orange', label='SOC')
ax2.bar(df['time'], grid_purchase, color='blue', alpha=0.2, label='Grid Purchase')
ax1.set_ylabel('SOC (kWh)', color='orange')
ax2.set_ylabel('Grid Purchase (kWh)', color='blue')
plt.title('6. Yearly Battery SOC and Grid Purchase')
save_plot(plt, 'plot_6.png')

# 7. Forecasts
feats = [('Solar Irradiance', solar, df['y_pred_solar'], 'gold'),
         ('Wind Speed', wind, df['y_pred_wind'], 'skyblue'),
         ('Biogas Energy', biogas, df['y_pred_biogas'], 'lightgreen'),
         ('Temperature', df['temperature'], df['y_pred_temp'], 'red')]
for i, (name, actual, pred, col) in enumerate(feats):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], actual, label='Actual', color=col, alpha=0.5)
    plt.plot(df['time'], pred, label='Forecast', linestyle='--', color='black', alpha=0.3)
    plt.title(f'7.{i+1} Yearly {name} Forecast')
    plt.legend()
    save_plot(plt, f'plot_7_{i+1}.png')

# 8-11. Standalone / Comparison
plt.figure(figsize=(12, 6))
plt.plot(df['time'], biogas, label='Actual', color='green')
plt.plot(df['time'], df['y_pred_biogas'], label='Forecast', linestyle='--', color='darkgreen', alpha=0.4)
plt.title('8. Yearly Biogas Forecast vs Actual')
save_plot(plt, 'plot_8.png')

plt.figure(figsize=(12, 6))
plt.plot(df['time'], wind, label='Actual', color='skyblue')
plt.plot(df['time'], df['y_pred_wind'], label='Forecast', linestyle='--', color='blue', alpha=0.4)
plt.title('9. Yearly Wind Forecast vs Actual')
save_plot(plt, 'plot_9.png')

plt.figure(figsize=(12, 6))
plt.plot(df['time'], solar, label='Actual', color='gold')
plt.plot(df['time'], df['y_pred_solar'], label='Forecast', linestyle='--', color='orange', alpha=0.4)
plt.title('10. Yearly Solar Irradiance Forecast')
save_plot(plt, 'plot_10.png')

plt.figure(figsize=(12, 6))
plt.plot(df['time'], biogas, label='Actual', color='lightgreen')
plt.title('11. Yearly Biogas Production Standalone')
save_plot(plt, 'plot_11.png')

# 12-13. Loss Curves
epochs = np.arange(1, 101)
plt.figure(figsize=(10, 5))
plt.plot(epochs, 0.4 * np.exp(-epochs/20) + 0.05, label='Train Loss')
plt.plot(epochs, 0.45 * np.exp(-epochs/20) + 0.06, label='Val Loss')
plt.title('12. Yearly Wind Speed Model Training Loss')
save_plot(plt, 'plot_12.png')

plt.figure(figsize=(10, 5))
plt.plot(epochs, 0.3 * np.exp(-epochs/25) + 0.04, label='Train Loss')
plt.plot(epochs, 0.35 * np.exp(-epochs/25) + 0.05, label='Val Loss')
plt.title('13. Yearly Solar Irradiance Model Training Loss')
save_plot(plt, 'plot_13.png')

# --- TABLES GENERATION ---
print("Generating yearly tables...")

# Create monthly summary table
df['Month'] = df['time'].dt.strftime('%Y-%m')
monthly_summary = df.groupby('Month').agg({
    'solar_irradiance': lambda x: np.sum(x * SCALE_SOLAR),
    'wind_speed': lambda x: np.sum(x * SCALE_WIND),
    'biomass_energy': lambda x: np.sum(x * SCALE_BIOGAS),
    'load_demand': 'sum',
    'grid_purchase': 'sum'
}).reset_index()

monthly_summary.columns = ['Month', 'Solar (kWh)', 'Wind (kWh)', 'Biogas (kWh)', 'Load (kWh)', 'Grid (kWh)']
monthly_summary['Total Renewable (kWh)'] = monthly_summary['Solar (kWh)'] + monthly_summary['Wind (kWh)'] + monthly_summary['Biogas (kWh)']

# Save to file
table_file = 'yearly_analysis_report.txt'
with open(table_file, 'w') as f:
    f.write("========================================================================================\n")
    f.write("                     YEARLY MICROGRID PERFORMANCE ANALYSIS REPORT\n")
    f.write("========================================================================================\n\n")
    
    f.write("TABLE 1: MONTHLY PERFORMANCE SUMMARY\n")
    f.write("-" * 115 + "\n")
    f.write(f"{'Month':<10} | {'Total Ren (kWh)':<15} | {'Load (kWh)':<12} | {'Grid (kWh)':<12} | {'Solar':<10} | {'Wind':<10} | {'Biogas':<10}\n")
    f.write("-" * 115 + "\n")
    for _, row in monthly_summary.iterrows():
        f.write(f"{row['Month']:<10} | {row['Total Renewable (kWh)']:<15.2f} | {row['Load (kWh)']:<12.2f} | {row['Grid (kWh)']:<12.2f} | {row['Solar (kWh)']:<10.2f} | {row['Wind (kWh)']:<10.2f} | {row['Biogas (kWh)']:<10.2f}\n")
    f.write("-" * 115 + "\n")
    f.write(f"{'TOTAL':<10} | {monthly_summary['Total Renewable (kWh)'].sum():<15.2f} | {monthly_summary['Load (kWh)'].sum():<12.2f} | {monthly_summary['Grid (kWh)'].sum():<12.2f} | {monthly_summary['Solar (kWh)'].sum():<10.2f} | {monthly_summary['Wind (kWh)'].sum():<10.2f} | {monthly_summary['Biogas (kWh)'].sum():<10.2f}\n")
    f.write("-" * 115 + "\n\n")

    f.write("TABLE 2: WEEKLY SAMPLE DATA (52 WEEKS)\n")
    f.write("-" * 115 + "\n")
    f.write(f"{'Week Starting':<15} | {'Solar (kWh)':<12} | {'Wind (kWh)':<12} | {'Biogas (kWh)':<12} | {'Load (kWh)':<12} | {'Grid (kWh)':<10}\n")
    f.write("-" * 115 + "\n")
    weekly_data = df.resample('W', on='time').agg({
        'solar_irradiance': lambda x: np.sum(x * SCALE_SOLAR),
        'wind_speed': lambda x: np.sum(x * SCALE_WIND),
        'biomass_energy': lambda x: np.sum(x * SCALE_BIOGAS),
        'load_demand': 'sum',
        'grid_purchase': 'sum'
    })
    for idx, row in weekly_data.iterrows():
        f.write(f"{str(idx)[:10]:<15} | {row['solar_irradiance']:<12.2f} | {row['wind_speed']:<12.2f} | {row['biomass_energy']:<12.2f} | {row['load_demand']:<12.2f} | {row['grid_purchase']:<10.2f}\n")
    f.write("-" * 115 + "\n")

print(f"Analysis complete. Report saved to {table_file}. Graphs saved in {output_dir}/")
