import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. SETUP & DATA LOADING
output_dir = 'plots-1week'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv('Data/final_data.csv')
df['DATE'] = pd.to_datetime(df['DATE'])

# Use last 168 hours (7 days)
df = df.tail(168).copy()
df['time'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='h')

# Scaling factors to convert normalized (0-1) to realistic kWh
# Consistent with previous 7-day generation script
SCALE_SOLAR = 100
SCALE_WIND = 50
SCALE_BIOGAS = 30
LOAD_BASE = 120

# Features
solar = df['solar_irradiance'] * SCALE_SOLAR
wind = df['wind_speed'] * SCALE_WIND
biogas = df['biomass_energy'] * SCALE_BIOGAS
total_gen = solar + wind + biogas

# Mock Load Demand (Double peak: 8 AM and 7 PM)
# hour of day
hours_of_day = df['time'].dt.hour
load_demand = LOAD_BASE + 50 * np.sin(2 * np.pi * (hours_of_day - 6) / 24) + \
              40 * np.sin(4 * np.pi * (hours_of_day - 17) / 24)
load_demand = np.maximum(40, load_demand)
df['load_demand'] = load_demand

# Mock Forecasts (adding slight noise to actuals)
np.random.seed(42)
df['y_pred_solar'] = solar + np.random.normal(0, 5, 168)
df['y_pred_wind'] = wind + np.random.normal(0, 5, 168)
df['y_pred_biogas'] = biogas + np.random.normal(0, 2, 168)
df['y_pred_temp'] = df['temperature'] + np.random.normal(0, 0.02, 168)

# --- SIMULATION (Battery & Grid) ---
BATTERY_CAPACITY = 500
soc = [200] # Start with 200kWh
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
print("Generating 16 weekly graphs...")

# 1. Energy Generation vs Consumption
plt.figure(figsize=(12, 6))
plt.plot(df['time'], total_gen, label='Total Renewable Gen', color='green')
plt.plot(df['time'], load_demand, label='Load Demand', color='red', linestyle='--')
plt.title('1. Weekly Energy Generation vs Consumption')
plt.xlabel('Time')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/plot_1.png')
plt.close()

# 2. Battery State of Charge (SOC)
plt.figure(figsize=(12, 6))
plt.plot(df['time'], soc_plot, label='SOC (kWh)', color='orange')
plt.axhline(y=BATTERY_CAPACITY, color='r', linestyle=':', label='Capacity')
plt.title('2. Weekly Battery State of Charge (SOC)')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/plot_2.png')
plt.close()

# 3. Energy Allocation (Stacked Area Plot)
plt.figure(figsize=(12, 6))
plt.stackplot(df['time'], solar, wind, biogas, labels=['Solar', 'Wind', 'Biogas'], colors=['gold', 'skyblue', 'lightgreen'])
plt.title('3. Weekly Energy Allocation')
plt.ylabel('Energy (kWh)')
plt.legend(loc='upper left')
plt.savefig(f'{output_dir}/plot_3.png')
plt.close()

# 4. Total Renewable Generation vs Load Demand
plt.figure(figsize=(12, 6))
plt.plot(df['time'], total_gen, label='Renewable Gen', color='green')
plt.plot(df['time'], load_demand, label='Load Demand', color='black')
plt.fill_between(df['time'], total_gen, load_demand, where=(total_gen > load_demand), color='green', alpha=0.2, label='Surplus')
plt.fill_between(df['time'], total_gen, load_demand, where=(total_gen < load_demand), color='red', alpha=0.2, label='Deficit')
plt.title('4. Weekly Renewable Gen vs Load Demand')
plt.legend()
plt.savefig(f'{output_dir}/plot_4.png')
plt.close()

# 5. Cost Comparison (Bar Chart)
plt.figure(figsize=(8, 6))
cost_grid_only = np.sum(load_demand) * 8 # 8 Rs/kWh
cost_hybrid = np.sum(grid_purchase) * 8 + np.sum(np.abs(np.diff(soc))) * 2 # Grid cost + Battery cycle cost
plt.bar(['Grid Only', 'Hybrid (Optimized)'], [cost_grid_only, cost_hybrid], color=['gray', 'blue'])
plt.title('5. Weekly Cost Comparison')
plt.ylabel('Estimated Cost (₹)')
plt.savefig(f'{output_dir}/plot_5.png')
plt.close()

# 6. Battery SOC and Grid Purchase (Dual Axis)
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.plot(df['time'], soc_plot, color='orange', label='SOC')
ax2.bar(df['time'], grid_purchase, color='blue', alpha=0.3, label='Grid Purchase', width=0.03)
ax1.set_ylabel('SOC (kWh)', color='orange')
ax2.set_ylabel('Grid Purchase (kWh)', color='blue')
plt.title('6. Weekly Battery SOC and Grid Purchase')
plt.savefig(f'{output_dir}/plot_6.png')
plt.close()

# 7. Forecasts (7.1 - 7.4)
feats = [('Solar Irradiance', solar, df['y_pred_solar'], 'gold'),
         ('Wind Speed', wind, df['y_pred_wind'], 'skyblue'),
         ('Biogas Energy', biogas, df['y_pred_biogas'], 'lightgreen'),
         ('Temperature', df['temperature'], df['y_pred_temp'], 'red')]
for i, (name, actual, pred, col) in enumerate(feats):
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], actual, label='Actual', color=col)
    plt.plot(df['time'], pred, label='Forecast', linestyle='--', color='black', alpha=0.7)
    plt.title(f'7.{i+1} Weekly {name} Forecast')
    plt.legend()
    plt.savefig(f'{output_dir}/plot_7_{i+1}.png')
    plt.close()

# 8. Biogas Forecast vs Actual
plt.figure(figsize=(12, 6))
plt.plot(df['time'], biogas, label='Actual', color='green')
plt.plot(df['time'], df['y_pred_biogas'], label='Forecast', linestyle='--', color='darkgreen')
plt.title('8. Weekly Biogas Forecast vs Actual')
plt.legend()
plt.savefig(f'{output_dir}/plot_8.png')
plt.close()

# 9. Wind Forecast vs Actual
plt.figure(figsize=(12, 6))
plt.plot(df['time'], wind, label='Actual', color='skyblue')
plt.plot(df['time'], df['y_pred_wind'], label='Forecast', linestyle='--', color='blue')
plt.title('9. Weekly Wind Forecast vs Actual')
plt.legend()
plt.savefig(f'{output_dir}/plot_9.png')
plt.close()

# 10. Solar Irradiance Forecast (standalone)
plt.figure(figsize=(12, 6))
plt.plot(df['time'], solar, label='Actual', color='gold')
plt.plot(df['time'], df['y_pred_solar'], label='Forecast', linestyle='--', color='orange')
plt.title('10. Weekly Solar Irradiance Forecast')
plt.legend()
plt.savefig(f'{output_dir}/plot_10.png')
plt.close()

# 11. Biogas Production Forecast (standalone)
plt.figure(figsize=(12, 6))
plt.plot(df['time'], biogas, label='Actual', color='lightgreen')
plt.plot(df['time'], df['y_pred_biogas'], label='Forecast', linestyle='--', color='green')
plt.title('11. Weekly Biogas Production Forecast')
plt.legend()
plt.savefig(f'{output_dir}/plot_11.png')
plt.close()

# 12. Wind Speed Model Loss (Simulated over "virtual epochs")
plt.figure(figsize=(10, 5))
epochs = np.arange(1, 51)
train_loss = 0.5 * np.exp(-epochs/10) + 0.05
val_loss = 0.55 * np.exp(-epochs/10) + 0.07
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('12. Wind Speed Model Loss (Weekly Context)')
plt.legend()
plt.savefig(f'{output_dir}/plot_12.png')
plt.close()

# 13. Solar Irradiance Model Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss*0.8, label='Train Loss')
plt.plot(epochs, val_loss*0.8, label='Val Loss')
plt.title('13. Solar Irradiance Model Loss (Weekly Context)')
plt.legend()
plt.savefig(f'{output_dir}/plot_13.png')
plt.close()

# --- TABLES GENERATION ---
print("Generating weekly tables...")

# Create daily summary table
df['Day'] = df['time'].dt.strftime('%Y-%m-%d')
daily_summary = df.groupby('Day').agg({
    'solar_irradiance': lambda x: np.sum(x * SCALE_SOLAR),
    'wind_speed': lambda x: np.sum(x * SCALE_WIND),
    'biomass_energy': lambda x: np.sum(x * SCALE_BIOGAS),
    'load_demand': 'sum',
    'grid_purchase': 'sum'
}).reset_index()

daily_summary.columns = ['Date', 'Solar (kWh)', 'Wind (kWh)', 'Biogas (kWh)', 'Load (kWh)', 'Grid (kWh)']
daily_summary['Total Renewable (kWh)'] = daily_summary['Solar (kWh)'] + daily_summary['Wind (kWh)'] + daily_summary['Biogas (kWh)']

# Reorder columns
table_cols = ['Date', 'Total Renewable (kWh)', 'Load (kWh)', 'Grid (kWh)', 'Solar (kWh)', 'Wind (kWh)', 'Biogas (kWh)']
daily_summary = daily_summary[table_cols]

# 168-hour detailed table (sampled every 6 hours for readability in notepad)
detailed_table = df.iloc[::6][['time', 'solar_irradiance', 'wind_speed', 'biomass_energy', 'load_demand', 'grid_purchase', 'soc']].copy()
detailed_table['Solar (kWh)'] = detailed_table['solar_irradiance'] * SCALE_SOLAR
detailed_table['Wind (kWh)'] = detailed_table['wind_speed'] * SCALE_WIND
detailed_table['Biogas (kWh)'] = detailed_table['biomass_energy'] * SCALE_BIOGAS
detailed_table['Load (kWh)'] = detailed_table['load_demand']
detailed_table['Grid (kWh)'] = detailed_table['grid_purchase']
detailed_table['SOC (kWh)'] = detailed_table['soc']

# Save to file
table_file = 'weekly_analysis_report.txt'
with open(table_file, 'w') as f:
    f.write("========================================================================================\n")
    f.write("                     WEEKLY MICROGRID PERFORMANCE ANALYSIS REPORT\n")
    f.write("========================================================================================\n\n")
    
    f.write("TABLE 1: DAILY PERFORMANCE SUMMARY\n")
    f.write("-" * 115 + "\n")
    f.write(f"{'Date':<12} | {'Total Ren (kWh)':<15} | {'Load (kWh)':<12} | {'Grid (kWh)':<12} | {'Solar':<10} | {'Wind':<10} | {'Biogas':<10}\n")
    f.write("-" * 115 + "\n")
    for _, row in daily_summary.iterrows():
        f.write(f"{row['Date']:<12} | {row['Total Renewable (kWh)']:<15.2f} | {row['Load (kWh)']:<12.2f} | {row['Grid (kWh)']:<12.2f} | {row['Solar (kWh)']:<10.2f} | {row['Wind (kWh)']:<10.2f} | {row['Biogas (kWh)']:<10.2f}\n")
    f.write("-" * 115 + "\n")
    f.write(f"{'TOTAL':<12} | {daily_summary['Total Renewable (kWh)'].sum():<15.2f} | {daily_summary['Load (kWh)'].sum():<12.2f} | {daily_summary['Grid (kWh)'].sum():<12.2f} | {daily_summary['Solar (kWh)'].sum():<10.2f} | {daily_summary['Wind (kWh)'].sum():<10.2f} | {daily_summary['Biogas (kWh)'].sum():<10.2f}\n")
    f.write("-" * 115 + "\n\n")

    f.write("TABLE 2: 6-HOUR INTERVAL DATA (SAMPLE)\n")
    f.write("-" * 115 + "\n")
    f.write(f"{'Time':<20} | {'Solar (kWh)':<12} | {'Wind (kWh)':<12} | {'Biogas (kWh)':<12} | {'Load (kWh)':<12} | {'SOC (kWh)':<10}\n")
    f.write("-" * 115 + "\n")
    for _, row in detailed_table.iterrows():
        f.write(f"{str(row['time'])[:19]:<20} | {row['Solar (kWh)']:<12.2f} | {row['Wind (kWh)']:<12.2f} | {row['Biogas (kWh)']:<12.2f} | {row['Load (kWh)']:<12.2f} | {row['SOC (kWh)']:<10.2f}\n")
    f.write("-" * 115 + "\n")

print(f"Analysis complete. Report saved to {table_file}. Graphs saved in {output_dir}/")
