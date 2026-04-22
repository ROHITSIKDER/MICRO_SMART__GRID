import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset (CSV)
df = pd.read_csv('Data/final_data.csv')

# 2. Convert 'DATE' column to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# 3. Keep only last 24 rows
df = df.tail(24).copy()

# 4. Replace time column with latest 24-hour timeline
df['time'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='h')

# --- Mock Model Outputs for Demonstration ---
# In a real scenario, these would come from your CNN-LSTM model
np.random.seed(42)
df['y_pred_solar'] = df['solar_irradiance'] + np.random.normal(0, 0.05, 24)
df['y_pred_wind'] = df['wind_speed'] + np.random.normal(0, 0.05, 24)
df['y_pred_biogas'] = df['biomass_energy'] + np.random.normal(0, 0.05, 24)

# Mock history (for losses)
history_loss = np.linspace(0.5, 0.1, 24)
history_val_loss = np.linspace(0.6, 0.15, 24)

# --- Define Helper Variables ---
solar = df['solar_irradiance']
wind = df['wind_speed']
biogas = df['biomass_energy']
load_demand = 0.5  # Assumed constant demand for simulation
total_gen = solar + wind + biogas

# --- CREATE GRAPHS AND SAVE ---
output_dir = 'plots-24hours'

# 1. Energy Generation vs Consumption
plt.figure(figsize=(10, 5))
plt.plot(df['time'], total_gen, label='Total Generation')
plt.plot(df['time'], [load_demand]*24, label='Load Demand', linestyle='--')
plt.title('Energy Generation vs Consumption')
plt.xlabel('Time')
plt.ylabel('Energy (Normalized)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_1.png')
plt.close()

# 2. Battery State of Charge (SOC)
soc = [50] # Initial SOC
for i in range(1, 24):
    delta = total_gen.iloc[i] - load_demand
    soc.append(max(0, min(100, soc[-1] + delta * 10)))
plt.figure(figsize=(10, 5))
plt.plot(df['time'], soc, label='SOC (%)')
plt.title('Battery State of Charge (SOC)')
plt.xlabel('Time')
plt.ylabel('SOC (%)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_2.png')
plt.close()

# 3. Energy Allocation (Stacked Area Plot)
plt.figure(figsize=(10, 5))
plt.stackplot(df['time'], solar, wind, biogas, labels=['Solar', 'Wind', 'Biogas'])
plt.title('Energy Allocation')
plt.xlabel('Time')
plt.ylabel('Energy Contribution')
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_3.png')
plt.close()

# 4. Total Renewable Generation vs Load Demand
plt.figure(figsize=(10, 5))
plt.plot(df['time'], total_gen, label='Renewable Gen')
plt.plot(df['time'], [load_demand]*24, label='Load Demand')
plt.fill_between(df['time'], total_gen, [load_demand]*24, where=(total_gen>[load_demand]*24), color='green', alpha=0.3, label='Surplus')
plt.fill_between(df['time'], total_gen, [load_demand]*24, where=(total_gen<[load_demand]*24), color='red', alpha=0.3, label='Deficit')
plt.title('Total Renewable Generation vs Load Demand')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_4.png')
plt.close()

# 5. Cost Comparison (Bar Chart)
plt.figure(figsize=(8, 5))
plt.bar(['Hybrid System', 'Grid-only'], [150, 400], color=['blue', 'gray'])
plt.title('Cost Comparison (24 Hours)')
plt.ylabel('Cost ($)')
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_5.png')
plt.close()

# 6. Battery SOC and Grid Purchase (Dual Axis)
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(df['time'], soc, color='blue', label='SOC')
ax2.plot(df['time'], [max(0, load_demand-g) for g in total_gen], color='red', label='Grid Purchase', linestyle='--')
ax1.set_xlabel('Time')
ax1.set_ylabel('SOC (%)', color='blue')
ax2.set_ylabel('Grid Purchase', color='red')
plt.title('Battery SOC and Grid Purchase')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_6.png')
plt.close()

# 7. Forecasts (7.1 - 7.4)
features = ['solar_irradiance', 'wind_speed', 'biomass_energy', 'temperature']
for i, feat in enumerate(features):
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], df[feat], label='Actual')
    
    if feat == 'solar_irradiance': pred_col = 'y_pred_solar'
    elif feat == 'wind_speed': pred_col = 'y_pred_wind'
    elif feat == 'biomass_energy': pred_col = 'y_pred_biogas'
    else: pred_col = feat
        
    plt.plot(df['time'], df[pred_col], label='Forecast', linestyle='--')
    plt.title(f'{feat.replace("_", " ").title()} Forecast')
    plt.xlabel('Time')
    plt.ylabel(feat.replace("_", " ").title())
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_7_{i+1}.png')
    plt.close()

# 8. Biogas Forecast vs Actual
plt.figure(figsize=(10, 5))
plt.plot(df['time'], biogas, label='Actual')
plt.plot(df['time'], df['y_pred_biogas'], label='Forecast', linestyle='--')
plt.title('Biogas Forecast vs Actual')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_8.png')
plt.close()

# 9. Wind Forecast vs Actual
plt.figure(figsize=(10, 5))
plt.plot(df['time'], wind, label='Actual')
plt.plot(df['time'], df['y_pred_wind'], label='Forecast', linestyle='--')
plt.title('Wind Forecast vs Actual')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_9.png')
plt.close()

# 10. Solar Irradiance Forecast (standalone)
plt.figure(figsize=(10, 5))
plt.plot(df['time'], solar, label='Actual')
plt.plot(df['time'], df['y_pred_solar'], label='Forecast', linestyle='--')
plt.title('Solar Irradiance Forecast')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_10.png')
plt.close()

# 11. Biogas Production Forecast (standalone)
plt.figure(figsize=(10, 5))
plt.plot(df['time'], biogas, label='Actual')
plt.plot(df['time'], df['y_pred_biogas'], label='Forecast', linestyle='--')
plt.title('Biogas Production Forecast')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_11.png')
plt.close()

# 12. Wind Speed Model Loss
plt.figure(figsize=(10, 5))
plt.plot(history_loss, label='Train Loss')
plt.plot(history_val_loss, label='Val Loss')
plt.title('Wind Speed Model Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_12.png')
plt.close()

# 13. Solar Irradiance Model Loss
plt.figure(figsize=(10, 5))
plt.plot(history_loss, label='Train Loss')
plt.plot(history_val_loss, label='Val Loss')
plt.title('Solar Irradiance Model Loss')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/plot_13.png')
plt.close()
