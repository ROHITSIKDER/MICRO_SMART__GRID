# Micro-Smart-Grid Energy Management System

This project implements a comprehensive Micro-Smart-Grid management system featuring time-series forecasting for renewable energy sources (Solar, Wind, Biomass) and an intelligent optimization module for energy storage and cost reduction.

## Features

- **Multi-Source Forecasting**: Deep learning models (LSTM and CNN-LSTM) to predict energy generation from solar, wind, and biomass.
- **Microgrid Optimization**: An intelligent energy management system that optimizes battery usage and grid dependency to minimize costs.
- **Model Comparison**: Automated scripts to evaluate and compare different neural network architectures.
- **Visualization**: Graphical representation of forecasting accuracy and optimization results.

## Project Structure

- `Data/`: Contains raw, cleaned, and merged datasets (`biogas_dataset.csv`, `final_data.csv`, etc.).
- `preprocessing/`: Scripts for data cleaning and sequence preparation (`clean_data.py`, `merge_data.py`, `prepare_sequences.py`).
- `model/`: Training scripts and saved model files.
    - `train_lstm.py`: Train a standard LSTM model.
    - `train_cnn_lstm.py`: Train a hybrid CNN-LSTM model.
    - `saved_models/`: Serialized Keras models (`.keras`).
- `optimization_module.py`: The core optimization engine that uses model predictions to manage energy flow.
- `microgrid_optimization_system.py`: A simulation environment for testing energy management strategies.
- `evaluate_models.py`: Script to validate model performance on test data.
- `compare_models.py`: Generates comparative metrics and visualizations between different models.
- `run_all.py`: Orchestrator script to run the full training and preprocessing pipeline.
- `requirements.txt`: Project dependencies.

## Getting Started

### Prerequisites

Ensure you have Python installed (Python 3.10+ recommended). The project uses the Keras 3 framework with the PyTorch backend.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

To execute the entire pipeline (data cleaning, sequence preparation, and model training):

```bash
python run_all.py
```

### Energy Optimization

To run the energy management optimization and see the cost-saving results:

```bash
python optimization_module.py
```

This will generate `microgrid_results.png` showing the energy distribution and battery state.

### Model Comparison

To compare the performance of the trained models:

```bash
python compare_models.py
```

This will generate `model_comparison.png` with key performance indicators (RMSE, MAE, R², etc.).

## Results

The system significantly reduces energy costs by:
1.  **Prioritizing Renewables**: Using predicted solar/wind/biomass energy first.
2.  **Smart Battery Management**: Charging the battery during periods of excess renewable generation and discharging during peak demand or low generation.
3.  **Grid Reduction**: Minimizing reliance on expensive grid power.

Graphs like `microgrid_results.png` and `model_comparison.png` provide a visual summary of these improvements.
