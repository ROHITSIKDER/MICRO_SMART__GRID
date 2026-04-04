# Micro-Smart-Grid Solar Forecasting

This project implements time-series forecasting for solar irradiance using NASA POWER daily data. It features two deep learning architectures: LSTM and CNN-LSTM.

## Project Structure

- `Data/`: Contains raw and cleaned datasets.
- `preprocessing/`: Scripts for data cleaning and sequence preparation.
- `model/`: Training scripts and saved model files.
- `app/`: Placeholder for future web application.
- `requirements.txt`: Project dependencies.
- `run_all.py`: Orchestrator script to run the entire pipeline.

## Getting Started

### Prerequisites

Ensure you have Python installed (Python 3.10+ recommended). The project uses the Keras 3 framework with the PyTorch backend.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

To execute the entire pipeline (data cleaning, sequence preparation, and model training), run:

```bash
python run_all.py
```

### Individual Steps

1.  **Clean Data**: `python preprocessing/clean_data.py`
2.  **Prepare Sequences**: `python preprocessing/prepare_sequences.py`
3.  **Train LSTM**: `python model/train_lstm.py`
4.  **Train CNN-LSTM**: `python model/train_cnn_lstm.py`

## Models

The models are saved in the `model/saved_models/` directory in `.keras` format.

- `lstm_model.keras`: A recurrent neural network model using LSTM layers.
- `cnn_lstm_model.keras`: A hybrid model combining a 1D Convolutional layer for feature extraction and an LSTM layer for sequence learning.
