import os
import subprocess
import sys

def run_script(script_path):
    """
    Executes a python script and returns its success status.
    """
    print(f"\n--- Running: {script_path} ---")
    try:
        # Using sys.executable to ensure we use the same python interpreter
        result = subprocess.run([sys.executable, script_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")
        return False

def main():
    print("Starting Micro-Smart-Grid Project Execution...")
    
    # 1. Clean Data
    if not run_script(os.path.join('preprocessing', 'clean_data.py')):
        print("Data cleaning failed. Aborting.")
        return

    # 2. Prepare Sequences
    if not run_script(os.path.join('preprocessing', 'prepare_sequences.py')):
        print("Sequence preparation failed. Aborting.")
        return

    # 3. Train LSTM Model
    if not run_script(os.path.join('model', 'train_lstm.py')):
        print("LSTM training failed.")
        # We continue even if one model fails to see if others work
    
    # 4. Train CNN-LSTM Model
    if not run_script(os.path.join('model', 'train_cnn_lstm.py')):
        print("CNN-LSTM training failed.")

    print("\nProject execution finished.")

if __name__ == "__main__":
    main()
