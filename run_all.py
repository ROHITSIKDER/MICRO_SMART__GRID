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
    
    # 1. Clean NASA Data
    if not run_script(os.path.join('preprocessing', 'clean_data.py')):
        print("Data cleaning failed. Aborting.")
        return

    # 1.b Clean Biomass Data
    if not run_script(os.path.join('preprocessing', 'clean_biomass.py')):
        print("Biomass cleaning failed. Aborting.")
        return

    # 1.c Merge Data
    if not run_script(os.path.join('preprocessing', 'merge_data.py')):
        print("Data merging failed. Aborting.")
        return

    # 2. Prepare Sequences
    if not run_script(os.path.join('preprocessing', 'prepare_sequences.py')):
        print("Sequence preparation failed. Aborting.")
        return

    # 3. Train LSTM Model
    if not run_script(os.path.join('model', 'train_lstm.py')):
        print("LSTM training failed. Aborting critical step.")
        return
    
    # 4. Train CNN-LSTM Model
    if not run_script(os.path.join('model', 'train_cnn_lstm.py')):
        print("CNN-LSTM training failed. Aborting critical step.")
        return

    # 5. Evaluate Models (Generates metrics.json)
    if not run_script('evaluate_models.py'):
        print("Model evaluation failed.")
        return

    # 6. Run Microgrid Optimization System (Uses CNN-LSTM predictions)
    # Using optimization_module.py which is the real integration script
    if not run_script('optimization_module.py'):
        print("Microgrid optimization module failed.")
        return

    # 7. Compare Models (Uses metrics.json and optimization_results.json)
    if not run_script('compare_models.py'):
        print("Model comparison failed.")

    print("\nProject execution finished successfully.")

if __name__ == "__main__":
    main()
