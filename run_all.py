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
    
    # 1. Preprocessing Steps
    if not run_script(os.path.join('preprocessing', 'clean_data.py')):
        return

    if not run_script(os.path.join('preprocessing', 'clean_biomass.py')):
        return

    if not run_script(os.path.join('preprocessing', 'merge_data.py')):
        return

    if not run_script(os.path.join('preprocessing', 'prepare_sequences.py')):
        return

    # 2. Training Steps (Now in model/)
    if not run_script(os.path.join('model', 'train_lstm.py')):
        return
    
    if not run_script(os.path.join('model', 'train_cnn_lstm.py')):
        return

    # 3. Core Logic (Moved to core/)
    if not run_script(os.path.join('core', 'evaluate_models.py')):
        return

    if not run_script(os.path.join('core', 'optimization_module.py')):
        return

    if not run_script(os.path.join('core', 'compare_models.py')):
        return

    # 4. Optional Reporting (Moved to scripts/)
    # We run the report generation as the final step
    if os.path.exists(os.path.join('scripts', 'generate_full_report.py')):
        run_script(os.path.join('scripts', 'generate_full_report.py'))

    print("\nProject execution and reporting finished successfully.")

if __name__ == "__main__":
    main()
