import os
import shutil

print("--- Starting Model Sync ---")

# --- Configuration ---
# Define the relative path to the source repository
source_repo_path = os.path.join('..', 'NeuralNets')

# Define the source filenames
source_params_file = os.path.join(source_repo_path, 'outputs', 'best_hyperparameters_cv_latest.json')
source_model_file = os.path.join(source_repo_path, 'models', 'best_model_cv_latest.pth')

# Define the destination filenames
dest_params_file = 'outputs/best_hyperparameters_cv_latest.json'
dest_model_file = 'models/best_model_cv_latest.pth'

# --- Copy Parameter File ---
if os.path.exists(source_params_file):
    print(f"Copying {source_params_file} to {dest_params_file}...")
    shutil.copy(source_params_file, dest_params_file)
else:
    print(f"ERROR: Source file not found: {source_params_file}")

# --- Copy Model File ---
if os.path.exists(source_model_file):
    print(f"Copying {source_model_file} to {dest_model_file}...")
    shutil.copy(source_model_file, dest_model_file)
else:
    print(f"ERROR: Source file not found: {source_model_file}")

print("\n--- Sync Complete ---")