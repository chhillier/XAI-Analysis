import os
import shutil
import sys
from datetime import datetime

print("--- Starting Model Sync (ULTIMATE PARANOID VERSION) ---")

# --- Configuration ---
source_repo_path = os.path.join('..', 'NeuralNets')

source_params_file = os.path.join(source_repo_path, 'outputs', 'nas_best_hyperparameters_3obj_1D_2Feature_Signal_Second_Pass_100_Trials_latest.json')
source_model_file = os.path.join(source_repo_path, 'models', 'nas_best_model_3obj_1D_2Feature_Signal_Second_Pass_100_Trials_latest.pth')

dest_params_file = 'outputs/nas_best_hyperparameters_3obj_1D_2Feature_Signal_latest_XAI_pass2.json'
dest_model_file = 'models/nas_best_model_3obj_1D_2Feature_Signal_latest_XAI_pass2.pth'

# --- Paranoid File Checks ---
if not os.path.exists(source_params_file):
    print(f"\nFATAL ERROR: Source parameter file not found at: {os.path.abspath(source_params_file)}")
    sys.exit(1)

if not os.path.exists(source_model_file):
    print(f"\nFATAL ERROR: Source model file not found at: {os.path.abspath(source_model_file)}")
    sys.exit(1)

print("\nSource files found successfully!")

# --- Get Source File Metadata ---
source_model_stat = os.stat(source_model_file)
source_model_size = source_model_stat.st_size
source_model_mtime = datetime.fromtimestamp(source_model_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

print("\n--- SOURCE MODEL METADATA ---")
print(f"  - Path: {os.path.abspath(source_model_file)}")
print(f"  - Size: {source_model_size} bytes")
print(f"  - Last Modified: {source_model_mtime}")

# --- Create destination directories ---
os.makedirs(os.path.dirname(dest_params_file), exist_ok=True)
os.makedirs(os.path.dirname(dest_model_file), exist_ok=True)

# --- Copy Files ---
print("\nCopying files...")
shutil.copy(source_params_file, dest_params_file)
shutil.copy(source_model_file, dest_model_file)
print("Copy complete.")

# --- Get Destination File Metadata ---
dest_model_stat = os.stat(dest_model_file)
dest_model_size = dest_model_stat.st_size
dest_model_mtime = datetime.fromtimestamp(dest_model_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

print("\n--- DESTINATION MODEL METADATA ---")
print(f"  - Path: {os.path.abspath(dest_model_file)}")
print(f"  - Size: {dest_model_size} bytes")
print(f"  - Last Modified: {dest_model_mtime}")

# --- Final Verification ---
if source_model_size == dest_model_size:
    print("\nSUCCESS: File sizes match.")
else:
    print("\nCRITICAL ERROR: File sizes DO NOT match. The copy failed or was corrupted.")

print("\n--- Sync Complete ---")