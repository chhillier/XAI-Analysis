import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from captum.attr import Saliency

# Make sure your cnn1D.py is in the mymodelzoo subfolder
from mymodelzoo.cnn1D import DynamicCNN

# --- 1. Configuration (Uses files from your sync script and new test_data folder) ---
PARAMS_PATH = "outputs/nas_best_hyperparameters_3obj_1D_2Feature_Signal_latest_XAI_pass1.json"
MODEL_PATH = "models/nas_best_model_3obj_1D_2Feature_Signal_latest_XAI_pass1.pth"
# --- LOOK HERE: Updated paths to point to the test_data folder ---
X_TEST_PATH = "../NeuralNets/test data/X_test.npy" 
Y_TEST_PATH = "../NeuralNets/test data/y_test.npy"
# ----------------------------------------------------------------

# This must match the order from your converter script
CLASS_NAMES = ['Brain Rot', 'Brain Tumor', 'Metabolic Encephalopathy', 'Normal', 'Sleep Disorder']
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_data():
    """Loads the trained model, hyperparameters, and test data."""
    print(f"Using device: {DEVICE}")
    print("Loading model, hyperparameters, and test data...")

    if not all(os.path.exists(p) for p in [PARAMS_PATH, MODEL_PATH, X_TEST_PATH, Y_TEST_PATH]):
        print("Error: One or more required files not found.")
        print(f"Please ensure the following files exist:")
        print(f"  - {PARAMS_PATH}")
        print(f"  - {MODEL_PATH}")
        print(f"  - {X_TEST_PATH}")
        print(f"  - {Y_TEST_PATH}")
        exit()

    with open(PARAMS_PATH, 'r') as f:
        saved_results = json.load(f)
    best_params = saved_results['hyperparameters']

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    num_timesteps, num_features = X_test.shape[1], X_test.shape[2]
    input_shape = (num_features, num_timesteps)
    num_classes = len(CLASS_NAMES)

    model = DynamicCNN(best_params, input_shape, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print("Model and data loaded successfully.")
    return model, X_test, y_test

def find_misclassified_sample(model, X_test, y_test, target_class_name):
    """Finds the first instance of a misclassified sample for a given class."""
    print(f"\nSearching for a misclassified '{target_class_name}' sample...")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)

    target_class_index = CLASS_NAMES.index(target_class_name)
    for i in range(len(y_test_tensor)):
        if y_test_tensor[i].item() == target_class_index and predictions[i].item() != target_class_index:
            print(f"Found a sample at index {i}.")
            print(f"  - True Label: '{CLASS_NAMES[y_test_tensor[i].item()]}'")
            print(f"  - Predicted Label: '{CLASS_NAMES[predictions[i].item()]}'")
            return X_test_tensor[i].unsqueeze(0), y_test_tensor[i].item(), predictions[i].item()
    
    print(f"Could not find a misclassified '{target_class_name}' sample.")
    return None, None, None

def explain_prediction(model, input_sample, pred_label_idx):
    """Uses Captum Saliency to generate attributions for a prediction."""
    saliency = Saliency(model)
    attributions = saliency.attribute(input_sample, target=pred_label_idx)
    return attributions.squeeze(0).cpu().detach().numpy()

def visualize_attributions(signal, attributions, true_label_name, pred_label_name):
    """Plots the signal and the corresponding attribution map."""
    print("\nGenerating attribution plot...")
    # Average attributions across features for a cleaner plot
    attr_scores = np.mean(np.abs(attributions), axis=1)
    timesteps = np.arange(signal.shape[0])

    fig, ax1 = plt.subplots(figsize=(18, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('EEG Signal (Scaled)', color=color)
    ax1.plot(timesteps, signal[:, 0], color=color, linewidth=1.5, label='EEG Signal')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Attribution Score (Importance)', color=color)
    ax2.fill_between(timesteps, 0, attr_scores, color=color, alpha=0.4, label='Model Attention (Saliency)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(f"XAI Analysis: Why was '{true_label_name}' misclassified as '{pred_label_name}'?", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("xai_saliency_plot.png")
    print("Plot saved to xai_saliency_plot.png")
    plt.show()

if __name__ == "__main__":
    model, X_test, y_test = load_model_and_data()
    
    # Let's find out why the model is confused about our "confuser" class
    input_sample, true_label_idx, pred_label_idx = find_misclassified_sample(
        model, X_test, y_test, 'Metabolic Encephalopathy'
    )
    
    if input_sample is not None:
        attributions = explain_prediction(model, input_sample, pred_label_idx)
        
        visualize_attributions(
            signal=input_sample.squeeze(0).cpu().numpy(),
            attributions=attributions,
            true_label_name=CLASS_NAMES[true_label_idx],
            pred_label_name=CLASS_NAMES[pred_label_idx]
        )
