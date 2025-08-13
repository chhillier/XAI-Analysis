import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from captum.attr import Saliency

# Make sure your cnn1D.py is in the mymodelzoo subfolder
from mymodelzoo.cnn1D import DynamicCNN

# --- 1. Configuration ---
PARAMS_PATH = "outputs/nas_best_hyperparameters_3obj_1D_2Feature_Signal_latest_XAI_pass2.json"
MODEL_PATH = "models/nas_best_model_3obj_1D_2Feature_Signal_latest_XAI_pass2.pth" 
X_TEST_PATH = "../NeuralNets/test data/X_test.npy" 
Y_TEST_PATH = "../NeuralNets/test data/y_test.npy"

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

def find_correctly_classified_sample(model, X_test_tensor, y_test_tensor, predictions, target_class_name):
    """Finds the first instance of a correctly classified sample for a given class."""
    print(f"\nSearching for a correctly classified '{target_class_name}' sample...")
    target_class_index = CLASS_NAMES.index(target_class_name)
    for i in range(len(y_test_tensor)):
        if y_test_tensor[i].item() == target_class_index and predictions[i].item() == target_class_index:
            print(f"Found a sample at index {i}.")
            print(f"  - True Label: '{CLASS_NAMES[y_test_tensor[i].item()]}'")
            print(f"  - Predicted Label: '{CLASS_NAMES[predictions[i].item()]}'")
            return X_test_tensor[i].unsqueeze(0), y_test_tensor[i].item(), predictions[i].item()
    
    print(f"Could not find a correctly classified '{target_class_name}' sample.")
    return None, None, None

def find_misclassified_sample(model, X_test_tensor, y_test_tensor, predictions, target_class_name):
    """Finds the first instance of a misclassified sample for a given class."""
    print(f"\nSearching for a misclassified '{target_class_name}' sample...")
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

def visualize_attributions(signal, attributions, true_label_name, pred_label_name, output_filename):
    """Plots the signal and the corresponding attribution map."""
    print("\nGenerating attribution plot...")
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

    title = f"XAI Analysis: Why was '{true_label_name}' correctly classified as '{pred_label_name}'?"
    if true_label_name != pred_label_name:
        title = f"XAI Analysis: Why was '{true_label_name}' misclassified as '{pred_label_name}'?"

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    model, X_test, y_test = load_model_and_data()
    
    # Pre-calculate all predictions once to avoid re-computing
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)

    output_dir = "3rd Run EEG/Saliency"
    os.makedirs(output_dir, exist_ok=True)
        # --- DIAGNOSTIC BLOCK ---
    print("\n--- Running Diagnostic for 'Sleep Disorder' ---")
    sleep_disorder_idx = CLASS_NAMES.index('Sleep Disorder')
    actual_sd_indices = np.where(y_test == sleep_disorder_idx)[0]
    
    if len(actual_sd_indices) > 0:
        correct_count = 0
        print(f"Found {len(actual_sd_indices)} 'Sleep Disorder' samples in the test set.")
        for idx in actual_sd_indices:
            true_label = CLASS_NAMES[y_test[idx]]
            pred_label = CLASS_NAMES[predictions[idx].item()]
            is_correct = "CORRECT" if true_label == pred_label else "INCORRECT"
            if is_correct == "CORRECT":
                correct_count += 1
            print(f"  - Sample {idx}: True Label = {true_label}, Predicted Label = {pred_label}  ({is_correct})")
        
        print(f"\n  --> Accuracy for this class: {correct_count / len(actual_sd_indices):.2%}")
    else:
        print("No 'Sleep Disorder' samples found in the test set.")
    print("--- End of Diagnostic ---")
    # --------------------------------------------------

    # --- Loop to analyze a CORRECTLY classified sample for EACH class ---
    for target_class in CLASS_NAMES:
        input_sample, true_label_idx, pred_label_idx = find_correctly_classified_sample(
            model, X_test_tensor, y_test_tensor, predictions, target_class
        )
        
        if input_sample is not None:
            attributions = explain_prediction(model, input_sample, pred_label_idx)
            output_filename = f"{output_dir}/xai_saliency_plot_CORRECT_{target_class.replace(' ', '_')}.png"
            visualize_attributions(
                signal=input_sample.squeeze(0).cpu().numpy(),
                attributions=attributions,
                true_label_name=CLASS_NAMES[true_label_idx],
                pred_label_name=CLASS_NAMES[pred_label_idx],
                output_filename=output_filename
            )

    # --- Specific analysis for our MISCLASSIFIED "confuser" class ---
    target_class = 'Metabolic Encephalopathy'
    input_sample, true_label_idx, pred_label_idx = find_misclassified_sample(
        model, X_test_tensor, y_test_tensor, predictions, target_class
    )
    
    if input_sample is not None:
        attributions = explain_prediction(model, input_sample, pred_label_idx)
        output_filename = f"2nd Run EEG/xai_saliency_plot_MISCLASSIFIED_{target_class.replace(' ', '_')}.png"
        visualize_attributions(
            signal=input_sample.squeeze(0).cpu().numpy(),
            attributions=attributions,
            true_label_name=CLASS_NAMES[true_label_idx],
            pred_label_name=CLASS_NAMES[pred_label_idx],
            output_filename=output_filename
        )
