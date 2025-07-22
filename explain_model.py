import torch
import torch.nn as nn
import numpy as np
import json
import os
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from mymodelzoo.dnn import DynamicDNN

from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def prepare_data():
    """Loads and prepares the Iris dataset for evaluation."""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72125, stratify=y)
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_test, iris.target_names, iris.feature_names, X_train

def evaluate_model(model, test_loader, device, class_names=None):
    """A standalone function to evaluate the model."""
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\n--- Model Evaluation on Test Set ---")
    print(classification_report(all_preds, all_labels, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    # --- Configuration ---
    PARAMS_FILENAME = "outputs/best_hyperparameters_cv_latest.json"
    MODEL_FILENAME = "models/best_model_cv_latest.pth"
    
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    if not os.path.exists(PARAMS_FILENAME) or not os.path.exists(MODEL_FILENAME):
        print(f"Error: Make sure '{PARAMS_FILENAME}' and '{MODEL_FILENAME}' are in the same folder.")
        print("Please ensure you have run a training script that generates these files first.")
        exit()

    # --- 1. Load Data ---
    X_train_scaled, X_test_scaled, y_test, class_names, feature_names, original_X_train = prepare_data()
    input_size, num_classes = X_train_scaled.shape[1], len(class_names)
    
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    # --- 2. Load Hyperparameters and Build Model ---
    with open(PARAMS_FILENAME, 'r') as f:
        best_params = json.load(f)

    final_model = DynamicDNN(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['n_layers'],
        activation_fn=getattr(nn, best_params['activation']),
        dropout_rate=best_params['dropout_rate']
    )

    # --- 3. Load Trained Model Weights ---
    final_model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
    final_model.eval()
    print("--- Model Loaded Successfully ---")

    # --- 4. Evaluate the Loaded Model ---
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=32)
    evaluate_model(final_model, test_loader, DEVICE, class_names)

    # =======================================================
    # === XAI 1: Explaining the Model with SHAP ===
    # =======================================================
    print("\n--- XAI 1: Generating SHAP Explanations ---")
    
    background_sample = X_train_scaled[np.random.choice(X_train_scaled.shape[0], min(100, X_train_scaled.shape[0]), replace=False)]
    background_sample_tensors = torch.tensor(background_sample, dtype=torch.float32).to(DEVICE)
    
    explainer = shap.DeepExplainer(final_model, background_sample_tensors)
    
    X_test_tensors = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    print("Calculating SHAP values (main effects)...")
    # For multi-output DeepExplainer, shap_values can be a list of arrays or a single 3D array.
    # The debug output showed (30, 4, 3), which implies a single 3D array.
    shap_values = explainer.shap_values(X_test_tensors)
    
    # --- DEBUGGING PRINTS ---
    print(f"Shape of X_test_scaled: {X_test_scaled.shape}")
    print(f"Shape of shap_values (raw from explainer): {np.array(shap_values).shape}")
    # --- END DEBUGGING PRINTS ---

    # --- SHAP Summary Plot (Beeswarm) ---
    print("\nGenerating SHAP Summary Plot (Global Explanation)...")
    
    # Assume shap_values is a 3D array: (num_samples, num_features, num_classes)
    # Iterate through classes and slice the shap_values for each class
    for i, name in enumerate(class_names):
        plt.figure(figsize=(10, 6))
        # Now correctly slice for the i-th class: all samples, all features, for this class
        shap.summary_plot(shap_values[:, :, i], X_test_scaled, feature_names=feature_names, show=False, title=f"SHAP Summary Plot for Class: {name}")
        plt.tight_layout()
        plt.savefig(f"plots/shap_summary_plot_class_{i}_{name}.png", dpi=300)
        plt.close()
    print("SHAP Summary Plots saved to the 'plots' directory.")

    # --- SHAP Interaction Values (Addressing TypeError) ---
    # Since DeepExplainer.shap_values() does not accept 'interactions=True' in your version,
    # we need to calculate them differently.
    # The most common approach is to use shap.GradientExplainer if DeepExplainer doesn't support it,
    # or rely on the general shap.explainers.Explainer for interactions.
    # However, a simpler way, if available and suitable for your version, is to use a specific
    # interaction explainer or check the SHAP documentation for DeepExplainer interactions.
    
    # Given the TypeError, the direct way to get interaction values with DeepExplainer
    # as in a `TreeExplainer` or newer `DeepExplainer` versions is not available.
    # A common workaround for Deep Learning models not directly supporting interactions in DeepExplainer
    # is to calculate approximate interaction values from main SHAP values, or use a different explainer.
    # For simplicity and to avoid another explainer, we will skip the dedicated interaction plot for now.
    # If you need this, you might need to:
    # 1. Update your SHAP library: `pip install --upgrade shap`
    # 2. If update doesn't work, consider shap.GradientExplainer which might have interaction support.
    #    However, GradientExplainer requires a specific `model.backward()` setup often.
    # 3. Use `shap.explainers.ExactExplainer` if dataset is small.
    
    # Keeping the old line commented out for reference on what was attempted
    # print("Calculating SHAP interaction values...")
    # shap_interaction_values = explainer.shap_values(X_test_tensors, interactions=True)
    
    # Placeholder for the interaction plot, currently skipped due to TypeError
    print("\nSkipping SHAP Interaction Plot due to 'TypeError: interactions' argument not supported by DeepExplainer in current SHAP version.")
    print("Consider upgrading SHAP (`pip install --upgrade shap`) or exploring alternative explainer types for interaction values.")

    # =======================================================
    # === XAI 2: Explaining the Model with LIME ===
    # =======================================================
    print("\n--- XAI 2: Generating LIME Explanation ---")

    def predictor(numpy_array):
        tensor = torch.from_numpy(numpy_array).float().to(DEVICE)
        model_output = final_model(tensor)
        probabilities = torch.nn.functional.softmax(model_output, dim=1)
        return probabilities.cpu().detach().numpy()

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=original_X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    instance_to_explain_index = 0
    instance_to_explain_scaled = X_test_scaled[instance_to_explain_index]

    print(f"\nGenerating LIME explanation for the first test sample (index {instance_to_explain_index})...")
    explanation = lime_explainer.explain_instance(
        data_row=instance_to_explain_scaled,
        predict_fn=predictor,
        num_features=len(feature_names),
        num_samples=5000
    )

    lime_output_path = 'plots/lime_explanation.html'
    explanation.save_to_file(lime_output_path)
    print(f"LIME explanation saved to {lime_output_path}")

    # Optional: Display a single SHAP force plot for an individual prediction
    print("\nGenerating SHAP Force Plot for a single prediction (Class 0)...")
    # For force plots with a 3D shap_values array, you need to provide:
    # base_value (explainer.expected_value[class_index])
    # shap_values_for_instance_and_class (shap_values[instance_index, :, class_index])
    # features_for_instance (X_test_scaled[instance_index])
    
    # Ensure we have enough data points and classes for the chosen instance
    if shap_values.shape[0] > instance_to_explain_index and shap_values.shape[2] > 0:
        # Select SHAP values for the first instance and the first class (index 0)
        shap.force_plot(explainer.expected_value[0],
                        shap_values[instance_to_explain_index, :, 0], # Correct slice: this instance, all features, for class 0
                        features=X_test_scaled[instance_to_explain_index],
                        feature_names=feature_names,
                        matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(f"plots/shap_force_plot_instance_{instance_to_explain_index}_class_0.png", dpi=300)
        plt.close()
        print(f"SHAP Force Plot for instance {instance_to_explain_index} (class 0) saved to 'plots' directory.")
    else:
        print(f"Not enough data points or classes to generate force plot for instance {instance_to_explain_index}, class 0.")

    print("\n--- XAI Analysis Complete ---")
    print("All plots and explanations have been saved to the 'plots' directory.")