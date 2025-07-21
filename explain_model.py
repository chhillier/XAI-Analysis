import torch
import torch.nn as nn
import numpy as np
import json
import os
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from mymodelzoo.dnn import DynamicDNN # Import from your library

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
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test_scaled, y_test, iris.target_names, iris.feature_names

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
    
    if not os.path.exists(PARAMS_FILENAME) or not os.path.exists(MODEL_FILENAME):
        print(f"Error: Make sure '{PARAMS_FILENAME}' and '{MODEL_FILENAME}' are in the same folder.")
        exit()

    # --- 1. Load Data ---
    X_train, X_test_scaled, y_test, class_names, feature_names = prepare_data()
    input_size, num_classes = X_train.shape[1], len(class_names)
    DEVICE = torch.device("mps")
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
    print("--- Model Loaded Successfully ---")

    # --- 4. Evaluate the Loaded Model ---
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=32)
    evaluate_model(final_model, test_loader, DEVICE, class_names)

    # =======================================================
    # === XAI 1: Explaining the Model with SHAP ===
    # =======================================================
    print("\n--- XAI 1: Generating SHAP Explanations ---")
    
    X_train_tensors = torch.tensor(StandardScaler().fit_transform(X_train), dtype=torch.float32).to(DEVICE)
    X_test_tensors = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    
    background_sample = X_train_tensors[np.random.choice(X_train_tensors.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(final_model, background_sample)
    
    shap_values = explainer.shap_values(X_test_tensors)
    
    print("\nGenerating SHAP Summary Plot (Global Explanation)...")
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, class_names=class_names)
    
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
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    instance_to_explain_index = 0
    instance_to_explain = X_test_scaled[instance_to_explain_index]

    print(f"\nGenerating LIME explanation for the first test sample...")
    explanation = lime_explainer.explain_instance(
        data_row=instance_to_explain,
        predict_fn=predictor,
        num_features=len(feature_names)
    )

    explanation.save_to_file('lime_explanation.html')
    print("LIME explanation saved to lime_explanation.html")