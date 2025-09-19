import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import weibull_min
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import os
import math
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd
import random

# --- 1. DATA AND MODEL SETUP ---
def create_digital_signal_dataset(num_samples_per_class=200, signal_length=256, num_classes=15):
    signals, ancillary_features, labels = [], [], []
    weibull_params = [
        (1.5, 0.5, 1.0), (2.0, 1.0, 1.2), (3.0, 1.5, 1.5), (0.8, 0.2, 0.8), (1.0, 0.5, 1.0),
        (2.5, 0.8, 1.1), (1.2, 1.2, 0.9), (3.5, 2.0, 1.8), (0.7, 0.1, 0.7), (1.8, 0.6, 1.3),
        (2.2, 1.1, 1.4), (1.0, 1.5, 1.0), (4.0, 2.5, 2.0), (0.9, 0.3, 0.85),(1.3, 0.4, 0.95)
    ]
    for class_idx, (alpha, beta, scale) in enumerate(weibull_params[:num_classes]):
        for _ in range(num_samples_per_class):
            weibull_dist = weibull_min(c=alpha, loc=beta, scale=scale)
            x = np.linspace(weibull_dist.ppf(0.01), weibull_dist.ppf(0.99), signal_length)
            signal_pulse = weibull_dist.pdf(x)
            signal_pulse /= signal_pulse.max()
            noise = np.random.normal(0, 0.05, signal_length)
            signal_pulse += noise
            signals.append(torch.tensor(signal_pulse, dtype=torch.float32))
            avg_amplitude, std_amplitude = np.mean(signal_pulse), np.std(signal_pulse)
            ancillary_features.append(torch.tensor([avg_amplitude, std_amplitude, class_idx], dtype=torch.float32))
            labels.append(class_idx)
    return signals, ancillary_features, labels

class MessyDataset(Dataset):
    def __init__(self, signals, ancillary_and_labels):
        self.signals = signals
        self.ancillary_and_labels = ancillary_and_labels
    def __len__(self):
        return len(self.signals)
    def __getitem__(self, idx):
        ancillary_data = self.ancillary_and_labels[idx]
        ancillary_dict = {'aux1': ancillary_data[0], 'aux2': ancillary_data[1], 'labels': ancillary_data[2]}
        return (self.signals[idx], ancillary_dict, {})

class ResNet1DCNN(nn.Module):
    def __init__(self, signal_length, num_ancillary_features, num_classes):
        super(ResNet1DCNN, self).__init__()
        self.input_dim = signal_length + num_ancillary_features
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.relu = nn.ReLU()
        self.block1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.output_layer = nn.Linear(128, self.num_classes)
    def forward(self, signal, ancillary_dict, junk_dict):
        ancillary = torch.stack([ancillary_dict['aux1'], ancillary_dict['aux2']], dim=1)
        combined = torch.cat((signal.view(signal.size(0), -1), ancillary), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.block1(x)
        x = self.block2(x)
        return {'predictions': self.output_layer(x)}

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, signal, aux1, aux2):
        ancillary_dict = {'aux1': aux1.squeeze(1), 'aux2': aux2.squeeze(1)}
        model_output_dict = self.model(signal, ancillary_dict, {})
        return list(model_output_dict.values())[0]

# --- 2. CORE ANALYSIS AND PLOTTING SUITE ---

def calculate_ig_with_incremental_baselines(inputs, target, calculator, baseline_config):
    all_attributions = []
    all_deltas = []
    baseline_values = np.arange(baseline_config[0], baseline_config[1] + baseline_config[2], baseline_config[2])

    for angle_val in baseline_values:
        signal_baseline = torch.zeros_like(inputs[0])
        aux1_baseline = torch.tensor([angle_val], dtype=torch.float32)
        aux2_baseline = torch.tensor([angle_val], dtype=torch.float32)
        baselines = (signal_baseline, aux1_baseline, aux2_baseline)

        attributions, delta = calculator.attribute(inputs, baselines=baselines, target=target, return_convergence_delta=True)
        all_attributions.append(torch.cat([t.flatten() for t in attributions]))
        all_deltas.append(delta.item())

    avg_attributions_flat = torch.mean(torch.stack(all_attributions), dim=0)
    avg_delta = np.mean(all_deltas)

    signal_attr = avg_attributions_flat[:inputs[0].numel()].reshape(inputs[0].shape).squeeze(0)
    ancillary_attr = avg_attributions_flat[inputs[0].numel():]
    return signal_attr, ancillary_attr, avg_delta

def log_quantitative_scores(signal_sal, ancillary_sal, delta, title, log_file):
    abs_signal_sal = np.abs(signal_sal.cpu().detach().numpy().flatten())
    abs_ancillary_sal = np.abs(ancillary_sal.cpu().detach().numpy())
    total_signal_score = np.sum(abs_signal_sal)
    top_10_percent_threshold = np.percentile(abs_signal_sal, 90)
    top_scores = abs_signal_sal[abs_signal_sal >= top_10_percent_threshold]
    avg_top_10_percent_score = np.mean(top_scores) if len(top_scores) > 0 else 0
    avg_all_timesteps_score = np.mean(abs_signal_sal)
    ancillary_1_score = abs_ancillary_sal[0]
    ancillary_2_score = abs_ancillary_sal[1]

    log_file.write(f"\n--- Quantitative Analysis for: {title} ---\n")
    log_file.write(f"Completeness Axiom (Convergence Delta): {delta:.6f}\n")
    log_file.write(f"Total Signal Attribution Score: {total_signal_score:.4f}\n")
    log_file.write(f"      Avg. Score of Top 10% Timesteps: {avg_top_10_percent_score:.4f}\n")
    log_file.write("--------------------------------------------------\n")
    log_file.write(f"      Avg. Score of All Timesteps (Signal): {avg_all_timesteps_score:.4f}\n")
    log_file.write(f"      Absolute Score (Ancillary 1): {ancillary_1_score:.4f}\n")
    log_file.write(f"      Absolute Score (Ancillary 2): {ancillary_2_score:.4f}\n")
    log_file.write("==================================================\n")

def plot_signal_with_saliency_bar_overlay(signal_pulse, saliency_scores, title, ax):
    signal_np = signal_pulse.cpu().detach().numpy()
    saliency_np = saliency_scores.cpu().detach().numpy().flatten()
    x_axis = np.arange(len(signal_np))
    saliency_pos = np.maximum(0, saliency_np)
    saliency_neg = np.minimum(0, saliency_np)

    ax.plot(x_axis, signal_np, 'k-', label='Digital Signal', zorder=2, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Amplitude", color='black')
    ax.set_xlabel("Time Step")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    ax2.bar(x_axis, saliency_pos, color='red', alpha=0.6, label='Positive Attr (Towards)', zorder=1)
    ax2.bar(x_axis, saliency_neg, color='blue', alpha=0.6, label='Negative Attr (Away)', zorder=1)
    ax2.set_ylabel("Attribution Score", color='black')
    ax2.legend(loc='upper right')

def plot_ancillary_saliency_map(ancillary_scores, ancillary_input_values, title, ax, feature_names):
    scores_np = np.abs(ancillary_scores.cpu().detach().numpy())
    values_np = ancillary_input_values.cpu().detach().numpy()

    ax.bar(feature_names, scores_np, color='skyblue')
    ax.set_title(title)
    ax.set_ylabel("Attribution Score")
    ax.grid(axis='y', linestyle='--')
    for i, score in enumerate(scores_np):
        value = values_np[i]
        display_text = f"Attr: {score:.4f}\nValue: {value:.2f}"
        ax.text(i, score, display_text, ha='center', va='bottom', fontsize=9)

def plot_attribution_shape(saliency_scores, title, ax):
    saliency_np = saliency_scores.cpu().detach().numpy().flatten()
    x_axis = np.arange(len(saliency_np))
    
    saliency_pos = np.maximum(0, saliency_np)
    saliency_neg = np.minimum(0, saliency_np)

    ax.bar(x_axis, saliency_pos, color='red', alpha=0.7, label='Positive Attr (Towards)')
    ax.bar(x_axis, saliency_neg, color='blue', alpha=0.7, label='Negative Attr (Away)')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Attribution Score")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.set_xlim(-0.5, len(saliency_np) - 0.5)

def generate_analysis_suite(log_dir, samples_to_plot_all, final_signals, final_ancillary, calculator, plot_type, class_names, baseline_config):
    os.makedirs(log_dir, exist_ok=True)
    print(f"--- Generating analysis suite for '{log_dir.split('/')[-1]}'")
    log_path = os.path.join(log_dir, 'analysis_log.txt')

    if len(samples_to_plot_all) > 9:
        samples_to_plot = random.sample(samples_to_plot_all, 9)
    else:
        samples_to_plot = samples_to_plot_all

    with open(log_path, 'w') as log_file:
        if not samples_to_plot:
            title = f"{log_dir.split('/')[-1].replace('_', ' ').title()}: No Samples Found"
            log_file.write(f"{title}\n")
            print(f"{title}. Skipping plot generation.")
            for name in ['individual_signals', 'individual_ancillary', 'individual_shapes', 'AGGREGATE_SUMMARY']:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No Samples Found", ha='center', va='center', fontsize=20)
                ax.axis('off')
                fig.savefig(os.path.join(log_dir, f'{name}.png'))
                plt.close(fig)
            return

        class_label_int = samples_to_plot[0][0]
        if plot_type == 'correct':
            title_prefix = f"Correctly Classified: {class_names[class_label_int]}"
        else:
            pred_label_int = samples_to_plot[0][1]
            title_prefix = f"Incorrect: Actual {class_names[class_label_int]}, Pred {class_names[pred_label_int]}"

        log_file.write(f"Analysis Log for: {title_prefix}\n")

        fig_main, axs_main = plt.subplots(3, 3, figsize=(24, 18))
        fig_anc, axs_anc = plt.subplots(3, 3, figsize=(24, 12), sharey=True)
        fig_hist, axs_hist = plt.subplots(3, 3, figsize=(24, 12), sharey=True)
        fig_main.suptitle(f'{title_prefix} - Signal & Saliency', fontsize=20)
        fig_anc.suptitle(f'{title_prefix} - Ancillary Saliency', fontsize=20)
        fig_hist.suptitle(f'{title_prefix} - Attribution Shape', fontsize=20)

        all_signal_sals, all_ancillary_sals = [], []
        for i, sample_info in enumerate(samples_to_plot):
            idx = sample_info[-1]
            title = f"Sample {i+1}"
            signal_input = final_signals[idx:idx+1].clone().detach().requires_grad_(True)
            ancillary_vals = final_ancillary[idx]
            aux1 = torch.tensor([ancillary_vals[0]], requires_grad=True)
            aux2 = torch.tensor([ancillary_vals[1]], requires_grad=True)

            signal_sal, ancillary_sal, delta = calculate_ig_with_incremental_baselines((signal_input, aux1, aux2), class_label_int, calculator, baseline_config)
            all_signal_sals.append(signal_sal)
            all_ancillary_sals.append(ancillary_sal)
            log_quantitative_scores(signal_sal, ancillary_sal, delta, f"{title_prefix}, {title}", log_file)

            plot_signal_with_saliency_bar_overlay(signal_input.squeeze(0), signal_sal, title, axs_main.flat[i])
            plot_ancillary_saliency_map(ancillary_sal, ancillary_vals, title, axs_anc.flat[i], ['Avg Amp', 'Std Amp'])
            plot_attribution_shape(signal_sal, title, axs_hist.flat[i])

        for i in range(len(samples_to_plot), 9):
            axs_main.flat[i].axis('off')
            axs_anc.flat[i].axis('off')
            axs_hist.flat[i].axis('off')

        fig_main.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig_main.savefig(os.path.join(log_dir, 'individual_signals.png'))
        fig_anc.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig_anc.savefig(os.path.join(log_dir, 'individual_ancillary.png'))
        fig_hist.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig_hist.savefig(os.path.join(log_dir, 'individual_shapes.png'))
        plt.close('all')

        agg_signal_sal = torch.mean(torch.stack(all_signal_sals), dim=0)
        agg_ancillary_sal = torch.mean(torch.stack(all_ancillary_sals), dim=0)
        fig_agg, axs_agg = plt.subplots(1, 3, figsize=(24, 7))
        fig_agg.suptitle(f'{title_prefix} - AGGREGATE SUMMARY', fontsize=20)
        avg_signal_for_plot = torch.mean(final_signals[[s[-1] for s in samples_to_plot]], dim=0)
        plot_signal_with_saliency_bar_overlay(avg_signal_for_plot, agg_signal_sal, "Aggregate Signal", axs_agg[0])
        plot_ancillary_saliency_map(agg_ancillary_sal, final_ancillary[[s[-1] for s in samples_to_plot]].mean(dim=0), "Aggregate Ancillary", axs_agg[1], ['Avg Amp', 'Std Amp'])
        plot_attribution_shape(agg_signal_sal, "Aggregate Shape", axs_agg[2])
        fig_agg.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_agg.savefig(os.path.join(log_dir, 'AGGREGATE_SUMMARY.png'))
        plt.close(fig_agg)
        print(f"Saved full analysis suite for '{log_dir.split('/')[-1]}'")

def generate_model_misconception_suite(log_dir, predicted_class_label, all_prediction_indices, final_signals, final_ancillary, final_labels, calculator, class_names, baseline_config):
    print(f"\n--- Generating Model Misconception suite for predictions of '{class_names[predicted_class_label]}' ---")

    samples_to_analyze = all_prediction_indices[:9]
    if not samples_to_analyze:
        print("No samples were predicted as this class. Skipping.")
        return

    all_signal_sals, all_ancillary_sals = [], []
    for idx in samples_to_analyze:
        signal_input = final_signals[idx:idx+1].clone().detach().requires_grad_(True)
        ancillary_vals = final_ancillary[idx]
        aux1 = torch.tensor([ancillary_vals[0]], requires_grad=True)
        aux2 = torch.tensor([ancillary_vals[1]], requires_grad=True)

        signal_sal, ancillary_sal, _ = calculate_ig_with_incremental_baselines((signal_input, aux1, aux2), predicted_class_label, calculator, baseline_config)
        all_signal_sals.append(signal_sal)
        all_ancillary_sals.append(ancillary_sal)

    title_prefix = f"Model's Misconception of {class_names[predicted_class_label]}"
    agg_signal_sal = torch.mean(torch.stack(all_signal_sals), dim=0)
    agg_ancillary_sal = torch.mean(torch.stack(all_ancillary_sals), dim=0)
    fig_agg, axs_agg = plt.subplots(1, 3, figsize=(24, 7))
    fig_agg.suptitle(f'{title_prefix} - AGGREGATE SUMMARY', fontsize=20)

    avg_signal_for_plot = torch.mean(final_signals[samples_to_analyze], dim=0)
    plot_signal_with_saliency_bar_overlay(avg_signal_for_plot, agg_signal_sal, "Aggregate Signal", axs_agg[0])
    plot_ancillary_saliency_map(agg_ancillary_sal, final_ancillary[samples_to_analyze].mean(dim=0), "Aggregate Ancillary", axs_agg[1], ['Avg Amp', 'Std Amp'])
    plot_attribution_shape(agg_signal_sal, "Aggregate Shape", axs_agg[2])

    fig_agg.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_agg.savefig(os.path.join(log_dir, 'AGGREGATE_MISCONCEPTION.png'))
    plt.close(fig_agg)
    print(f"Saved AGGREGATE MISCONCEPTION for '{class_names[predicted_class_label]}'")

def generate_ccba_analysis(log_dir, class_a_idx, class_b_idx, final_signals, final_ancillary, final_labels, calculator, class_names):
    print(f"\n--- RUNNING CCBA: Baseline Class '{class_names[class_a_idx]}' vs. Input Class '{class_names[class_b_idx]}' ---")
    os.makedirs(log_dir, exist_ok=True)

    class_b_samples = (final_labels == class_b_idx).nonzero(as_tuple=True)[0]
    if not class_b_samples.nelement():
        print(f"CCBA Warning: No samples found for Class B ({class_names[class_b_idx]}). Skipping.")
        return
    sample_b_idx = class_b_samples[0]

    signal_input = final_signals[sample_b_idx:sample_b_idx+1].clone().detach().requires_grad_(True)
    ancillary_vals = final_ancillary[sample_b_idx]
    aux1_input = torch.tensor([ancillary_vals[0]], requires_grad=True)
    aux2_input = torch.tensor([ancillary_vals[1]], requires_grad=True)

    class_a_mask = final_labels == class_a_idx
    if not class_a_mask.any():
        print(f"CCBA Warning: No samples found for baseline Class A ({class_names[class_a_idx]}). Skipping.")
        return

    signal_baseline = final_signals[class_a_mask].mean(dim=0, keepdim=True)
    ancillary_baselines = final_ancillary[class_a_mask].mean(dim=0)
    aux1_baseline = torch.tensor([ancillary_baselines[0]])
    aux2_baseline = torch.tensor([ancillary_baselines[1]])

    attributions, delta = calculator.attribute((signal_input, aux1_input, aux2_input), baselines=(signal_baseline, aux1_baseline, aux2_baseline), target=class_b_idx, return_convergence_delta=True)
    signal_sal, ancillary_sal = attributions[0].squeeze(0), torch.cat(attributions[1:])

    title = f"CCBA: Input '{class_names[class_b_idx]}' vs. Baseline '{class_names[class_a_idx]}'"
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle(title, fontsize=16)
    plot_signal_with_saliency_bar_overlay(signal_input.squeeze(0), signal_sal, "Signal Attribution", axs[0])
    plot_ancillary_saliency_map(ancillary_sal, ancillary_vals, "Ancillary Attribution", axs[1], ['Avg Amp', 'Std Amp'])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(log_dir, f"CCBA_{class_names[class_b_idx]}_vs_{class_names[class_a_idx]}.png"))
    plt.close(fig)
    print(f"Saved CCBA plot to {log_dir}")

def generate_signal_variance_log(log_dir, final_signals, final_labels, class_names):
    print("\n--- Generating Signal Intra-Class Variance Report ---")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "_SIGNAL_VARIANCE_REPORT.txt")

    with open(log_path, 'w') as f:
        f.write("--- Signal Intra-Class Variance Report ---\n")
        f.write("Calculates the average standard deviation of signal shapes within each class.\n")
        f.write("A higher score means the signals for that class are less consistent.\n")
        f.write("="*60 + "\n\n")

        for i, class_name in enumerate(class_names):
            class_mask = final_labels == i
            if not class_mask.any():
                f.write(f"Class: {class_name}\n  -> No samples found in dataset.\n\n")
                continue
            class_signals = final_signals[class_mask]
            std_dev_per_timestep = class_signals.std(dim=0)
            avg_std_dev_score = std_dev_per_timestep.mean().item()
            f.write(f"Class: {class_name}\n  -> Average Signal Shape StDev: {avg_std_dev_score:.6f}\n\n")
    print(f"Saved signal variance report to {log_path}")

def calculate_and_save_metrics(true_labels, predictions, log_dir, experiment_name, class_names):
    print(f"--- Calculating performance metrics for {experiment_name} ---")
    os.makedirs(log_dir, exist_ok=True)

    num_classes = len(class_names)
    accuracy = accuracy_score(true_labels, predictions)
    report_dict = classification_report(true_labels, predictions, labels=np.arange(num_classes), target_names=class_names, zero_division=0, output_dict=True)

    report_path = os.path.join(log_dir, "metrics_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"--- Performance Metrics for Experiment: {experiment_name} ---\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(true_labels, predictions, labels=np.arange(num_classes), target_names=class_names, zero_division=0))
    print(f"Saved metrics report to {report_path}")

    cm = confusion_matrix(true_labels, predictions, labels=np.arange(num_classes))
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {experiment_name}', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(log_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    return accuracy, report_dict

# --- 3. MAIN EXECUTION ---
if __name__ == '__main__':
    CLASS_NAMES = ["iPhone", "Samsung", "Pixel", "Motorola", "Nokia", "Sony", "LG", "Huawei", "OnePlus", "Xiaomi", "Oppo", "Vivo", "Realme", "Asus", "ZTE"]
    NUM_CLASSES = len(CLASS_NAMES)
    BASE_LOG_DIR = "ig_advanced_experiments"
    np.random.seed(42)
    torch.manual_seed(42)

    ANCILLARY_BASELINE_CONFIG = [0, 180, 10]
    CLASSES_TO_COMPARE_CCBA = [(7, 9), (2, 9), (1, 10)]
    CLASSES_TO_COMPARE_INCORRECT = [(7, 9), (2, 9), (1, 10)]

    signals, ancillary_features, _ = create_digital_signal_dataset(num_classes=NUM_CLASSES)
    dataloader = DataLoader(MessyDataset(signals, ancillary_features), batch_size=64)
    final_signals, final_ancillary, final_labels = [], [], []
    for sig_b, anc_d, _ in dataloader:
        final_signals.append(sig_b)
        final_ancillary.append(torch.stack([anc_d['aux1'], anc_d['aux2']], dim=1))
        final_labels.append(anc_d['labels'].long())
    final_signals = torch.cat(final_signals)
    final_ancillary = torch.cat(final_ancillary)
    final_labels = torch.cat(final_labels)

    generate_signal_variance_log(BASE_LOG_DIR, final_signals, final_labels, CLASS_NAMES)

    model = ResNet1DCNN(256, 2, NUM_CLASSES)
    model.eval()
    wrapped_model = ModelWrapper(model)
    ig_calculator = IntegratedGradients(wrapped_model)

    signals_zeros = torch.zeros_like(final_signals)
    shuffled_indices = torch.randperm(final_ancillary.size(0))
    shuffled_ancillary = final_ancillary[shuffled_indices]
    ancillary_no_aux1 = final_ancillary.clone()
    ancillary_no_aux1[:, 0] = shuffled_ancillary[:, 0]
    ancillary_no_aux2 = final_ancillary.clone()
    ancillary_no_aux2[:, 1] = shuffled_ancillary[:, 1]
    experiments = {
        "CONTROL": (final_signals, final_ancillary),
        "NO_SIGNAL": (signals_zeros, final_ancillary),
        "NO_ANCILLARY_1": (final_signals, ancillary_no_aux1),
        "NO_ANCILLARY_2": (final_signals, ancillary_no_aux2)
    }
    all_experiment_metrics = {}

    for name, (exp_signals, exp_ancillary) in experiments.items():
        print(f"\n\n--- RUNNING EXPERIMENT: {name} ---")
        with torch.no_grad():
            output_tensor = wrapped_model(exp_signals, exp_ancillary[:, 0], exp_ancillary[:, 1])
            predictions = torch.argmax(output_tensor, dim=1)

        metrics_log_dir = os.path.join(BASE_LOG_DIR, name)
        accuracy, report_dict = calculate_and_save_metrics(final_labels.numpy(), predictions.numpy(), metrics_log_dir, name, CLASS_NAMES)
        all_experiment_metrics[name] = {'accuracy': accuracy, 'report': report_dict}

        correct_samples, incorrect_samples = {i: [] for i in range(NUM_CLASSES)}, {}
        for i in range(len(final_labels)):
            true_label, pred_label = final_labels[i].item(), predictions[i].item()
            if true_label in correct_samples:
                if pred_label == true_label:
                    correct_samples[true_label].append((true_label, i))
                else:
                    key = (true_label, pred_label)
                    if key not in incorrect_samples:
                        incorrect_samples[key] = []
                    incorrect_samples[key].append((true_label, pred_label, i))

        all_predicted_indices = {i: [] for i in range(NUM_CLASSES)}
        for i in range(len(predictions)):
            all_predicted_indices[predictions[i].item()].append(i)

        for class_label, samples in correct_samples.items():
            log_dir = os.path.join(BASE_LOG_DIR, name, f"correct_predictions/Class_{class_label}")
            generate_analysis_suite(log_dir, samples, final_signals, final_ancillary, ig_calculator, 'correct', CLASS_NAMES, ANCILLARY_BASELINE_CONFIG)

        for true_label, pred_label in CLASSES_TO_COMPARE_INCORRECT:
            key = (true_label, pred_label)
            samples = incorrect_samples.get(key, [])
            log_dir = os.path.join(BASE_LOG_DIR, name, f"incorrect_predictions/Actual_{true_label}_Pred_{pred_label}")
            generate_analysis_suite(log_dir, samples, final_signals, final_ancillary, ig_calculator, 'incorrect', CLASS_NAMES, ANCILLARY_BASELINE_CONFIG)

            if not correct_samples.get(pred_label):
                print(f"\nWARNING: No correct samples found for the predicted class ({CLASS_NAMES[pred_label]}).")
                print("Running a 'Model Misconception' analysis instead...")
                generate_model_misconception_suite(log_dir, pred_label, all_predicted_indices[pred_label], final_signals, final_ancillary, final_labels, ig_calculator, CLASS_NAMES, ANCILLARY_BASELINE_CONFIG)

    ccba_log_dir = os.path.join(BASE_LOG_DIR, "CCBA_Analysis")
    for class_a, class_b in CLASSES_TO_COMPARE_CCBA:
        generate_ccba_analysis(ccba_log_dir, class_a, class_b, final_signals, final_ancillary, final_labels, ig_calculator, CLASS_NAMES)

    summary_data = {}
    for name, metrics in all_experiment_metrics.items():
        summary_data[name] = {'Overall Accuracy': metrics['accuracy']}
        for class_name in CLASS_NAMES:
            f1_score = metrics['report'].get(class_name, {}).get('f1-score', 0.0)
            summary_data[name][f'F1-Score {class_name}'] = f1_score
    summary_df = pd.DataFrame(summary_data).T
    summary_path = os.path.join(BASE_LOG_DIR, "_AGGREGATE_METRICS_REPORT.txt")
    with open(summary_path, 'w') as f:
        f.write("--- Perturbation Experiment Summary Report ---\n\n")
        f.write(summary_df.to_string())
    print(f"\n\nAggregate summary report saved to {summary_path}")
    print("All advanced Integrated Gradients experiments complete.")