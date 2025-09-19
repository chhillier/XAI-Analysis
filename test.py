import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import weibull_min
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        ancillary_dict = {'aux1': aux1.view(-1), 'aux2': aux2.view(-1)}
        model_output_dict = self.model(signal, ancillary_dict, {})
        return list(model_output_dict.values())[0]

# --- 2. CORE ANALYSIS AND PLOTTING SUITE ---

def calculate_ig_with_opposite_baseline(inputs, target, calculator, n_steps):
    signal_input, aux1_input, aux2_input = inputs
    
    signal_baseline = torch.zeros_like(signal_input)
    angle1 = aux1_input.item()
    angle2 = aux2_input.item()
    opposite_angle1 = (angle1 + 180) % 360
    opposite_angle2 = (angle2 + 180) % 360
    aux1_baseline = torch.tensor([[opposite_angle1]], dtype=torch.float32)
    aux2_baseline = torch.tensor([[opposite_angle2]], dtype=torch.float32)
    baselines = (signal_baseline, aux1_baseline, aux2_baseline)

    attributions, delta = calculator.attribute(inputs, baselines=baselines, target=target, n_steps=n_steps, return_convergence_delta=True)
    
    model = calculator.forward_func
    with torch.no_grad():
        output_input = model(signal_input, aux1_input, aux2_input)
        output_baseline = model(signal_baseline, aux1_baseline, aux2_baseline)
    
    logit_input = output_input[0, target].item()
    logit_baseline = output_baseline[0, target].item()
    total_change = logit_input - logit_baseline

    signal_attr = attributions[0].squeeze(0)
    ancillary_attr = torch.cat(attributions[1:]).flatten()

    return signal_attr, ancillary_attr, delta, total_change

def log_quantitative_scores(signal_sal, ancillary_sal, delta, total_change, title, log_file):
    sum_attributions = total_change - delta.item()
    completeness_percent = 0.0
    if abs(total_change) > 1e-6:
        completeness_percent = (1 - (abs(delta.item()) / abs(total_change))) * 100

    log_file.write(f"\n--- Quantitative Analysis for: {title} ---\n")
    log_file.write("--- Completeness Axiom Check ---\n")
    log_file.write(f"Total Model Output Change to Explain:   {total_change:.4f}\n")
    log_file.write(f"Sum of All Feature Attributions:        {sum_attributions:.4f}\n")
    log_file.write("-" * 43 + "\n")
    log_file.write(f"Unexplained Remainder (Delta):            {delta.item():.4f}\n")
    log_file.write(f"=> Explanation Completeness:               {completeness_percent:.2f}%\n")
    log_file.write("="*43 + "\n")
    
    abs_signal_sal = np.abs(signal_sal.cpu().detach().numpy().flatten())
    abs_ancillary_sal = np.abs(ancillary_sal.cpu().detach().numpy())
    total_signal_score = np.sum(abs_signal_sal)
    ancillary_1_score = abs_ancillary_sal[0]
    ancillary_2_score = abs_ancillary_sal[1]
    log_file.write(f"Total Signal Attribution Score: {total_signal_score:.4f}\n")
    log_file.write(f"Absolute Score (Ancillary 1): {ancillary_1_score:.4f}\n")
    log_file.write(f"Absolute Score (Ancillary 2): {ancillary_2_score:.4f}\n")

def plot_signal_with_saliency_bar_overlay(signal_pulse, saliency_scores, title, ax):
    signal_np = signal_pulse.cpu().detach().numpy()
    saliency_np = saliency_scores.cpu().detach().numpy().flatten()
    x_axis = np.arange(len(signal_np))
    saliency_pos_plot = np.maximum(0, saliency_np)
    saliency_neg_plot = np.abs(np.minimum(0, saliency_np))
    total_score = np.sum(np.abs(saliency_np))
    full_title = f"{title} | Score: {total_score:.4f}"
    ax.set_title(full_title)
    ax.plot(x_axis, signal_np, 'k-', label='Digital Signal', zorder=2, linewidth=1.5)
    ax.set_ylabel("Amplitude", color='black')
    ax.set_xlabel("Time Step")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.bar(x_axis, saliency_pos_plot, color='red', alpha=0.6, label='Positive Attr (Towards)', zorder=1, bottom=0)
    ax2.bar(x_axis, saliency_neg_plot, color='blue', alpha=0.6, label='Negative Attr (Away)', zorder=1, bottom=0)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7, zorder=0)
    ax2.set_ylabel("Attribution Score (Magnitude)", color='black')
    red_patch = mpatches.Patch(color='red', label='Positive (Towards)')
    blue_patch = mpatches.Patch(color='blue', label='Negative (Away)')
    ax2.legend(handles=[red_patch, blue_patch], loc='upper right')

def plot_ancillary_saliency_map(ancillary_scores, ancillary_input_values, title, ax, feature_names):
    original_scores_np = ancillary_scores.cpu().detach().numpy()
    abs_scores_np = np.abs(original_scores_np)
    values_np = ancillary_input_values.cpu().detach().numpy()
    colors = ['red' if score >= 0 else 'blue' for score in original_scores_np]
    ax.bar(feature_names, abs_scores_np, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Attribution Score (Magnitude)")
    ax.grid(axis='y', linestyle='--')
    for i, score in enumerate(abs_scores_np):
        original_score = original_scores_np[i]
        value = values_np[i]
        display_text = f"Attr: {original_score:.4f}\nValue: {value:.2f}"
        ax.text(i, score, display_text, ha='center', va='bottom', fontsize=9)
    red_patch = mpatches.Patch(color='red', label='Positive (Towards)')
    blue_patch = mpatches.Patch(color='blue', label='Negative (Away)')
    ax.legend(handles=[red_patch, blue_patch])

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

def generate_analysis_suite(log_dir, samples_to_plot_all, final_signals, final_ancillary, calculator, plot_type, class_names, n_steps):
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
            for name in ['individual_signals', 'individual_ancillary', 'individual_shapes']:
                fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No Samples Found", ha='center', va='center', fontsize=20)
                ax.axis('off'); fig.savefig(os.path.join(log_dir, f'{name}.png')); plt.close(fig)
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
        fig_main.suptitle(f'{title_prefix} - Signal & Saliency (9 Random Samples)', fontsize=20)
        fig_anc.suptitle(f'{title_prefix} - Ancillary Saliency (9 Random Samples)', fontsize=20)
        fig_hist.suptitle(f'{title_prefix} - Attribution Shape (9 Random Samples)', fontsize=20)
        all_deltas = []
        for i, sample_info in enumerate(samples_to_plot):
            idx = sample_info[-1]
            title = f"Sample {i+1}"
            signal_input = final_signals[idx:idx+1].clone().detach().requires_grad_(True)
            ancillary_vals = final_ancillary[idx]
            aux1 = torch.tensor([[ancillary_vals[0]]], requires_grad=True)
            aux2 = torch.tensor([[ancillary_vals[1]]], requires_grad=True)
            signal_sal, ancillary_sal, delta, total_change = calculate_ig_with_opposite_baseline((signal_input, aux1, aux2), class_label_int, calculator, n_steps)
            all_deltas.append(delta.item())
            log_quantitative_scores(signal_sal, ancillary_sal, delta, total_change, f"{title_prefix}, {title}", log_file)
            plot_signal_with_saliency_bar_overlay(signal_input.squeeze(0), signal_sal, title, axs_main.flat[i])
            plot_ancillary_saliency_map(ancillary_sal, ancillary_vals, title, axs_anc.flat[i], ['Avg Amp', 'Std Amp'])
            plot_attribution_shape(signal_sal, title, axs_hist.flat[i])
        for i in range(len(samples_to_plot), 9):
            axs_main.flat[i].axis('off')
            axs_anc.flat[i].axis('off')
            axs_hist.flat[i].axis('off')
        fig_main.tight_layout(rect=[0, 0.03, 1, 0.97]); fig_main.savefig(os.path.join(log_dir, 'individual_signals.png'))
        fig_anc.tight_layout(rect=[0, 0.03, 1, 0.97]); fig_anc.savefig(os.path.join(log_dir, 'individual_ancillary.png'))
        fig_hist.tight_layout(rect=[0, 0.03, 1, 0.97]); fig_hist.savefig(os.path.join(log_dir, 'individual_shapes.png'))
        plt.close('all')
        if all_deltas:
            avg_delta = np.mean(all_deltas)
            log_file.write("\n\n--- SUMMARY (from 9 random samples) ---\n")
            log_file.write(f"Average Completeness Delta: {avg_delta:.6f}\n")

def generate_full_class_aggregate_report(log_dir, class_samples, class_name, calculator, final_signals, final_ancillary, final_labels, n_steps):
    print(f"--- Generating Full Aggregate Report for Class: {class_name} ---")
    os.makedirs(log_dir, exist_ok=True)
    if not class_samples:
        return

    all_signal_sals, all_ancillary_sals, all_deltas, all_total_changes, all_conf_paths, all_aha_percents = [], [], [], [], [], []
    class_label_int = class_samples[0][0]

    for sample_info in class_samples:
        idx = sample_info[-1]
        signal = final_signals[idx]
        ancillary = final_ancillary[idx]
        signal_input = signal.unsqueeze(0).clone().detach().requires_grad_(True)
        aux1 = torch.tensor([[ancillary[0]]], requires_grad=True)
        aux2 = torch.tensor([[ancillary[1]]], requires_grad=True)
        signal_sal, ancillary_sal, delta, total_change = calculate_ig_with_opposite_baseline((signal_input, aux1, aux2), class_label_int, calculator, n_steps)
        all_signal_sals.append(signal_sal)
        all_ancillary_sals.append(ancillary_sal)
        all_deltas.append(delta.item())
        all_total_changes.append(total_change)
        all_conf_paths.append(_get_confidence_path(signal, ancillary, class_label_int, calculator))
        aha_moment = _calculate_single_aha_moment(signal, ancillary, class_label_int, calculator)
        if aha_moment is not None:
            all_aha_percents.append(aha_moment)

    avg_signal_sal = torch.mean(torch.stack(all_signal_sals), dim=0)
    avg_ancillary_sal = torch.mean(torch.stack(all_ancillary_sals), dim=0)
    avg_delta = np.mean(all_deltas)
    avg_total_change = np.mean(all_total_changes)
    avg_conf_path = np.mean(all_conf_paths, axis=0)
    avg_aha = np.mean(all_aha_percents) if all_aha_percents else None
    
    report_path = os.path.join(log_dir, f"_FULL_AGGREGATE_REPORT_{class_name}.txt")
    with open(report_path, 'w') as f:
        f.write(f"--- Full Aggregate Analysis Report for Class: {class_name} ---\n")
        f.write(f"Aggregated over {len(class_samples)} samples.\n")
        f.write("="*60 + "\n")
        sum_attributions = avg_total_change - avg_delta
        completeness_percent = 0.0
        if abs(avg_total_change) > 1e-6:
            completeness_percent = (1 - (abs(avg_delta) / abs(avg_total_change))) * 100
        f.write("--- Average Completeness Axiom Check ---\n")
        f.write(f"Avg. Total Model Output Change:   {avg_total_change:.4f}\n")
        f.write(f"Avg. Sum of Attributions:         {sum_attributions:.4f}\n")
        f.write(f"Avg. Unexplained Remainder (Delta): {avg_delta:.4f}\n")
        f.write(f"=> Explanation Completeness:       {completeness_percent:.2f}%\n")
        if avg_aha is not None:
            f.write(f"Average 'AHA!' Moment:            {avg_aha:.1f}%\n")

    fig_agg, axs_agg = plt.subplots(2, 2, figsize=(20, 14))
    fig_agg.suptitle(f'Full Aggregate for {class_name} ({len(class_samples)} samples)', fontsize=20)
    class_indices = [s[-1] for s in class_samples]
    avg_signal_for_plot = torch.mean(final_signals[class_indices], dim=0)
    
    plot_signal_with_saliency_bar_overlay(avg_signal_for_plot, avg_signal_sal, "Aggregate Signal", axs_agg[0, 0])
    plot_ancillary_saliency_map(avg_ancillary_sal, final_ancillary[class_indices].mean(dim=0), "Aggregate Ancillary", axs_agg[0, 1], ['Avg Amp', 'Std Amp'])
    plot_attribution_shape(avg_signal_sal, "Aggregate Shape", axs_agg[1, 0])
    plot_confidence_curve(axs_agg[1, 1], avg_conf_path, avg_aha, class_name)
    
    fig_agg.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = os.path.join(log_dir, f"_FULL_AGGREGATE_PLOT_{class_name}.png")
    fig_agg.savefig(plot_path)
    plt.close(fig_agg)
    print(f"Saved full aggregate report and 4-part plot for {class_name} to its directory.")

def _get_confidence_path(signal, ancillary, target_class, calculator, num_steps=20):
    model = calculator.forward_func
    softmax = nn.Softmax(dim=1)
    signal_baseline = torch.zeros_like(signal)
    angle1, angle2 = ancillary[0].item(), ancillary[1].item()
    opposite_angle1, opposite_angle2 = (angle1 + 180) % 360, (angle2 + 180) % 360
    path_confidences = []
    for i in range(num_steps + 1):
        alpha = i / num_steps
        interpolated_signal = signal_baseline + alpha * (signal - signal_baseline)
        diff1 = (angle1 - opposite_angle1 + 180) % 360 - 180
        interp_angle1 = (opposite_angle1 + alpha * diff1 + 360) % 360
        diff2 = (angle2 - opposite_angle2 + 180) % 360 - 180
        interp_angle2 = (opposite_angle2 + alpha * diff2 + 360) % 360
        interp_aux1 = torch.tensor([[interp_angle1]], dtype=torch.float32)
        interp_aux2 = torch.tensor([[interp_angle2]], dtype=torch.float32)
        with torch.no_grad():
            outputs = model(interpolated_signal.unsqueeze(0), interp_aux1, interp_aux2)
            confidence_scores = softmax(outputs)
            target_confidence = confidence_scores[0, target_class].item()
            path_confidences.append(target_confidence)
    return path_confidences

def plot_confidence_curve(ax, avg_confidence_path, avg_aha_moment_percent, class_name):
    num_steps = len(avg_confidence_path) - 1
    x_axis = np.linspace(0, 100, num_steps + 1)
    ax.plot(x_axis, avg_confidence_path, marker='.', linestyle='-', color='purple')
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8)
    ax.text(99, 0.51, '50% Confidence', va='bottom', ha='right', color='grey', fontsize=9)
    if avg_aha_moment_percent is not None:
        ax.axvline(avg_aha_moment_percent, color='green', linestyle='--', linewidth=1.2)
        ax.text(avg_aha_moment_percent + 2, 0.2, f'Avg. AHA! Moment\n({avg_aha_moment_percent:.1f}%)', color='green', fontsize=9)
    ax.set_title(f"Average Confidence Curve for {class_name}")
    ax.set_xlabel("Path Percentage (from Baseline to Input)")
    ax.set_ylabel("Model Confidence")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.6)

def generate_model_misconception_suite(log_dir, predicted_class_label, all_prediction_indices, final_signals, final_ancillary, final_labels, calculator, class_names, n_steps):
    print(f"\n--- Generating Model Misconception suite for predictions of '{class_names[predicted_class_label]}' ---")
    samples_to_analyze = all_prediction_indices[:9]
    if not samples_to_analyze:
        print("No samples were predicted as this class. Skipping.")
        return
    all_signal_sals, all_ancillary_sals = [], []
    for idx in samples_to_analyze:
        signal_input = final_signals[idx:idx+1].clone().detach().requires_grad_(True)
        ancillary_vals = final_ancillary[idx]
        aux1 = torch.tensor([[ancillary_vals[0]]], requires_grad=True)
        aux2 = torch.tensor([[ancillary_vals[1]]], requires_grad=True)
        signal_sal, ancillary_sal, _, _ = calculate_ig_with_opposite_baseline((signal_input, aux1, aux2), predicted_class_label, calculator, n_steps)
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

def generate_ccba_analysis(log_dir, class_a_idx, class_b_idx, final_signals, final_ancillary, final_labels, calculator, class_names, n_steps):
    print(f"\n--- RUNNING CCBA: Baseline Class '{class_names[class_a_idx]}' vs. Input Class '{class_names[class_b_idx]}' ---")
    os.makedirs(log_dir, exist_ok=True)
    class_b_samples = (final_labels == class_b_idx).nonzero(as_tuple=True)[0]
    if not class_b_samples.nelement():
        print(f"CCBA Warning: No samples found for Class B ({class_names[class_b_idx]}). Skipping.")
        return
    sample_b_idx = class_b_samples[0]
    signal_input = final_signals[sample_b_idx:sample_b_idx+1].clone().detach().requires_grad_(True)
    ancillary_vals = final_ancillary[sample_b_idx]
    aux1_input = torch.tensor([[ancillary_vals[0]]], requires_grad=True)
    aux2_input = torch.tensor([[ancillary_vals[1]]], requires_grad=True)
    class_a_mask = final_labels == class_a_idx
    if not class_a_mask.any():
        print(f"CCBA Warning: No samples found for baseline Class A ({class_names[class_a_idx]}). Skipping.")
        return
    signal_baseline = final_signals[class_a_mask].mean(dim=0, keepdim=True)
    ancillary_baselines = final_ancillary[class_a_mask].mean(dim=0)
    aux1_baseline = torch.tensor([[ancillary_baselines[0]]])
    aux2_baseline = torch.tensor([[ancillary_baselines[1]]])
    attributions, delta = calculator.attribute((signal_input, aux1_input, aux2_input), baselines=(signal_baseline, aux1_baseline, aux2_baseline), target=class_b_idx, n_steps=n_steps, return_convergence_delta=True)
    signal_sal, ancillary_sal = attributions[0].squeeze(0), torch.cat(attributions[1:]).flatten()
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

def _calculate_single_aha_moment(signal, ancillary, target_class, calculator, num_steps=20):
    model = calculator.forward_func
    signal_baseline = torch.zeros_like(signal)
    angle1 = ancillary[0].item()
    angle2 = ancillary[1].item()
    opposite_angle1 = (angle1 + 180) % 360
    opposite_angle2 = (angle2 + 180) % 360
    for i in range(num_steps + 1):
        alpha = i / num_steps
        interpolated_signal = signal_baseline + alpha * (signal - signal_baseline)
        diff1 = (angle1 - opposite_angle1 + 180) % 360 - 180
        interp_angle1 = (opposite_angle1 + alpha * diff1 + 360) % 360
        diff2 = (angle2 - opposite_angle2 + 180) % 360 - 180
        interp_angle2 = (opposite_angle2 + alpha * diff2 + 360) % 360
        interp_aux1 = torch.tensor([[interp_angle1]], dtype=torch.float32)
        interp_aux2 = torch.tensor([[interp_angle2]], dtype=torch.float32)
        with torch.no_grad():
            outputs = model(interpolated_signal.unsqueeze(0), interp_aux1, interp_aux2)
            prediction_idx = torch.argmax(outputs, dim=1).item()
        if prediction_idx == target_class:
            return alpha * 100
    return None

def generate_aha_moment_report(log_dir, final_signals, final_ancillary, final_labels, calculator, class_names, num_samples_per_class=5):
    print("\n--- Generating Average 'AHA!' Moment Report ---")
    os.makedirs(log_dir, exist_ok=True)
    report_path = os.path.join(log_dir, "_AHA_MOMENT_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("--- Average 'AHA!' Moment Report ---\n")
        f.write("Calculates the average point along the baseline-to-input path where the model's prediction flips to the correct class.\n")
        f.write("A lower percentage indicates a more robustly learned class.\n")
        f.write("="*80 + "\n\n")
        for class_idx, class_name in enumerate(class_names):
            correct_indices = (final_labels == class_idx).nonzero(as_tuple=True)[0]
            if not correct_indices.nelement():
                f.write(f"Class: {class_name}\n  -> No correct samples found to analyze.\n\n")
                continue
            if len(correct_indices) > num_samples_per_class:
                sample_indices = random.sample(correct_indices.tolist(), num_samples_per_class)
            else:
                sample_indices = correct_indices.tolist()
            aha_percentages = []
            for sample_idx in sample_indices:
                signal = final_signals[sample_idx]
                ancillary = final_ancillary[sample_idx]
                aha_percent = _calculate_single_aha_moment(signal, ancillary, class_idx, calculator)
                if aha_percent is not None:
                    aha_percentages.append(aha_percent)
            if aha_percentages:
                avg_aha = np.mean(aha_percentages)
                result_str = f"Class: {class_name}\n  -> Avg. AHA! Moment: {avg_aha:.1f}%\n\n"
                f.write(result_str)
                print(f"  -> Avg. AHA! Moment for {class_name}: {avg_aha:.1f}%")
            else:
                f.write(f"Class: {class_name}\n  -> Model never flipped to correct prediction for tested samples.\n\n")
                print(f"  -> Model never flipped to correct prediction for tested samples of {class_name}.")
    print(f"Saved AHA! Moment report to {report_path}")

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
    BASE_LOG_DIR = "ig_final_experiments"
    IG_STEPS = 200 # <-- NEW SETTING
    np.random.seed(42)
    torch.manual_seed(42)

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

    generate_aha_moment_report(BASE_LOG_DIR, final_signals, final_ancillary, final_labels, ig_calculator, CLASS_NAMES)

    signals_zeros = torch.zeros_like(final_signals)
    shuffled_indices = torch.randperm(final_ancillary.size(0))
    shuffled_ancillary = final_ancillary[shuffled_indices]
    ancillary_no_aux1 = final_ancillary.clone()
    ancillary_no_aux1[:, 0] = shuffled_ancillary[:, 0]
    ancillary_no_aux2 = final_ancillary.clone()
    ancillary_no_aux2[:, 1] = shuffled_ancillary[:, 1]
    ancillary_shuffled = shuffled_ancillary.clone()
    ancillary_only_aux1 = final_ancillary.clone()
    ancillary_only_aux1[:, 1] = shuffled_ancillary[:, 1]
    ancillary_only_aux2 = final_ancillary.clone()
    ancillary_only_aux2[:, 0] = shuffled_ancillary[:, 0]
    experiments = {
        "CONTROL": (final_signals, final_ancillary),
        "NO_SIGNAL": (signals_zeros, final_ancillary),
        "NO_ANCILLARY_1": (final_signals, ancillary_no_aux1),
        "NO_ANCILLARY_2": (final_signals, ancillary_no_aux2),
        "ONLY_SIGNAL": (final_signals, ancillary_shuffled),
        "ONLY_ANCILLARY_1": (signals_zeros, ancillary_only_aux1),
        "ONLY_ANCILLARY_2": (signals_zeros, ancillary_only_aux2),
        "ALL_PERTURBED": (signals_zeros, ancillary_shuffled)
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
            generate_analysis_suite(log_dir, samples, final_signals, final_ancillary, ig_calculator, 'correct', CLASS_NAMES, IG_STEPS)
        
        print(f"\n--- Generating Full Class Aggregate Reports for Experiment: {name} ---")
        for class_label, samples in correct_samples.items():
            if samples:
                log_dir = os.path.join(BASE_LOG_DIR, name, f"correct_predictions/Class_{class_label}")
                class_name = CLASS_NAMES[class_label]
                generate_full_class_aggregate_report(log_dir, samples, class_name, ig_calculator, final_signals, final_ancillary, final_labels, IG_STEPS)

        for true_label, pred_label in CLASSES_TO_COMPARE_INCORRECT:
            key = (true_label, pred_label)
            samples = incorrect_samples.get(key, [])
            log_dir = os.path.join(BASE_LOG_DIR, name, f"incorrect_predictions/Actual_{true_label}_Pred_{pred_label}")
            generate_analysis_suite(log_dir, samples, final_signals, final_ancillary, ig_calculator, 'incorrect', CLASS_NAMES, IG_STEPS)
            if not correct_samples.get(pred_label):
                print(f"\nWARNING: No correct samples found for the predicted class ({CLASS_NAMES[pred_label]}).")
                print("Running a 'Model Misconception' analysis instead...")
                generate_model_misconception_suite(log_dir, pred_label, all_predicted_indices[pred_label], final_signals, final_ancillary, final_labels, ig_calculator, CLASS_NAMES, IG_STEPS)

    ccba_log_dir = os.path.join(BASE_LOG_DIR, "CCBA_Analysis")
    for class_a, class_b in CLASSES_TO_COMPARE_CCBA:
        generate_ccba_analysis(ccba_log_dir, class_a, class_b, final_signals, final_ancillary, final_labels, ig_calculator, CLASS_NAMES, IG_STEPS)

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