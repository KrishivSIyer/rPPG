import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Ensure src and src/training directories are in sys.path
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TRAINING_DIR)
BASE_DIR = os.path.dirname(SRC_DIR)

for path in [SRC_DIR, TRAINING_DIR, BASE_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model import rPPGNet
from dataset import rPPGDataset
from heart_rate import bandpass_filter, extract_chrom_pulse, calculate_heart_rate
from evaluate import load_ground_truth


def reconstruct_full_waveform(model, dataset, full_length, window_size=150, stride=75):
    """
    Runs model inference over windowed samples and reconstructs full continuous signal via overlap-add.
    """
    device = next(model.parameters()).device
    model.eval()

    full_pred = np.zeros(full_length, dtype=np.float32)
    count_map = np.zeros(full_length, dtype=np.float32)

    with torch.no_grad():
        for i in range(len(dataset)):
            rgb_window, _ = dataset[i]  # rgb_window: (3, 150)
            rgb_tensor = rgb_window.unsqueeze(0).to(device)  # (1, 3, 150)

            pred = model(rgb_tensor).squeeze(0).cpu().numpy()  # (150,)

            start = i * stride
            end = start + window_size

            if end <= full_length:
                full_pred[start:end] += pred
                count_map[start:end] += 1.0

    # Avoid divide by zero
    count_map[count_map == 0] = 1.0
    reconstructed = full_pred / count_map
    return reconstructed


def evaluate_model_pipeline(
    rgb_csv_path: str = "results/subject1_rgb.csv",
    gt_path: str = "results/gtdump.xmp",
    model_path: str = "models/rppgnet_best.pth",
    fps: float = 30.0,
    output_plot: str = "results/model_vs_baseline.png"
):
    print("==================================================")
    print("       rPPGNet Model vs. Baseline Evaluation      ")
    print("==================================================")

    if not os.path.exists(rgb_csv_path):
        raise FileNotFoundError(f"RGB file not found: {rgb_csv_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground Truth file not found: {gt_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1. Load Data
    df = pd.read_csv(rgb_csv_path)
    rgb_signals = df[["R", "G", "B"]].values.astype(np.float32)
    gt_data = load_ground_truth(gt_path)
    gt_ppg_raw = gt_data["gt_ppg_signal"]

    # Interpolate GT PPG to match video frame count
    x_old = np.linspace(0, 1, len(gt_ppg_raw))
    x_new = np.linspace(0, 1, len(rgb_signals))
    gt_ppg = np.interp(x_new, x_old, gt_ppg_raw).astype(np.float32)

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = rPPGNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained rPPGNet model from: {model_path}")

    # 3. Model Inference & Signal Reconstruction
    dataset = rPPGDataset(rgb_signals, gt_ppg, window_size=150, stride=75)
    model_raw_signal = reconstruct_full_waveform(model, dataset, len(rgb_signals))
    model_pulse = bandpass_filter(model_raw_signal, fps, lowcut=0.75, highcut=2.5)

    # 4. Classical Baseline (CHROM) Signal Extraction
    chrom_pulse = extract_chrom_pulse(rgb_signals, fps, lowcut=0.75, highcut=2.5)

    # Ground Truth PPG (Normalized & Filtered)
    gt_pulse = bandpass_filter(gt_ppg - np.mean(gt_ppg), fps, lowcut=0.75, highcut=2.5)

    # 5. Heart Rate Estimation via FFT / Welch PSD
    model_bpm, model_freqs, model_psd, _ = calculate_heart_rate(model_pulse, fps)
    chrom_bpm, chrom_freqs, chrom_psd, _ = calculate_heart_rate(chrom_pulse, fps)
    gt_bpm, gt_freqs, gt_psd, _ = calculate_heart_rate(gt_pulse, fps)

    # 6. Metrics Calculation
    # Waveform Pearson Correlation
    r_chrom, _ = pearsonr(chrom_pulse, gt_pulse)
    r_model, _ = pearsonr(model_pulse, gt_pulse)

    # Heart Rate MAE (BPM)
    mae_chrom = abs(chrom_bpm - gt_bpm)
    mae_model = abs(model_bpm - gt_bpm)

    print("\n---------------- EVALUATION SUMMARY ----------------")
    print(f"{'Method':<20} | {'Heart Rate (BPM)':<16} | {'MAE (BPM)':<12} | {'Pearson r (Waveform)':<20}")
    print("-" * 75)
    print(f"{'Ground Truth (PPG)':<20} | {gt_bpm:<16.2f} | {'0.00':<12} | {'1.0000':<20}")
    print(f"{'CHROM (Baseline)':<20} | {chrom_bpm:<16.2f} | {mae_chrom:<12.2f} | {r_chrom:<20.4f}")
    print(f"{'rPPGNet (Learned)':<20} | {model_bpm:<16.2f} | {mae_model:<12.2f} | {r_model:<20.4f}")
    print("----------------------------------------------------\n")

    # 7. Visualization
    time = np.arange(len(rgb_signals)) / fps
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    # Z-score normalize waveforms for visual comparison on identical y-axis scale [-3, 3]
    def z_score(s):
        return (s - np.mean(s)) / (np.std(s) + 1e-8)

    gt_norm = z_score(gt_pulse)
    chrom_norm = z_score(chrom_pulse)
    model_norm = z_score(model_pulse)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle("rPPG Performance Comparison: Classical CHROM vs. Learned rPPGNet", fontsize=14, fontweight="bold")

    # Plot 1: Pulse Waveform Comparison (First 15 seconds)
    plot_samples = min(len(time), int(15 * fps))
    axes[0].plot(time[:plot_samples], gt_norm[:plot_samples], label="Ground Truth PPG", color="black", alpha=0.7, linewidth=1.5)
    axes[0].plot(time[:plot_samples], chrom_norm[:plot_samples], label=f"CHROM Baseline (r={r_chrom:.2f})", color="tab:blue", alpha=0.6, linestyle="--")
    axes[0].plot(time[:plot_samples], model_norm[:plot_samples], label=f"rPPGNet Model (r={r_model:.2f})", color="tab:red", linewidth=1.8)
    axes[0].set_title("Normalized Pulse Waveform Comparison (First 15 Seconds)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Normalized Amplitude (Z-score)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Welch Power Spectral Density (PSD) - Normalized to Peak Height 1.0
    valid_mask = (gt_freqs >= 0.75) & (gt_freqs <= 2.5)
    axes[1].plot(gt_freqs[valid_mask] * 60, gt_psd[valid_mask] / np.max(gt_psd[valid_mask]), label=f"GT Peak ({gt_bpm:.1f} BPM)", color="black", linewidth=1.5)
    axes[1].plot(chrom_freqs[valid_mask] * 60, chrom_psd[valid_mask] / np.max(chrom_psd[valid_mask]), label=f"CHROM Peak ({chrom_bpm:.1f} BPM)", color="tab:blue", alpha=0.6, linestyle="--")
    axes[1].plot(model_freqs[valid_mask] * 60, model_psd[valid_mask] / np.max(model_psd[valid_mask]), label=f"rPPGNet Peak ({model_bpm:.1f} BPM)", color="tab:red", linewidth=1.8)
    axes[1].set_title("Normalized Power Spectral Density (PSD) & Heart Rate Peak Alignment")
    axes[1].set_xlabel("Frequency (BPM)")
    axes[1].set_ylabel("Normalized Power Density")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Evaluation comparison plot saved to: {output_plot}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained rPPGNet against CHROM baseline.")
    parser.add_argument("--rgb", type=str, default=os.path.join(BASE_DIR, "results", "subject1_rgb.csv"), help="Path to RGB CSV.")
    parser.add_argument("--gt", type=str, default=os.path.join(BASE_DIR, "results", "gtdump.xmp"), help="Path to Ground Truth file.")
    parser.add_argument("--model", type=str, default=os.path.join(BASE_DIR, "models", "rppgnet_best.pth"), help="Path to trained model weights.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS.")
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "results", "model_vs_baseline.png"), help="Output plot path.")
    args = parser.parse_args()

    evaluate_model_pipeline(
        rgb_csv_path=args.rgb,
        gt_path=args.gt,
        model_path=args.model,
        fps=args.fps,
        output_plot=args.output
    )


if __name__ == "__main__":
    main()
