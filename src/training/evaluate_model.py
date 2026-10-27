import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def pearsonr_np(x: np.ndarray, y: np.ndarray):
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


# Ensure src and src/training directories are in sys.path
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.dirname(TRAINING_DIR)
BASE_DIR     = os.path.dirname(SRC_DIR)

for path in [SRC_DIR, TRAINING_DIR, BASE_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model   import rPPGNet
from dataset import rPPGDataset
from heart_rate import bandpass_filter, extract_chrom_pulse, calculate_heart_rate
from evaluate   import load_ground_truth
from pipeline   import load_rgb_csv


def reconstruct_full_waveform(model, dataset, full_length, window_size=150, stride=75):
    """
    Runs model inference over windowed samples and reconstructs the full
    continuous signal via Hann-weighted overlap-add.

    A5 fix:
    - Uses Hann window weights so the weighted sum is constant across all
      frames (assuming >= 50% overlap), removing the amplitude dip at the tail.
    - The dataset already adds a final window ending exactly at N (see dataset.py),
      so every frame receives at least one window's contribution.
    - Asserts that no frame has zero total weight after reconstruction.
    """
    device = next(model.parameters()).device
    model.eval()

    full_pred  = np.zeros(full_length, dtype=np.float32)
    weight_sum = np.zeros(full_length, dtype=np.float32)

    # Hann window for weighted overlap-add (weights sum to a near-constant value
    # when overlap >= 50%, giving uniform reconstruction amplitude).
    hann = np.hanning(window_size).astype(np.float32)

    with torch.no_grad():
        for i, start in enumerate(dataset.start_indices):
            rgb_window, _ = dataset[i]             # (3, window_size)
            rgb_tensor = rgb_window.unsqueeze(0).to(device)  # (1, 3, window_size)

            pred = model(rgb_tensor).squeeze(0).cpu().numpy()  # (window_size,)

            end = start + window_size
            if end <= full_length:
                full_pred[start:end]  += pred * hann
                weight_sum[start:end] += hann

    # A5 assertion: no frame should have zero weight after reconstruction
    assert np.all(weight_sum > 0), (
        "A5 violation: some frames received zero Hann weight. "
        "Check that the dataset covers all frames."
    )

    reconstructed = full_pred / weight_sum
    return reconstructed


def evaluate_model_pipeline(
    rgb_csv_path: str = "results/subject1_rgb.csv",
    gt_path: str      = "results/gtdump.xmp",
    model_path: str   = "models/rppgnet_best.pth",
    fps: float        = 30.0,
    output_plot: str  = "results/model_vs_baseline.png"
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

    # 1. Load Data — A2: load_rgb_csv handles NaN interpolation
    rgb_signals, _ = load_rgb_csv(rgb_csv_path, fps=fps)

    gt_data     = load_ground_truth(gt_path)
    gt_ppg_raw  = gt_data["gt_ppg_signal"]

    # A3 note: for now we use linspace resampling (same assumption as before —
    # both recordings cover the same duration). A dedicated sync.py aligns by
    # timestamp cross-correlation; integrate that here once A3 is fully done.
    x_old  = np.linspace(0, 1, len(gt_ppg_raw))
    x_new  = np.linspace(0, 1, len(rgb_signals))
    gt_ppg = np.interp(x_new, x_old, gt_ppg_raw).astype(np.float32)

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = rPPGNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded trained rPPGNet model from: {model_path}")

    # 3. Model Inference & Signal Reconstruction (A5: Hann-weighted overlap-add)
    dataset          = rPPGDataset(rgb_signals, gt_ppg, window_size=150, stride=75)
    model_raw_signal = reconstruct_full_waveform(model, dataset, len(rgb_signals))
    model_pulse      = bandpass_filter(model_raw_signal, fps, lowcut=0.75, highcut=2.5)

    # 4. Classical Baseline (CHROM) Signal Extraction
    chrom_pulse = extract_chrom_pulse(rgb_signals, fps, lowcut=0.75, highcut=2.5)

    # Ground Truth PPG (zero-centred then bandpass filtered)
    gt_pulse = bandpass_filter(gt_ppg - np.mean(gt_ppg), fps, lowcut=0.75, highcut=2.5)

    # 5. Heart Rate Estimation via Hann-windowed periodogram
    # B2 fix: this is a single Hann-windowed FFT (zero-padded to 2048),
    # NOT Welch multi-segment averaging.
    model_bpm, model_freqs, model_psd, _ = calculate_heart_rate(model_pulse, fps)
    chrom_bpm, chrom_freqs, chrom_psd, _ = calculate_heart_rate(chrom_pulse, fps)
    gt_bpm,    gt_freqs,    gt_psd,    _ = calculate_heart_rate(gt_pulse,    fps)

    # 6. Metrics
    r_chrom   = pearsonr_np(chrom_pulse, gt_pulse)
    r_model   = pearsonr_np(model_pulse, gt_pulse)
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
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)

    def z_score(s):
        return (s - np.mean(s)) / (np.std(s) + 1e-8)

    gt_norm    = z_score(gt_pulse)
    chrom_norm = z_score(chrom_pulse)
    model_norm = z_score(model_pulse)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle("rPPG Performance Comparison: Classical CHROM vs. Learned rPPGNet", fontsize=14, fontweight="bold")

    plot_samples = min(len(time), int(15 * fps))
    axes[0].plot(time[:plot_samples], gt_norm[:plot_samples],    label="Ground Truth PPG",                  color="black",   alpha=0.7, linewidth=1.5)
    axes[0].plot(time[:plot_samples], chrom_norm[:plot_samples], label=f"CHROM Baseline (r={r_chrom:.2f})", color="tab:blue",alpha=0.6, linestyle="--")
    axes[0].plot(time[:plot_samples], model_norm[:plot_samples], label=f"rPPGNet Model (r={r_model:.2f})",  color="tab:red", linewidth=1.8)
    axes[0].set_title("Normalized Pulse Waveform Comparison (First 15 Seconds)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Normalized Amplitude (Z-score)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # B2 fix: label as "Hann-Windowed Periodogram", not Welch PSD
    valid_mask = (gt_freqs >= 0.75) & (gt_freqs <= 2.5)
    axes[1].plot(gt_freqs[valid_mask]    * 60, gt_psd[valid_mask]    / np.max(gt_psd[valid_mask]),    label=f"GT Peak ({gt_bpm:.1f} BPM)",    color="black",   linewidth=1.5)
    axes[1].plot(chrom_freqs[valid_mask] * 60, chrom_psd[valid_mask] / np.max(chrom_psd[valid_mask]), label=f"CHROM Peak ({chrom_bpm:.1f} BPM)", color="tab:blue",alpha=0.6, linestyle="--")
    axes[1].plot(model_freqs[valid_mask] * 60, model_psd[valid_mask] / np.max(model_psd[valid_mask]), label=f"rPPGNet Peak ({model_bpm:.1f} BPM)", color="tab:red", linewidth=1.8)
    axes[1].set_title("Hann-Windowed Periodogram & Heart Rate Peak Alignment")
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
    parser.add_argument("--rgb",    type=str, default=os.path.join(BASE_DIR, "results", "subject1_rgb.csv"))
    parser.add_argument("--gt",     type=str, default=os.path.join(BASE_DIR, "results", "gtdump.xmp"))
    parser.add_argument("--model",  type=str, default=os.path.join(BASE_DIR, "models", "rppgnet_best.pth"))
    parser.add_argument("--fps",    type=float, default=30.0)
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "results", "model_vs_baseline.png"))
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
