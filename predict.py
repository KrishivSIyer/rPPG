import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Add src to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
TRAINING_DIR = os.path.join(SRC_DIR, "training")
for path in [BASE_DIR, SRC_DIR, TRAINING_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from pipeline import FaceLandmarkExtractor
from heart_rate import extract_chrom_pulse, calculate_heart_rate, bandpass_filter
from model import rPPGNet
from dataset import rPPGDataset
from evaluate_model import reconstruct_full_waveform


def predict_heart_rate(video_path: str, output_plot: str = "results/prediction_output.png", model_path: str = "models/rppgnet_best.pth", manual_fps: float = None):
    """
    End-to-end inference pipeline: Takes a video, extracts facial RGB signals,
    and estimates heart rate using both classical CHROM and fine-tuned rPPGNet.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found at: {video_path}")

    full_model_path = os.path.join(BASE_DIR, model_path) if not os.path.isabs(model_path) else model_path
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Trained rPPGNet model weights not found at: {full_model_path}")

    print("==================================================")
    print("        rPPG Heart Rate Estimation Pipeline       ")
    print("==================================================")
    print(f"Input Video : {video_path}")

    # 1. Extract RGB signals
    extractor = FaceLandmarkExtractor()
    rgb_signals, detected_fps = extractor.extract_video_rgb(video_path)
    extractor.close()

    fps = manual_fps if manual_fps is not None else detected_fps
    print(f"Extracted {len(rgb_signals)} frames at {fps:.2f} FPS")

    # 2. Classical CHROM Pulse & Heart Rate Estimation
    chrom_raw = extract_chrom_pulse(rgb_signals, fps)
    chrom_bpm, chrom_freqs, chrom_psd, _ = calculate_heart_rate(chrom_raw, fps)

    # 3. Learned rPPGNet Model Inference & Heart Rate Estimation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = rPPGNet().to(device)
    model.load_state_dict(torch.load(full_model_path, map_location=device))
    model.eval()

    # Create dummy GT array for dataset wrapper
    dummy_gt = np.zeros(len(rgb_signals), dtype=np.float32)
    dataset = rPPGDataset(rgb_signals, dummy_gt, window_size=150, stride=75)

    model_raw = reconstruct_full_waveform(model, dataset, len(rgb_signals))
    model_pulse = bandpass_filter(model_raw, fps, lowcut=0.75, highcut=2.5)
    model_bpm, model_freqs, model_psd, _ = calculate_heart_rate(model_pulse, fps)

    print("\n---------------- PREDICTION SUMMARY ----------------")
    print(f"{'Method':<25} | {'Estimated Heart Rate (BPM)':<25}")
    print("-" * 55)
    print(f"{'CHROM (Classical Baseline)':<25} | {chrom_bpm:<25.2f}")
    print(f"{'rPPGNet (Fine-Tuned Model)':<25} | {model_bpm:<25.2f}")
    print("----------------------------------------------------\n")

    # 4. Generate Visualization Plot
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
    time = np.arange(len(rgb_signals)) / fps

    def z_score(s):
        return (s - np.mean(s)) / (np.std(s) + 1e-8)

    chrom_norm = z_score(chrom_raw)
    model_norm = z_score(model_pulse)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Remote Photoplethysmography (rPPG) Pulse & Heart Rate Prediction\nFile: {os.path.basename(video_path)}", fontsize=13, fontweight="bold")

    # Plot 1: Pulse Waveform Comparison (First 15 seconds)
    plot_samples = min(len(time), int(15 * fps))
    axes[0].plot(time[:plot_samples], chrom_norm[:plot_samples], label=f"CHROM Baseline ({chrom_bpm:.1f} BPM)", color="tab:blue", linestyle="--", alpha=0.8)
    axes[0].plot(time[:plot_samples], model_norm[:plot_samples], label=f"rPPGNet Learned ({model_bpm:.1f} BPM)", color="tab:red", linewidth=1.8)
    axes[0].set_title("Predicted Pulse Waveforms (First 15 Seconds)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Normalized Amplitude (Z-score)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Power Spectral Density (PSD)
    valid_mask = (chrom_freqs >= 0.75) & (chrom_freqs <= 2.5)
    axes[1].plot(chrom_freqs[valid_mask] * 60, chrom_psd[valid_mask] / np.max(chrom_psd[valid_mask]), label=f"CHROM Peak ({chrom_bpm:.1f} BPM)", color="tab:blue", linestyle="--")
    axes[1].plot(model_freqs[valid_mask] * 60, model_psd[valid_mask] / np.max(model_psd[valid_mask]), label=f"rPPGNet Peak ({model_bpm:.1f} BPM)", color="tab:red", linewidth=1.8)
    axes[1].set_title("Power Spectral Density (PSD) & Dominant Heart Rate Peak")
    axes[1].set_xlabel("Heart Rate (BPM)")
    axes[1].set_ylabel("Normalized Power Density")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"Saved prediction comparison plot to: {output_plot}\n")


def main():
    parser = argparse.ArgumentParser(description="Estimate Heart Rate from a video using fine-tuned rPPGNet & CHROM.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--output", type=str, default="results/prediction_output.png", help="Path to output plot image.")
    parser.add_argument("--model", type=str, default="models/rppgnet_best.pth", help="Path to trained model weights.")
    parser.add_argument("--fps", type=float, default=None, help="Manual video FPS override.")
    args = parser.parse_args()

    predict_heart_rate(args.video, output_plot=args.output, model_path=args.model, manual_fps=args.fps)


if __name__ == "__main__":
    main()
