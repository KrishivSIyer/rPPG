import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from heart_rate import extract_green_pulse, extract_chrom_pulse, calculate_heart_rate


def load_ground_truth(gt_path: str):
    """
    Parses UBFC-rPPG gtdump.xmp file.
    
    Format:
        Col 0: Timestamp (ms)
        Col 1: Ground Truth Heart Rate (BPM) from pulse oximeter
        Col 2: SpO2 (%)
        Col 3: Raw PPG pulse waveform
    """
    data = np.loadtxt(gt_path, delimiter=",")
    timestamps_ms = data[:, 0]
    gt_hr_series = data[:, 1]
    spo2_series = data[:, 2]
    gt_ppg_signal = data[:, 3]

    # Calculate GT sampling rate
    time_sec = (timestamps_ms - timestamps_ms[0]) / 1000.0
    gt_fps = len(time_sec) / (time_sec[-1] - time_sec[0])

    return {
        "time_sec": time_sec,
        "gt_hr_series": gt_hr_series,
        "spo2_series": spo2_series,
        "gt_ppg_signal": gt_ppg_signal,
        "gt_fps": gt_fps,
        "mean_gt_hr": np.mean(gt_hr_series)
    }


def compute_sliding_window_hr(rgb_signals: np.ndarray, fps: float, window_sec: float = 10.0, step_sec: float = 1.0):
    """
    Computes continuous Heart Rate over sliding temporal windows.
    """
    window_samples = int(window_sec * fps)
    step_samples = int(step_sec * fps)
    total_samples = len(rgb_signals)

    time_centers = []
    green_hr_list = []
    chrom_hr_list = []

    for start in range(0, total_samples - window_samples + 1, step_samples):
        end = start + window_samples
        sub_rgb = rgb_signals[start:end]
        center_time = (start + window_samples / 2.0) / fps

        g_pulse = extract_green_pulse(sub_rgb, fps)
        c_pulse = extract_chrom_pulse(sub_rgb, fps)

        try:
            g_bpm, _, _, _ = calculate_heart_rate(g_pulse, fps)
            c_bpm, _, _, _ = calculate_heart_rate(c_pulse, fps)
        except Exception:
            continue

        time_centers.append(center_time)
        green_hr_list.append(g_bpm)
        chrom_hr_list.append(c_bpm)

    return np.array(time_centers), np.array(green_hr_list), np.array(chrom_hr_list)


def evaluate_ground_truth(rgb_csv_path: str, gt_path: str, video_fps: float = 30.0, output_plot: str = "results/ground_truth_comparison.png"):
    """
    Evaluates rPPG heart rate estimation against ground truth recording.
    """
    if not os.path.exists(rgb_csv_path):
        raise FileNotFoundError(f"RGB CSV file not found: {rgb_csv_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground Truth file not found: {gt_path}")

    # Load RGB signals & GT data
    df = pd.read_csv(rgb_csv_path)
    rgb_signals = df[["R", "G", "B"]].values
    gt_data = load_ground_truth(gt_path)

    # 1. Global Heart Rate Estimation (Entire Recording)
    g_pulse_full = extract_green_pulse(rgb_signals, video_fps)
    c_pulse_full = extract_chrom_pulse(rgb_signals, video_fps)

    g_bpm_global, g_freqs, g_psd, _ = calculate_heart_rate(g_pulse_full, video_fps)
    c_bpm_global, c_freqs, c_psd, _ = calculate_heart_rate(c_pulse_full, video_fps)

    # Compute GT PPG PSD Peak HR
    gt_pulse_norm = gt_data["gt_ppg_signal"] - np.mean(gt_data["gt_ppg_signal"])
    gt_freqs, gt_psd = welch(gt_pulse_norm, fs=gt_data["gt_fps"], nperseg=min(len(gt_pulse_norm), int(gt_data["gt_fps"]*10)), nfft=2048)
    valid_mask = (gt_freqs >= 0.75) & (gt_freqs <= 2.5)
    gt_peak_freq = gt_freqs[valid_mask][np.argmax(gt_psd[valid_mask])]
    gt_psd_bpm = gt_peak_freq * 60.0

    mean_gt_bpm = gt_data["mean_gt_hr"]

    # 2. Continuous Sliding Window HR Evaluation
    time_centers, g_hr_win, c_hr_win = compute_sliding_window_hr(rgb_signals, video_fps, window_sec=10.0, step_sec=1.0)
    gt_hr_interpolated = np.interp(time_centers, gt_data["time_sec"], gt_data["gt_hr_series"])

    # Compute Error Metrics
    g_mae = np.mean(np.abs(g_hr_win - gt_hr_interpolated))
    c_mae = np.mean(np.abs(c_hr_win - gt_hr_interpolated))

    g_rmse = np.sqrt(np.mean((g_hr_win - gt_hr_interpolated) ** 2))
    c_rmse = np.sqrt(np.mean((c_hr_win - gt_hr_interpolated) ** 2))

    g_mape = np.mean(np.abs((g_hr_win - gt_hr_interpolated) / gt_hr_interpolated)) * 100.0
    c_mape = np.mean(np.abs((c_hr_win - gt_hr_interpolated) / gt_hr_interpolated)) * 100.0

    g_corr, _ = pearsonr(g_hr_win, gt_hr_interpolated) if len(g_hr_win) > 1 else (0.0, 0.0)
    c_corr, _ = pearsonr(c_hr_win, gt_hr_interpolated) if len(c_hr_win) > 1 else (0.0, 0.0)

    # 3. Plotting Comprehensive Ground Truth Comparison
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    fig.suptitle("rPPG vs. Ground Truth Pulse Oximeter Evaluation", fontsize=15, fontweight="bold")

    # Plot 1: Waveforms (Normalized comparison over first 10s)
    crop_t = 10.0
    video_time = np.arange(len(rgb_signals)) / video_fps
    idx_v = video_time <= crop_t
    idx_gt = gt_data["time_sec"] <= crop_t

    axes[0].plot(gt_data["time_sec"][idx_gt], (gt_data["gt_ppg_signal"][idx_gt] - np.mean(gt_data["gt_ppg_signal"])) / np.std(gt_data["gt_ppg_signal"]), label="Ground Truth PPG (Pulse Oximeter)", color="black", linewidth=1.5)
    axes[0].plot(video_time[idx_v], c_pulse_full[idx_v] / np.std(c_pulse_full), label="Extracted rPPG (CHROM Method)", color="purple", linestyle="--", alpha=0.85)
    axes[0].plot(video_time[idx_v], g_pulse_full[idx_v] / np.std(g_pulse_full), label="Extracted rPPG (Green Channel)", color="green", linestyle=":", alpha=0.85)
    axes[0].set_title("1. Normalized Pulse Waveforms (First 10 Seconds)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Normalized Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Continuous Heart Rate over Time
    axes[1].plot(gt_data["time_sec"], gt_data["gt_hr_series"], label=f"Ground Truth Oximeter HR (Mean: {mean_gt_bpm:.1f} BPM)", color="red", linewidth=2.0)
    axes[1].plot(time_centers, c_hr_win, label=f"CHROM Sliding HR (MAE: {c_mae:.2f} BPM)", color="purple", linestyle="--")
    axes[1].plot(time_centers, g_hr_win, label=f"Green Channel Sliding HR (MAE: {g_mae:.2f} BPM)", color="green", linestyle=":")
    axes[1].set_title("2. Instantaneous Heart Rate Tracking Over Time (10s Window)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Heart Rate (BPM)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Plot 3: Power Spectral Density Comparison
    axes[2].plot(gt_freqs * 60.0, gt_psd / np.max(gt_psd[valid_mask]), label=f"GT PPG Spectral Peak ({gt_psd_bpm:.1f} BPM)", color="black", linewidth=1.8)
    axes[2].plot(g_freqs * 60.0, g_psd / np.max(g_psd[(g_freqs>=0.75)&(g_freqs<=2.5)]), label=f"Green Channel Peak ({g_bpm_global:.1f} BPM)", color="green", linestyle=":")
    axes[2].plot(c_freqs * 60.0, c_psd / np.max(c_psd[(c_freqs>=0.75)&(c_freqs<=2.5)]), label=f"CHROM Peak ({c_bpm_global:.1f} BPM)", color="purple", linestyle="--")
    axes[2].set_xlim(40, 150)
    axes[2].set_title("3. Frequency Spectra Comparison (FFT / Welch PSD)")
    axes[2].set_xlabel("Heart Rate (BPM)")
    axes[2].set_ylabel("Normalized Spectral Power")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print("\n========================================================")
    print("      rPPG vs GROUND TRUTH EVALUATION REPORT           ")
    print("========================================================")
    print(f"Ground Truth File         : {gt_path}")
    print(f"GT Sampling Rate          : {gt_data['gt_fps']:.2f} Hz")
    print(f"GT Mean Heart Rate        : {mean_gt_bpm:.2f} BPM")
    print(f"GT Spectral Peak HR       : {gt_psd_bpm:.2f} BPM")
    print("--------------------------------------------------------")
    print("GLOBAL (ENTIRE RECORDING) ESTIMATION:")
    print(f"  Green Channel Global HR : {g_bpm_global:.2f} BPM (Abs Error: {abs(g_bpm_global - mean_gt_bpm):.2f} BPM)")
    print(f"  CHROM Baseline Global HR: {c_bpm_global:.2f} BPM (Abs Error: {abs(c_bpm_global - mean_gt_bpm):.2f} BPM)")
    print("--------------------------------------------------------")
    print("CONTINUOUS SLIDING WINDOW METRICS:")
    print(f"  Green Channel Method   -> MAE: {g_mae:.2f} BPM | RMSE: {g_rmse:.2f} BPM | MAPE: {g_mape:.2f}% | Pearson r: {g_corr:.3f}")
    print(f"  CHROM Baseline Method  -> MAE: {c_mae:.2f} BPM | RMSE: {c_rmse:.2f} BPM | MAPE: {c_mape:.2f}% | Pearson r: {c_corr:.3f}")
    print("--------------------------------------------------------")
    print(f"Saved comparison plot to : {output_plot}")
    print("========================================================\n")

    return {
        "gt_mean_bpm": mean_gt_bpm,
        "green_global_bpm": g_bpm_global,
        "chrom_global_bpm": c_bpm_global,
        "green_mae": g_mae,
        "chrom_mae": c_mae,
        "green_rmse": g_rmse,
        "chrom_rmse": c_rmse,
        "green_corr": g_corr,
        "chrom_corr": c_corr,
        "plot_path": output_plot
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate rPPG heart rate against Ground Truth gtdump.xmp.")
    parser.add_argument("--csv", type=str, default="results/subject1_rgb.csv", help="Path to extracted RGB CSV file.")
    parser.add_argument("--gt", type=str, default="data/ground truth/gtdump.xmp", help="Path to ground truth gtdump.xmp file.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS.")
    parser.add_argument("--output", type=str, default="results/ground_truth_comparison.png", help="Path for output plot.")
    args = parser.parse_args()

    evaluate_ground_truth(args.csv, args.gt, video_fps=args.fps, output_plot=args.output)


if __name__ == "__main__":
    main()
