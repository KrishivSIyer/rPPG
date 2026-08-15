import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt


def bandpass_filter(signal: np.ndarray, fps: float, lowcut: float = 0.75, highcut: float = 2.5, order: int = 3) -> np.ndarray:
    """
    Applies a 3rd order zero-phase Butterworth bandpass filter to a 1D signal.
    
    Args:
        signal: 1D temporal signal array.
        fps: Sampling rate / frames per second.
        lowcut: Lower cutoff frequency in Hz (0.75 Hz = 45 BPM).
        highcut: Upper cutoff frequency in Hz (2.5 Hz = 150 BPM).
        order: Order of the Butterworth filter.
        
    Returns:
        Filtered 1D signal.
    """
    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def extract_green_pulse(rgb_signals: np.ndarray, fps: float, lowcut: float = 0.75, highcut: float = 2.5) -> np.ndarray:
    """
    Extracts pulse signal using the raw Green channel normalization + bandpass filtering.
    
    Args:
        rgb_signals: (N, 3) array of mean [R, G, B] values over time.
        fps: Sampling rate (FPS).
        
    Returns:
        Filtered 1D green pulse signal.
    """
    g_channel = rgb_signals[:, 1]
    g_mean = np.mean(g_channel)
    g_norm = (g_channel / g_mean) - 1.0
    g_filtered = bandpass_filter(g_norm, fps, lowcut=lowcut, highcut=highcut)
    return g_filtered


def extract_chrom_pulse(rgb_signals: np.ndarray, fps: float, lowcut: float = 0.75, highcut: float = 2.5) -> np.ndarray:
    """
    Extracts pulse signal using the CHROM (Chrominance-based) method.
    
    Args:
        rgb_signals: (N, 3) array of mean [R, G, B] values over time.
        fps: Sampling rate (FPS).
        
    Returns:
        Filtered 1D CHROM pulse signal.
    """
    r_norm = rgb_signals[:, 0] / np.mean(rgb_signals[:, 0])
    g_norm = rgb_signals[:, 1] / np.mean(rgb_signals[:, 1])
    b_norm = rgb_signals[:, 2] / np.mean(rgb_signals[:, 2])

    x = 3.0 * r_norm - 2.0 * g_norm
    y = 1.5 * r_norm + g_norm - 1.5 * b_norm

    x_filt = bandpass_filter(x, fps, lowcut=lowcut, highcut=highcut)
    y_filt = bandpass_filter(y, fps, lowcut=lowcut, highcut=highcut)

    alpha = np.std(x_filt) / (np.std(y_filt) + 1e-8)
    s_chrom = x_filt - alpha * y_filt
    return s_chrom


def calculate_heart_rate(pulse_signal: np.ndarray, fps: float, lowcut: float = 0.75, highcut: float = 2.5):
    """
    Computes Welch Power Spectral Density (PSD) and estimates Heart Rate (BPM) from peak frequency.
    
    Args:
        pulse_signal: Filtered 1D pulse signal.
        fps: Sampling rate (FPS).
        lowcut: Minimum valid heart rate frequency in Hz (0.75 Hz = 45 BPM).
        highcut: Maximum valid heart rate frequency in Hz (2.5 Hz = 150 BPM).
        
    Returns:
        bpm: Estimated Heart Rate in Beats Per Minute.
        freqs: Frequency axis array (Hz).
        psd: Power Spectral Density array.
        peak_freq: Dominant frequency (Hz).
    """
    nperseg = min(len(pulse_signal), int(fps * 10))  # 10 second window or signal length
    freqs, psd = welch(pulse_signal, fs=fps, nperseg=nperseg, nfft=2048)

    # Filter frequencies to valid physiological heart rate range [0.75 Hz, 2.5 Hz]
    valid_mask = (freqs >= lowcut) & (freqs <= highcut)
    valid_freqs = freqs[valid_mask]
    valid_psd = psd[valid_mask]

    if len(valid_psd) == 0:
        raise ValueError("No frequencies found in the target heart rate range.")

    peak_idx = np.argmax(valid_psd)
    peak_freq = valid_freqs[peak_idx]
    bpm = peak_freq * 60.0

    return bpm, freqs, psd, peak_freq


def plot_heart_rate_analysis(rgb_signals: np.ndarray, fps: float, save_path: str = "results/heart_rate_analysis.png"):
    """
    Generates a visual summary plot comparing Green Channel and CHROM methods,
    showing raw RGB signals, filtered pulse waveforms, and Welch PSD spectra with peak markers.
    """
    time = np.arange(len(rgb_signals)) / fps

    # Extract signals
    green_pulse = extract_green_pulse(rgb_signals, fps)
    chrom_pulse = extract_chrom_pulse(rgb_signals, fps)

    # Compute HR
    green_bpm, g_freqs, g_psd, g_peak = calculate_heart_rate(green_pulse, fps)
    chrom_bpm, c_freqs, c_psd, c_peak = calculate_heart_rate(chrom_pulse, fps)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("rPPG Heart Rate Estimation (Step 4)", fontsize=14, fontweight="bold")

    # Plot 1: Raw RGB Mean Signals
    axes[0].plot(time, rgb_signals[:, 0], color="red", label="Red Channel", alpha=0.8)
    axes[0].plot(time, rgb_signals[:, 1], color="green", label="Green Channel", alpha=0.8)
    axes[0].plot(time, rgb_signals[:, 2], color="blue", label="Blue Channel", alpha=0.8)
    axes[0].set_title("1. Raw RGB Temporal Signals (Step 3 Output)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Mean Pixel Value")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Filtered Pulse Signals
    axes[1].plot(time, green_pulse, color="darkgreen", label=f"Green Channel Pulse (Est. {green_bpm:.1f} BPM)", alpha=0.8)
    axes[1].plot(time, chrom_pulse, color="purple", label=f"CHROM Pulse (Est. {chrom_bpm:.1f} BPM)", alpha=0.8)
    axes[1].set_title("2. Bandpass Filtered rPPG Pulse Waveforms (0.75 - 2.5 Hz)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Plot 3: Power Spectral Density (FFT / Welch)
    g_bpm_axis = g_freqs * 60.0
    axes[2].plot(g_bpm_axis, g_psd, color="darkgreen", label=f"Green PSD Peak: {green_bpm:.1f} BPM", alpha=0.8)
    axes[2].plot(g_bpm_axis, c_psd, color="purple", label=f"CHROM PSD Peak: {chrom_bpm:.1f} BPM", alpha=0.8)
    axes[2].axvline(green_bpm, color="green", linestyle=":", alpha=0.7)
    axes[2].axvline(chrom_bpm, color="purple", linestyle=":", alpha=0.7)
    axes[2].set_xlim(40, 160)
    axes[2].set_title("3. Welch Power Spectral Density (FFT Frequency Spectrum)")
    axes[2].set_xlabel("Heart Rate (Beats Per Minute - BPM)")
    axes[2].set_ylabel("Power")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n=========================================")
    print(f"       rPPG HEART RATE ESTIMATION       ")
    print(f"=========================================")
    print(f"Total Frames Analyzed : {len(rgb_signals)}")
    print(f"Video Sampling Rate   : {fps:.2f} FPS")
    print(f"Signal Duration       : {len(rgb_signals)/fps:.2f} seconds")
    print(f"-----------------------------------------")
    print(f"Green Channel HR Est. : {green_bpm:.2f} BPM ({g_peak:.3f} Hz)")
    print(f"CHROM Baseline HR Est.: {chrom_bpm:.2f} BPM ({c_peak:.3f} Hz)")
    print(f"-----------------------------------------")
    print(f"Saved analysis plot to: {save_path}")
    print(f"=========================================\n")

    return {
        "green_bpm": green_bpm,
        "chrom_bpm": chrom_bpm,
        "plot_path": save_path
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate Heart Rate from Extracted rPPG RGB signals.")
    parser.add_argument("--csv", type=str, default="results/subject1_rgb.csv", help="Path to input RGB CSV file.")
    parser.add_argument("--fps", type=float, default=30.0, help="Sampling rate / video FPS.")
    parser.add_argument("--output", type=str, default="results/heart_rate_analysis.png", help="Path to output plot image.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Input CSV file not found at: {args.csv}")

    df = pd.read_csv(args.csv)
    rgb_signals = df[["R", "G", "B"]].values
    plot_heart_rate_analysis(rgb_signals, fps=args.fps, save_path=args.output)


if __name__ == "__main__":
    main()
