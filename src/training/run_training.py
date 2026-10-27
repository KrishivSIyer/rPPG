import os
import sys
import argparse
import numpy as np
import pandas as pd
from train import train_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure pipeline's load_rgb_csv helper is available
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from pipeline import load_rgb_csv


def load_ground_truth(gt_path: str):
    """
    Parses UBFC-rPPG ground truth file (.txt or .xmp).
    """
    if gt_path.endswith(".xmp"):
        data = np.loadtxt(gt_path, delimiter=",")
        return data[:, 3]

    # Space-separated UBFC ground_truth.txt format
    data = np.loadtxt(gt_path)
    if data.ndim == 2:
        return data[0, :]
    return data


def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune rPPGNet on extracted RGB signals and GT PPG.")
    parser.add_argument("--rgb",        type=str, default=os.path.join(BASE_DIR, "results", "subject1_rgb.csv"),
                        help="Path to extracted RGB CSV file.")
    parser.add_argument("--gt",         type=str, default=os.path.join(BASE_DIR, "results", "gtdump.xmp"),
                        help="Path to ground truth PPG file.")
    parser.add_argument("--epochs",     type=int,   default=50,  help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int,   default=16,  help="Batch size for training.")
    parser.add_argument("--save_dir",   type=str,   default=os.path.join(BASE_DIR, "models"),
                        help="Parent directory for checkpoints. Actual checkpoint goes to <save_dir>/<run_name>/best.pth (A4).")
    parser.add_argument("--run_name",   type=str,   default=None,
                        help="A4: sub-directory name for this run (default: timestamp). "
                             "Use e.g. --run_name subject49_ft so fine-tuning never overwrites models/rppgnet_best.pth.")
    parser.add_argument("--weights",    type=str,   default=None,
                        help="Path to pre-trained model weights for fine-tuning.")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="Learning rate (use 1e-4 for fine-tuning).")
    parser.add_argument("--seed",       type=int,   default=0,
                        help="D1: random seed for reproducibility.")
    args = parser.parse_args()

    print(f"Loading RGB signals from {args.rgb}...")
    if not os.path.exists(args.rgb):
        print(f"Error: Could not find RGB file at {args.rgb}")
        return

    # A2: use load_rgb_csv so NaN gaps are interpolated before training
    rgb_signals, _ = load_rgb_csv(args.rgb)

    print(f"Loading Ground Truth PPG from {args.gt}...")
    if not os.path.exists(args.gt):
        print(f"Error: Could not find ground truth file at {args.gt}")
        return

    ppg_signal = load_ground_truth(args.gt)

    # A3 note: ideally timestamps are used here; for now we still use linspace
    # resampling as a placeholder (the dedicated sync.py module covers A3 fully).
    if len(rgb_signals) != len(ppg_signal):
        print(f"Warning: Length mismatch (RGB: {len(rgb_signals)}, PPG: {len(ppg_signal)}). "
              "Interpolating PPG onto RGB frame count (assumes same duration).")
        x_old = np.linspace(0, 1, len(ppg_signal))
        x_new = np.linspace(0, 1, len(rgb_signals))
        ppg_signal = np.interp(x_new, x_old, ppg_signal).astype(np.float32)

    print(f"Starting training on {len(rgb_signals)} frames...")
    train_model(
        rgb_signals,
        ppg_signal,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        run_name=args.run_name,
        pretrained_weights=args.weights,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
