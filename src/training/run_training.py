import os
import argparse
import numpy as np
import pandas as pd
from train import train_model

def load_ground_truth(gt_path: str):
    """
    Parses UBFC-rPPG ground truth file (.txt or .xmp).
    """
    if gt_path.endswith(".xmp"):
        data = np.loadtxt(gt_path, delimiter=",")
        return data[:, 3]
    
    # Try space-separated loadtxt for UBFC ground_truth.txt
    data = np.loadtxt(gt_path)
    if data.ndim == 2:
        # Row 0: PPG pulse waveform
        return data[0, :]
    return data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune rPPGNet on extracted RGB signals and GT PPG.")
    parser.add_argument("--rgb", type=str, default=os.path.join(BASE_DIR, "results", "subject1_rgb.csv"), help="Path to extracted RGB CSV file.")
    parser.add_argument("--gt", type=str, default=os.path.join(BASE_DIR, "results", "gtdump.xmp"), help="Path to ground truth PPG file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--save_dir", type=str, default=os.path.join(BASE_DIR, "models"), help="Directory to save the best model.")
    parser.add_argument("--weights", type=str, default=None, help="Path to pre-trained model weights for fine-tuning.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (use smaller LR e.g. 1e-4 for fine-tuning).")
    args = parser.parse_args()

    print(f"Loading RGB signals from {args.rgb}...")
    if not os.path.exists(args.rgb):
        print(f"Error: Could not find RGB file at {args.rgb}")
        return
        
    df = pd.read_csv(args.rgb)
    rgb_signals = df[['R', 'G', 'B']].values.astype(np.float32)

    print(f"Loading Ground Truth PPG from {args.gt}...")
    if not os.path.exists(args.gt):
        print(f"Error: Could not find ground truth file at {args.gt}")
        print("Please ensure the UBFC-rPPG dataset is downloaded and the path is correct.")
        return
        
    ppg_signal = load_ground_truth(args.gt)

    # Ensure lengths match or resample
    # In a full pipeline, the video fps and the ground truth fps might differ slightly,
    # so we should interpolate. For simplicity here, if lengths don't match, we interpolate to match RGB length.
    if len(rgb_signals) != len(ppg_signal):
        print(f"Warning: Length mismatch (RGB: {len(rgb_signals)}, PPG: {len(ppg_signal)}). Interpolating PPG...")
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
        pretrained_weights=args.weights,
        lr=args.lr
    )

if __name__ == "__main__":
    main()
