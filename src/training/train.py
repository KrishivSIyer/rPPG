import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import rPPGNet
import os
import json


def pearson_loss(pred, target):
    """
    Computes negative Pearson correlation between predicted and target waveforms.
    Minimizing this loss maximizes the shape similarity of the waveforms.
    pred, target: (B, T)

    C5 note: Pearson loss is invariant to scale and offset of the target.
    z-scoring the target in the Dataset does not change this loss value,
    but it does help numerical stability for evaluation plots.
    """
    pred   = pred   - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    num   = (pred * target).sum(dim=1)
    denom = pred.norm(dim=1) * target.norm(dim=1) + 1e-8

    corr = num / denom
    return -corr.mean()  # Negate to minimize


def train_model(
    rgb_signals,
    ppg_signal,
    epochs=50,
    batch_size=16,
    save_dir='models',
    run_name=None,
    pretrained_weights=None,
    lr=1e-3,
    seed=0,
):
    """
    Trains or fine-tunes rPPGNet on the provided RGB and PPG signals.

    A4 fix: checkpoints are saved to <save_dir>/<run_name>/best.pth so that
    fine-tuning runs never overwrite the baseline weights.  If the resolved
    output path equals the input weights path the function refuses to run.

    D1 fix: random seeds are set for reproducibility.
    """
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    # D1: set all seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True

    # A4: build the run directory
    if run_name is None:
        from datetime import datetime
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    out_checkpoint = os.path.join(run_dir, "best.pth")

    # A4: refuse to overwrite input weights
    if pretrained_weights is not None:
        if os.path.abspath(out_checkpoint) == os.path.abspath(pretrained_weights):
            raise ValueError(
                f"Output checkpoint path '{out_checkpoint}' is the same as the "
                f"input weights path '{pretrained_weights}'. Use a different "
                f"--run_name or --save_dir to avoid overwriting your base model."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        rgb_signals, ppg_signal, batch_size=batch_size
    )

    model = rPPGNet().to(device)

    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"Loading pre-trained weights from: {pretrained_weights} for fine-tuning...")
        model.load_state_dict(torch.load(pretrained_weights, map_location=device))
        print("Pre-trained weights loaded successfully!")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_corr = -1.0
    patience_counter = 0
    patience_limit = 10

    train_losses = []
    val_losses   = []
    val_corrs    = []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss_epoch = 0.0

        for batch_rgb, batch_ppg in train_loader:
            batch_rgb, batch_ppg = batch_rgb.to(device), batch_ppg.to(device)
            optimizer.zero_grad()
            pred_ppg = model(batch_rgb)
            loss = pearson_loss(pred_ppg, batch_ppg)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * batch_rgb.size(0)

        train_loss_epoch /= len(train_loader.dataset)
        train_losses.append(train_loss_epoch)

        # --- Validation Phase ---
        model.eval()
        val_loss_epoch = 0.0
        val_corr_epoch = 0.0

        with torch.no_grad():
            for batch_rgb, batch_ppg in val_loader:
                batch_rgb, batch_ppg = batch_rgb.to(device), batch_ppg.to(device)
                pred_ppg = model(batch_rgb)
                loss = pearson_loss(pred_ppg, batch_ppg)
                val_loss_epoch += loss.item() * batch_rgb.size(0)
                val_corr_epoch += (-loss.item()) * batch_rgb.size(0)

        val_loss_epoch /= len(val_loader.dataset)
        val_corr_epoch /= len(val_loader.dataset)

        val_losses.append(val_loss_epoch)
        val_corrs.append(val_corr_epoch)

        scheduler.step(val_loss_epoch)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | Val Corr: {val_corr_epoch:.4f}")

        if val_corr_epoch > best_val_corr:
            best_val_corr = val_corr_epoch
            torch.save(model.state_dict(), out_checkpoint)
            patience_counter = 0
            print(f"  --> Saved new best model -> {out_checkpoint}")
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # A4: write config.json next to the checkpoint
    config = {
        "run_name": run_name,
        "pretrained_weights": pretrained_weights,
        "best_val_corr": float(best_val_corr),
        "epochs_trained": len(train_losses),
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "checkpoint": out_checkpoint,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {os.path.join(run_dir, 'config.json')}")

    return model, train_losses, val_losses, val_corrs
