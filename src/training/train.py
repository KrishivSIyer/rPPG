import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import rPPGNet
import os

def pearson_loss(pred, target):
    """
    Computes negative Pearson correlation between predicted and target waveforms.
    Minimizing this loss maximizes the shape similarity of the waveforms.
    pred, target: (B, T)
    """
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    
    num = (pred * target).sum(dim=1)
    denom = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    
    corr = num / denom
    return -corr.mean()  # Negate to minimize

def train_model(rgb_signals, ppg_signal, epochs=50, batch_size=16, save_dir='models', pretrained_weights=None, lr=1e-3):
    """
    Trains or fine-tunes rPPGNet on the provided RGB and PPG signals.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        rgb_signals, ppg_signal, batch_size=batch_size
    )
    
    # Initialize model
    model = rPPGNet().to(device)
    
    # Load pre-trained weights if provided (for fine-tuning on noisy video)
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"Loading pre-trained weights from: {pretrained_weights} for fine-tuning...")
        model.load_state_dict(torch.load(pretrained_weights, map_location=device))
        print("Pre-trained weights loaded successfully!")
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Scheduler: ReduceLROnPlateau (patience=5) — halves LR if val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_corr = -1.0
    patience_counter = 0
    patience_limit = 10
    
    train_losses = []
    val_losses = []
    val_corrs = []
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss_epoch = 0.0
        
        for batch_rgb, batch_ppg in train_loader:
            batch_rgb, batch_ppg = batch_rgb.to(device), batch_ppg.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass -> predicted waveform (B, T)
            pred_ppg = model(batch_rgb)
            
            # Compute Pearson loss vs GT PPG window
            loss = pearson_loss(pred_ppg, batch_ppg)
            
            # Backward + optimizer step
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
                # the correlation is just -loss for pearson_loss since it returns -corr.mean()
                val_corr_epoch += (-loss.item()) * batch_rgb.size(0)
                
        val_loss_epoch /= len(val_loader.dataset)
        val_corr_epoch /= len(val_loader.dataset)
        
        val_losses.append(val_loss_epoch)
        val_corrs.append(val_corr_epoch)
        
        scheduler.step(val_loss_epoch)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | Val Corr: {val_corr_epoch:.4f}")
        
        # Checkpointing
        if val_corr_epoch > best_val_corr:
            best_val_corr = val_corr_epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'rppgnet_best.pth'))
            patience_counter = 0
            print("  --> Saved new best model")
        else:
            patience_counter += 1
            
        # Early Stopping
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    return model, train_losses, val_losses, val_corrs
