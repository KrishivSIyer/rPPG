import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class rPPGDataset(Dataset):
    def __init__(self, rgb_signals, ppg_signal, window_size=150, stride=75):
        """
        rgb_signals: (N, 3) array of mean RGB values per frame
        ppg_signal: (N,) array of ground truth PPG values
        """
        self.window_size = window_size
        self.stride = stride
        
        # Normalize the full signals before windowing
        self.rgb_signals = self.normalize_rgb(rgb_signals)
        self.ppg_signal = self.normalize_ppg(ppg_signal)
        
        self.windows = self._create_windows()
        
    def normalize_rgb(self, signals):
        # Zero mean, unit std per channel
        mean = np.mean(signals, axis=0, keepdims=True)
        std = np.std(signals, axis=0, keepdims=True)
        return (signals - mean) / (std + 1e-8)
        
    def normalize_ppg(self, signal):
        # Zero mean, unit std
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / (std + 1e-8)
        
    def _create_windows(self):
        windows = []
        n_frames = len(self.rgb_signals)
        for start in range(0, n_frames - self.window_size + 1, self.stride):
            end = start + self.window_size
            rgb_window = self.rgb_signals[start:end] # (150, 3)
            ppg_window = self.ppg_signal[start:end] # (150,)
            windows.append((rgb_window, ppg_window))
        return windows
        
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        rgb, ppg = self.windows[idx]
        # Model expects (B, C, T) -> transpose RGB from (150, 3) to (3, 150)
        rgb = np.transpose(rgb, (1, 0))
        return torch.FloatTensor(rgb), torch.FloatTensor(ppg)


def get_dataloaders(rgb_signals, ppg_signal, window_size=150, stride=75, batch_size=16, train_split=0.8):
    dataset = rPPGDataset(rgb_signals, ppg_signal, window_size, stride)
    
    # We do a sequential split rather than random split to avoid data leakage 
    # between overlapping windows in train and val sets.
    train_size = int(train_split * len(dataset))
    
    # Generate indices for sequential split
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # We can shuffle the train loader, but val loader is sequential
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
