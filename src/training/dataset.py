import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class rPPGDataset(Dataset):
    def __init__(self, rgb_signals, ppg_signal, window_size=150, stride=75):
        """
        rgb_signals: (N, 3) array of mean RGB values per frame
        ppg_signal:  (N,)  array of ground truth PPG values

        C5 note: PPG is z-scored for numerical stability in evaluation plots.
        Pearson loss is invariant to scale/offset, so this normalisation does
        NOT change the loss value during training.
        """
        self.window_size = window_size
        self.stride = stride

        self.rgb_signals = self.normalize_rgb(rgb_signals)
        self.ppg_signal  = self.normalize_ppg(ppg_signal)

        self.windows, self.start_indices = self._create_windows()

    def normalize_rgb(self, signals):
        # Zero mean, unit std per channel
        mean = np.mean(signals, axis=0, keepdims=True)
        std  = np.std(signals,  axis=0, keepdims=True)
        return (signals - mean) / (std + 1e-8)

    def normalize_ppg(self, signal):
        # Zero mean, unit std.
        # C5: This does not affect Pearson loss (scale/offset invariant),
        #     but keeps values in a numerically stable range for plot displays.
        mean = np.mean(signal)
        std  = np.std(signal)
        return (signal - mean) / (std + 1e-8)

    def _create_windows(self):
        """
        A5 fix: ensures a final window ending exactly at N is always included
        so the tail frames are never left at zero weight during overlap-add.
        Window start indices are stored for use in reconstruct_full_waveform.
        """
        windows = []
        starts  = []
        n_frames = len(self.rgb_signals)

        start = 0
        while start + self.window_size <= n_frames:
            end = start + self.window_size
            rgb_window = self.rgb_signals[start:end]  # (window_size, 3)
            ppg_window = self.ppg_signal[start:end]   # (window_size,)
            windows.append((rgb_window, ppg_window))
            starts.append(start)
            start += self.stride

        # A5: add final window ending at N if the last regular window does not
        if n_frames > self.window_size:
            last_start = n_frames - self.window_size
            if not starts or starts[-1] != last_start:
                rgb_window = self.rgb_signals[last_start:n_frames]
                ppg_window = self.ppg_signal[last_start:n_frames]
                windows.append((rgb_window, ppg_window))
                starts.append(last_start)

        return windows, starts

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        rgb, ppg = self.windows[idx]
        # Model expects (B, C, T) — transpose RGB from (window_size, 3) to (3, window_size)
        rgb = np.transpose(rgb, (1, 0))
        return torch.FloatTensor(rgb), torch.FloatTensor(ppg)


def get_dataloaders(rgb_signals, ppg_signal, window_size=150, stride=75, batch_size=16, train_split=0.8):
    dataset = rPPGDataset(rgb_signals, ppg_signal, window_size, stride)

    # Sequential split avoids data leakage between overlapping windows in train/val.
    train_size = int(train_split * len(dataset))
    indices    = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
