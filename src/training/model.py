import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCNN(nn.Module):
    """
    Three stacked 1D Conv blocks operating on the temporal dimension.
    Input channels: 3 (R, G, B per frame). Output channels: 64.
    Kernel size 3 along time; colour is the input channel dimension (3→16→32→64).

    C1 fix: added dilation (1, 2, 4) so the receptive field grows to
    1 + 2*(1-1) + 2*(2-1) + 2*(4-1) = 1+0+2+6 = 9 → with 3 layers:
    RF = 1 + (3-1)*1 + (3-1)*2 + (3-1)*4 = 1+2+4+8 = 15 frames (~0.5 s at 30 fps).
    This is broad enough to span half a cardiac cycle at resting rate.

    B4 fix: kernel is size 3 along the TIME axis; colour channels are the
    in_channels / out_channels dimension (3 → 16 → 32 → 64).
    """
    def __init__(self):
        super().__init__()
        # Dilation 1, 2, 4 → receptive field = 15 frames ≈ 0.5 s at 30 fps
        # C1: parameter count stays within 2× the original (≈ same, dilation is free)
        self.conv1 = nn.Conv1d(3,  16, kernel_size=3, padding=1, dilation=1)
        self.bn1   = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=2, dilation=2)
        self.bn2   = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=4)
        self.bn3   = nn.BatchNorm1d(64)

    def forward(self, x):
        # Input:  (B, 3, T)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        return x  # Output: (B, 64, T)


class TemporalAttention(nn.Module):
    """
    Self-attention over the time axis using Q/K/V projections (1×1 Conv1d).

    B3 fix: This is a T×T self-attention map, NOT softmax frame-pooling weights.
    Each output frame is a weighted mix of ALL other frames' values, where the
    weights are dot-product similarities between query and key projections.
    There is no positional encoding — time order is provided only by the
    upstream dilated convolutions.

    C2 fix: a learned positional embedding of shape (1, 64, window_size) is
    added to the features before computing attention so the module is aware of
    each frame's position in the window.
    """
    def __init__(self, channels: int = 64, window_size: int = 150):
        super().__init__()
        # Q, K, V projections via 1×1 Conv1d (no spatial mixing, only channel mix)
        self.q_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.k_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.v_conv = nn.Conv1d(channels, channels,      kernel_size=1)

        # C2: learned positional embedding — one vector per time step
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, window_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.window_size = window_size

    def forward(self, x):
        B, C, T = x.size()

        # C2: add positional embedding (handle variable-length clips gracefully)
        if T == self.window_size:
            x = x + self.pos_embed
        else:
            x = x + self.pos_embed[:, :, :T]

        # Q/K/V projections
        q = self.q_conv(x).permute(0, 2, 1)  # (B, T, C/8)
        k = self.k_conv(x)                    # (B, C/8, T)
        v = self.v_conv(x)                    # (B, C,   T)

        # Attention map: (B, T, T) — softmax over key dimension
        scale  = (C // 8) ** -0.5
        energy = torch.bmm(q, k) * scale      # (B, T, T)
        attention = F.softmax(energy, dim=-1)

        # Apply attention to V; residual connection for stable training
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, T)
        out = out + x                                     # residual
        return out


class rPPGNet(nn.Module):
    def __init__(self, window_size: int = 150):
        super().__init__()
        self.spatial_cnn        = SpatialCNN()
        self.temporal_attention = TemporalAttention(channels=64, window_size=window_size)

        # Decoder: 1×1 Conv1d (64 channels → 1 channel)
        self.decoder = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B, 3, T) windowed RGB signals
        """
        features          = self.spatial_cnn(x)               # (B, 64, T)
        # B3: attention reweights frames by content (T×T self-attention)
        attended_features = self.temporal_attention(features)  # (B, 64, T)
        out               = self.decoder(attended_features)    # (B, 1, T)
        return out.squeeze(1)                                   # (B, T)
