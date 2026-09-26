import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 × Conv1d blocks (3→16→32→64 channels)
        # BatchNorm + ELU after each block
        # Kernel size 3, same padding → preserves time length
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        # Input: (B, 3, T)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        return x  # Output: (B, 64, T)


class TemporalAttention(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        # Q, K, V projections via 1×1 Conv1d
        self.q_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.k_conv = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.v_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        B, C, T = x.size()
        
        # Projections
        q = self.q_conv(x).view(B, -1, T).permute(0, 2, 1)  # (B, T, C/8)
        k = self.k_conv(x).view(B, -1, T)                   # (B, C/8, T)
        v = self.v_conv(x).view(B, -1, T)                   # (B, C, T)
        
        # Attention map: (B, T, T) softmax scores
        energy = torch.bmm(q, k)  # (B, T, T)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to V
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, T)
        
        # Residual connection
        out = out + x
        return out


class rPPGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_cnn = SpatialCNN()
        self.temporal_attention = TemporalAttention(channels=64)
        
        # Decoder: 1×1 Conv1d (64 channels → 1 channel)
        self.decoder = nn.Conv1d(64, 1, kernel_size=1)
        
    def forward(self, x):
        """
        x: (B, 3, T) windowed RGB signals
        """
        # Feature Extraction
        features = self.spatial_cnn(x)              # (B, 64, T)
        
        # Temporal Attention to suppress noisy frames
        attended_features = self.temporal_attention(features)  # (B, 64, T)
        
        # Decode to 1D waveform
        out = self.decoder(attended_features)       # (B, 1, T)
        
        return out.squeeze(1)                       # (B, T)
