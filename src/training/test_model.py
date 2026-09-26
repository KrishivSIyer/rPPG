import torch
import numpy as np
from model import rPPGNet
from train import train_model

def test_forward_pass():
    print("--- Testing Forward Pass ---")
    model = rPPGNet()
    # Dummy input: Batch Size=16, Channels=3, Frames=150
    dummy_input = torch.randn(16, 3, 150)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (16, 150), "Output shape mismatch!"
    print("Forward pass successful!\n")

def test_training_loop():
    print("--- Testing Training Loop with Dummy Data ---")
    # Simulate 1000 frames of RGB and PPG data
    rgb_signals = np.random.randn(1000, 3).astype(np.float32)
    ppg_signal = np.random.randn(1000).astype(np.float32)
    
    # Run for just 2 epochs to verify no runtime errors
    train_model(rgb_signals, ppg_signal, epochs=2, batch_size=16, save_dir='test_models')
    print("Training loop successful!\n")

if __name__ == "__main__":
    test_forward_pass()
    test_training_loop()
