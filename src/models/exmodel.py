import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGSpectrogramCNN(nn.Module):
    def __init__(self, num_classes=3):
        """
        2D CNN for EEG Spectrogram classification.
        Classes: 0 (Interictal), 1 (Preictal), 2 (Ictal)
        """
        super(EEGSpectrogramCNN, self).__init__()
        
        # Input shape expected: (Batch_Size, 23 Channels, Freq_bins, Time_steps)
        self.conv1 = nn.Conv2d(in_channels=23, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Adaptive pooling ensures the output to the linear layer is always the same size (64 x 8 x 4)
        # regardless of slight variations in exact STFT frequency/time bin outputs
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 4, 128)
        self.dropout = nn.Dropout(0.5) # Prevent overfitting
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Adaptive Pooling & Flatten
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Quick architecture test
    model = EEGSpectrogramCNN()
    print("Model Architecture Initialized:")
    print(model)
    
    # Simulate a batch of 16 windows (16, 23 channels, 65 freq bins, 9 time steps)
    dummy_input = torch.randn(16, 23, 65, 9)
    output = model(dummy_input)
    print(f"\nDummy Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Batch Size x Num Classes)")