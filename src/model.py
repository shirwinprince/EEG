import torch
import torch.nn as nn

class EEGSpectrogramCNNLSTM(nn.Module):
    def __init__(self, num_classes=3, in_channels=22):
        super(EEGSpectrogramCNNLSTM, self).__init__()
        
        # 1. The Visual Feature Extractor (CNN)
        # Looks for the raw frequency shapes in the spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # This forces the output to be exactly 8x8 (Freq x Time) for the LSTM
            nn.AdaptiveMaxPool2d((8, 8)) 
        )
        
        # 2. The Timeline Processor (LSTM)
        # Sequence Length = 8 (time steps). Features per step = 64 channels * 8 freq bins = 512
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3
        )
        
        # 3. The Final Decision (Classifier)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Step 1: Pass through CNN
        # x shape starts as: (Batch, 22 channels, Freq, Time)
        c = self.cnn(x) 
        
        # c shape is now: (Batch, 64 channels, 8 Freq, 8 Time)
        batch_size, channels, freq, time = c.size()
        
        # Step 2: Reshape for the LSTM
        # LSTMs require the shape: (Batch, Sequence_Length, Features)
        # We permute to move 'Time' to the middle: (Batch, Time, Channels, Freq)
        c = c.permute(0, 3, 1, 2) 
        
        # Flatten the Channels and Freq into one continuous feature vector per time step
        c = c.reshape(batch_size, time, channels * freq) 
        # c shape is now perfectly prepared: (Batch, 8, 512)
        
        # Step 3: Read the timeline
        lstm_out, (h_n, c_n) = self.lstm(c)
        
        # We only want the very last time step's output (after it has read the whole 2 seconds)
        last_time_step = lstm_out[:, -1, :] 
        
        # Step 4: Make the final prediction
        out = self.fc1(last_time_step)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out