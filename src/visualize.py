import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import your directory configuration
from config import WINDOWS_DIR

# ==========================================
# 1. DEPENDENCIES (Reused from evaluate.py)
# ==========================================
class EEGDataset(Dataset):
    def __init__(self, npz_files):
        total_windows = sum(len(np.load(f)['y']) for f in npz_files)
        self.X = np.empty((total_windows, 23, 65, 9), dtype=np.float32)
        self.y = np.empty(total_windows, dtype=np.int64)
        
        current_idx = 0
        for f in npz_files:
            data = np.load(f)
            n = len(data['y'])
            self.X[current_idx : current_idx + n] = data['X']
            self.y[current_idx : current_idx + n] = data['y']
            current_idx += n
            
        mean = np.mean(self.X, axis=(-2, -1), keepdims=True, dtype=np.float32)
        std = np.std(self.X, axis=(-2, -1), keepdims=True, dtype=np.float32) + 1e-8
        self.X -= mean
        self.X /= std

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

class EEGSpectrogramTransformer(nn.Module):
    def __init__(self, num_classes=3, in_channels=23):
        super(EEGSpectrogramTransformer, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((8, 8))
        )
        self.d_model = 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=8, dim_feedforward=1024, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_classifier = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        c = self.cnn(x)
        batch_size, channels, freq, time = c.size()
        c = c.permute(0, 3, 1, 2).reshape(batch_size, time, channels * freq)
        transformer_out = self.transformer(c)
        return self.fc_classifier(transformer_out.mean(dim=1))

def apply_temporal_blend(raw_probs, window_size=5, warning_thresh=0.15, seizure_thresh=0.15):
    smoothed_probs = np.zeros_like(raw_probs)
    n_samples = len(raw_probs)
    for i in range(n_samples):
        start_idx = max(0, i - window_size + 1)
        smoothed_probs[i] = np.mean(raw_probs[start_idx : i + 1], axis=0)
        
    final_preds = np.zeros(n_samples, dtype=np.int64)
    for i, p in enumerate(smoothed_probs):
        if p[2] >= seizure_thresh:
            final_preds[i] = 2
        elif p[1] >= warning_thresh:
            final_preds[i] = 1
        else:
            final_preds[i] = 0
    return final_preds

# ==========================================
# 2. VISUALIZATION LOGIC
# ==========================================
def generate_timeline(patient_id="chb01", window_size=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Visualization for {patient_id} on {device}...")

    # Load Data & Model
    all_files = list(WINDOWS_DIR.glob(f"{patient_id}*_spectrograms.npz"))
    dataset = EEGDataset(all_files)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model_path = f"models/transformer_specific_{patient_id}.pth"
    model = EEGSpectrogramTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_raw_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            probs = torch.softmax(model(inputs.to(device)), dim=1)
            all_raw_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_raw_probs = np.concatenate(all_raw_probs, axis=0)
    all_labels = np.array(all_labels)
    blended_preds = apply_temporal_blend(all_raw_probs, window_size=window_size)

    # Find where a seizure happens to center the graph
    seizure_indices = np.where(all_labels == 2)[0]
    if len(seizure_indices) == 0:
        print("No seizures found in this dataset.")
        return

    first_seizure_idx = seizure_indices[0]
    
    # Slice the timeline to show 10 minutes before and 2 minutes after
    # Each window is 2 seconds. 10 mins = 300 windows. 2 mins = 60 windows.
    start_idx = max(0, first_seizure_idx - 300)
    end_idx = min(len(all_labels), first_seizure_idx + 60)
    
    time_axis = np.arange(start_idx, end_idx) * 2 / 60  # Convert windows to Minutes
    
    true_slice = all_labels[start_idx:end_idx]
    ai_slice = blended_preds[start_idx:end_idx]

    # Plotting
    plt.figure(figsize=(15, 4))
    
    # Plot Ground Truth Background
    plt.plot(time_axis, true_slice, label="Actual Brain State (Ground Truth)", color='black', linewidth=3, linestyle="--")
    
    # Plot AI Prediction Line
    plt.plot(time_axis, ai_slice, label="AI Filtered Prediction", color='#0078D7', linewidth=4, alpha=0.8)

    # Highlight Warning and Seizure Zones
    plt.fill_between(time_axis, 0, 2.5, where=(ai_slice==1), color='yellow', alpha=0.3, label="AI Warning Alarm Triggered")
    plt.fill_between(time_axis, 0, 2.5, where=(ai_slice==2), color='red', alpha=0.3, label="AI Seizure Alarm Triggered")

    # Formatting the Graph
    plt.yticks([0, 1, 2], ["Normal (0)", "Warning (1)", "Seizure (2)"], fontsize=12)
    plt.xlabel(f"Time (Minutes in Recording)", fontsize=12, fontweight='bold')
    plt.title(f"AI Real-Time Seizure Prediction Timeline (Patient: {patient_id})", fontsize=16, fontweight='bold')
    plt.legend(loc="upper left")
    plt.ylim(-0.2, 2.5)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Save and Show
    os.makedirs("plots", exist_ok=True)
    save_path = f"plots/timeline_{patient_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"--> Success! Visual timeline saved to: {save_path}")

if __name__ == "__main__":
    generate_timeline(patient_id="chb02", window_size=5) # Changed from chb01