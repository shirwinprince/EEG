import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import data directory from config
from config import WINDOWS_DIR


# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class EEGDataset(Dataset):
    def __init__(self, npz_files):
        print(f"Loading {len(npz_files)} file(s) for evaluation...")
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
            
        # In-place normalization
        mean = np.mean(self.X, axis=(-2, -1), keepdims=True, dtype=np.float32)
        std = np.std(self.X, axis=(-2, -1), keepdims=True, dtype=np.float32) + 1e-8
        self.X -= mean
        self.X /= std

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
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
        attended_features = transformer_out.mean(dim=1)
        return self.fc_classifier(attended_features)


# ==========================================
# 3. TEMPORAL PROBABILITY BLENDING FILTER
# ==========================================
def apply_temporal_blend(raw_probs, window_size=5, warning_thresh=0.4, seizure_thresh=0.5):
    """
    Blends probabilities over consecutive time windows using a moving average filter.
    """
    smoothed_probs = np.zeros_like(raw_probs)
    n_samples = len(raw_probs)
    
    # Calculate moving window average
    for i in range(n_samples):
        start_idx = max(0, i - window_size + 1)
        smoothed_probs[i] = np.mean(raw_probs[start_idx : i + 1], axis=0)
        
    final_preds = np.zeros(n_samples, dtype=np.int64)
    for i, p in enumerate(smoothed_probs):
        if p[2] >= seizure_thresh:
            final_preds[i] = 2  # Seizure / Ictal
        elif p[1] >= warning_thresh:
            final_preds[i] = 1  # Warning / Preictal
        else:
            final_preds[i] = 0  # Normal / Interictal
            
    return final_preds


# ==========================================
# 4. EVALUATION FUNCTION    
# ==========================================
def run_evaluation(patient_id="chb01", model_path=None, window_size=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    if model_path is None:
        model_path = f"models/transformer_specific_{patient_id}.pth"
        if not os.path.exists(model_path):
            model_path = f"models/transformer_specific_{patient_id}.pth"

    # Find patient files
    all_files = list(WINDOWS_DIR.glob(f"{patient_id}*_spectrograms.npz"))
    if not all_files:
        print(f"Error: No test files found for {patient_id} in {WINDOWS_DIR}")
        return

    test_dataset = EEGDataset(all_files)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load Model
    model = EEGSpectrogramTransformer(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_raw_probs = []
    all_labels = []

    print("Extracting raw prediction probabilities...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_raw_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_raw_probs = np.concatenate(all_raw_probs, axis=0)
    all_labels = np.array(all_labels)

    # 1. Raw Predictions (Unfiltered)
    raw_preds = np.argmax(all_raw_probs, axis=1)

    # 2. Blended Predictions (Filtered) - STRICTER CLINICAL THRESHOLDS
    # 2. Blended Predictions (Filtered) - OPTIMAL CLINICAL THRESHOLDS
    blended_preds = apply_temporal_blend(
        all_raw_probs, 
        window_size=window_size, 
        warning_thresh=0.15, 
        seizure_thresh=0.15  
    )

    # Display Comparison
    target_names = ["Interictal (0)", "Preictal (1)", "Ictal (2)"]
    
    print("\n" + "=" * 50)
    print("--- 1. RAW MODEL OUTPUT (No Blending) ---")
    print(classification_report(all_labels, raw_preds, target_names=target_names, zero_division=0))

    print("\n" + "=" * 50)
    print(f"--- 2. BLENDED OUTPUT (Moving Window = {window_size}) ---")
    print(classification_report(all_labels, blended_preds, target_names=target_names, zero_division=0))
    print("=" * 50)

    # Save Blended Confusion Matrix
    cm = confusion_matrix(all_labels, blended_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal (0)", "Warning (1)", "Seizure (2)"],
                yticklabels=["Normal (0)", "Warning (1)", "Seizure (2)"])
    plt.title(f'Blended Predictions vs Actual (Patient: {patient_id})')
    plt.ylabel('Actual True State')
    plt.xlabel('Blended Predicted State')

    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/blended_confusion_matrix_{patient_id}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n--> Saved Blended Confusion Matrix to: {plot_path}")


# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    run_evaluation(patient_id="chb02", window_size=5) # Changed from chb01