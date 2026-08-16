# ==========================================
# 1. IMPORTS & CONFIGURATION
# ==========================================
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure your data directory here! 
from config import WINDOWS_DIR

# ==========================================
# 2. DATASET & LOSS FUNCTION
# ==========================================
class EEGArrayDataset(Dataset):
    """Simplified Dataset for Patient-Specific Array Splitting"""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==========================================
# 3. THE CNN-TRANSFORMER ARCHITECTURE
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
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=True
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
        out = self.fc_classifier(attended_features)
        return out

# ==========================================
# 4. PATIENT-SPECIFIC TRAINING LOOP
# ==========================================
def train_patient_specific(target_patient="chb01", max_epochs=20, batch_size=64, initial_lr=0.0003, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Find ONLY the files for our target patient
    patient_files = sorted(list(WINDOWS_DIR.glob(f"{target_patient}*_spectrograms.npz")))
    
    if not patient_files:
        print(f"ERROR: No .npz files found for {target_patient} in {WINDOWS_DIR}")
        return
        
    print(f"\n{'='*60}")
    print(f"PATIENT-SPECIFIC TRAINING: {target_patient}")
    print(f"{'='*60}")
    
    # 2. Load all of this patient's data into memory
    print(f"Loading {len(patient_files)} files for {target_patient}...")
    X_all, y_all = [], []
    for f in patient_files:
        data = np.load(f)
        X_all.append(data['X'].astype(np.float32))
        y_all.append(data['y'])
        
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    print(f"Total windows found for {target_patient}: {len(y_all)}")
    
    # 3. Stratified Split (Guarantees seizures are in both Train and Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # 4. Normalize based ONLY on training data to prevent data leakage
   # 4. Normalize (Per-Window Z-Score)
    print("Normalizing training and test data...")
    
    # Normalize Training Set
    mean_train = np.mean(X_train, axis=(-2, -1), keepdims=True, dtype=np.float32)
    std_train = np.std(X_train, axis=(-2, -1), keepdims=True, dtype=np.float32) + 1e-8
    X_train -= mean_train
    X_train /= std_train
    
    # Normalize Test Set
    mean_test = np.mean(X_test, axis=(-2, -1), keepdims=True, dtype=np.float32)
    std_test = np.std(X_test, axis=(-2, -1), keepdims=True, dtype=np.float32) + 1e-8
    X_test -= mean_test
    X_test /= std_test
    
    # 5. Create PyTorch Datasets
    train_dataset = EEGArrayDataset(X_train, y_train)
    test_dataset = EEGArrayDataset(X_test, y_test)
    
    # 6. Balanced Sampler (Force AI to look at seizures)
    class_counts = np.bincount(y_train, minlength=3)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Class distribution in train split: {class_counts}")
    
    # 7. Model Setup
    model = EEGSpectrogramTransformer(num_classes=3).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    os.makedirs("models", exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # SAVES AS THE NEW SPECIFIC MODEL
    model_path = f"models/transformer_specific_{target_patient}.pth"
    
    # 8. Training Loop
    for epoch in range(max_epochs):
        model.train()
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        
        # Test/Validation Phase
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()
                
        avg_test_loss = running_test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{max_epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        
        scheduler.step(avg_test_loss)
        
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  [*] Test loss improved. Saved checkpoint.")
        else:
            epochs_no_improve += 1
            print(f"  [!] No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break
            
    print(f"\nFinished training. Best model saved to {model_path}")

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    train_patient_specific(target_patient="chb02") # Changed from chb01