import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split

from config import WINDOWS_DIR
from model import EEGSpectrogramCNN
from evaluate import evaluate_model

# 1. Custom Dataset Handler with Per-Window Z-Score Normalization
class EEGDataset(Dataset):
    def __init__(self, npz_files):
        self.X = []
        self.y = []
        for f in npz_files:
            data = np.load(f)
            self.X.append(data['X'])
            self.y.append(data['y'])
            
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        
        # Apply Z-score normalization per window and channel to strip patient amplitude differences
        mean = np.mean(self.X, axis=(-2, -1), keepdims=True)
        std = np.std(self.X, axis=(-2, -1), keepdims=True) + 1e-8
        self.X = (self.X - mean) / std

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# 2. Focal Loss Module
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

# 3. LOPO Training Loop with Balanced Batch Sampler
def train_lopo(max_epochs=20, batch_size=64, initial_lr=0.0003, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    all_npz_files = list(WINDOWS_DIR.glob("*_spectrograms.npz"))
    all_patients = sorted(list(set([f.name.split('_')[0] for f in all_npz_files])))
    
    os.makedirs("models", exist_ok=True)
    
    for test_patient in all_patients:
        print(f"\n{'='*60}")
        print(f"Fold: Testing on {test_patient} (Training on all others)")
        print(f"{'='*60}")
        
        train_pool_files = [f for f in all_npz_files if test_patient not in f.name]
        test_files = [f for f in all_npz_files if test_patient in f.name]
        
        if not test_files:
            continue
            
        train_files, val_files = train_test_split(train_pool_files, test_size=0.2, random_state=42)
            
        print("Loading training and validation data into memory...")
        train_dataset = EEGDataset(train_files)
        val_dataset = EEGDataset(val_files)
        
        # --- BALANCED BATCH SAMPLER ---
        class_counts = np.bincount(train_dataset.y, minlength=3)
        class_counts = np.where(class_counts == 0, 1, class_counts) # Prevent division by zero
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_dataset.y]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Class distribution in train split: {class_counts}")
        print("WeightedRandomSampler active. Every batch will have equal class probability.")
        
        model = EEGSpectrogramCNN(num_classes=3).to(device)
        criterion = FocalLoss(gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        model_path = f"models/cnn_fold_{test_patient}.pth"
        
        for epoch in range(max_epochs):
            # --- TRAIN PHASE ---
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
            
            # --- VALIDATION PHASE ---
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    
            avg_val_loss = running_val_loss / len(val_loader)
            
            print(f"Epoch [{epoch+1}/{max_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_path)
                print(f"  [*] Val loss improved. Saved checkpoint.")
            else:
                epochs_no_improve += 1
                print(f"  [!] No improvement for {epochs_no_improve} epoch(s).")
                
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break
                
        print(f"\nFinished fold training. Best model saved to {model_path}")
        
        # --- EVALUATE ON HELD-OUT PATIENT ---
        print(f"Loading test data for {test_patient}...")
        test_dataset = EEGDataset(test_files)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
        
        model.load_state_dict(torch.load(model_path, weights_only=True))
        evaluate_model(model, test_loader, device)
        
        break 

if __name__ == "__main__":
    train_lopo()