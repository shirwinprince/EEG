import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from config import WINDOWS_DIR
from model import EEGSpectrogramCNN
from evaluate import evaluate_model

# 1. Custom Dataset Handler
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
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# 2. Class Weight Calculator
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    
    weight_tensor = torch.ones(3, dtype=torch.float32)
    for c, w in zip(classes, weights):
        weight_tensor[int(c)] = w
    return weight_tensor

# 3. LOPO Training Loop with Early Stopping & LR Scheduler
def train_lopo(max_epochs=30, batch_size=32, initial_lr=0.001, patience=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    all_npz_files = list(WINDOWS_DIR.glob("*_spectrograms.npz"))
    all_patients = sorted(list(set([f.name.split('_')[0] for f in all_npz_files])))
    
    os.makedirs("models", exist_ok=True)
    
    for test_patient in all_patients:
        print(f"\n{'='*60}")
        print(f"Fold: Testing on {test_patient} (Training on all others)")
        print(f"{'='*60}")
        
        # Outer Split (Isolate test patient)
        train_pool_files = [f for f in all_npz_files if test_patient not in f.name]
        test_files = [f for f in all_npz_files if test_patient in f.name]
        
        if not test_files:
            continue
            
        # Inner Split (80% Train, 20% Val) to prevent data leakage
        train_files, val_files = train_test_split(train_pool_files, test_size=0.2, random_state=42)
            
        print("Loading training and validation data into memory...")
        train_dataset = EEGDataset(train_files)
        val_dataset = EEGDataset(val_files)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        class_weights = get_class_weights(train_dataset.y).to(device)
        print(f"Class Weights applied: {class_weights.cpu().numpy()}")
        
        model = EEGSpectrogramCNN(num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        
        # Setup Scheduler and Early Stopping trackers
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        best_val_loss = float('inf')
        epochs_no_improve = 0
        model_path = f"models/cnn_fold_{test_patient}.pth"
        
        # Epoch Loop
        for epoch in range(max_epochs):
            # --- TRAIN PHASE ---
            model.train()
            running_train_loss = 0.0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
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
            
            # Step the learning rate scheduler based on validation loss
            scheduler.step(avg_val_loss)
            
            # --- EARLY STOPPING CHECK ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_path)
                print(f"  [*] Val loss improved. Saved checkpoint.")
            else:
                epochs_no_improve += 1
                print(f"  [!] No improvement for {epochs_no_improve} epoch(s).")
                
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered! Training halted at epoch {epoch+1}.")
                break
                
        print(f"\nFinished training fold. Best model saved to {model_path}")
        
        # --- EVALUATE ON HELD-OUT PATIENT ---
        print(f"Loading test data for {test_patient}...")
        test_dataset = EEGDataset(test_files)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
        
        # Load the BEST weights before evaluating
        model.load_state_dict(torch.load(model_path, weights_only=True))
        evaluate_model(model, test_loader, device)
        
        # Break after one fold to verify pipeline stability
        print("\nBreaking after one fold to verify pipeline stability.")
        break 

if __name__ == "__main__":
    train_lopo()