import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import EEGSpectrogramCNN
from train import EEGDataset
from config import WINDOWS_DIR

class GradCAM:
    """Custom Grad-CAM implementation for our custom EEG CNN."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to intercept the gradients and activations during the forward/backward pass
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward(retain_graph=True)
        
        # Global Average Pooling on the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations by the pooled gradients
        activations = self.activations.detach()[0]
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]
            
        # Average across all channels and apply ReLU to only keep positive influences
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize between 0 and 1
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

def run_gradcam_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")
    
    # 1. Load Model
    model = EEGSpectrogramCNN(num_classes=3).to(device)
    model_path = "models/cnn_fold_chb01.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Did you complete training?")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Initialize Grad-CAM on the last convolutional layer
    cam = GradCAM(model, model.conv2)
    
    # 2. Load Data (Let's find an Ictal/Seizure window from chb01)
    test_files = list(WINDOWS_DIR.glob("chb01*_spectrograms.npz"))
    dataset = EEGDataset(test_files)
    
    # Find the index of the first seizure window (Label == 2)
    seizure_idx = np.where(dataset.y == 2)[0]
    
    if len(seizure_idx) == 0:
        print("No seizure windows found in this dataset split.")
        return
        
    target_idx = seizure_idx[0]
    input_tensor, true_label = dataset[target_idx]
    
    # Add batch dimension and move to device
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # 3. Generate Heatmap for the Ictal class (class 2)
    heatmap = cam.generate_heatmap(input_tensor, class_idx=2)
    
    # 4. Plotting
    os.makedirs("results", exist_ok=True)
    
    # Average the original spectrogram across all 23 channels for visualization
    original_spectrogram = input_tensor.squeeze().cpu().numpy().mean(axis=0)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original Spectrogram (Avg Channels)\nTrue Label: Ictal")
    plt.imshow(original_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.ylabel("Frequency Bins")
    plt.xlabel("Time Steps")
    
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap\n(Where the model is looking)")
    # Resize heatmap to match original spectrogram dimensions
    import scipy.ndimage
    heatmap_resized = scipy.ndimage.zoom(heatmap, (original_spectrogram.shape[0] / heatmap.shape[0], 
                                                   original_spectrogram.shape[1] / heatmap.shape[1]))
    
    plt.imshow(original_spectrogram, aspect='auto', origin='lower', cmap='gray')
    plt.imshow(heatmap_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.5) # Overlay
    plt.ylabel("Frequency Bins")
    plt.xlabel("Time Steps")
    
    save_path = "results/gradcam_chb01_seizure.png"
    plt.savefig(save_path)
    print(f"\nSuccess! Grad-CAM visualization saved to {save_path}")

if __name__ == "__main__":
    run_gradcam_inference()