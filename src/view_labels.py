import numpy as np
import matplotlib.pyplot as plt
import random
from config import WINDOWS_DIR

def visualize_eeg_classes():
    print("Searching your dataset for exactly one example of all 3 classes...")
    
    # Storage for our examples
    examples = {0: None, 1: None, 2: None}
    class_names = {0: "Normal (Interictal)", 1: "Warning (Preictal)", 2: "Seizure (Ictal)"}
    
    # Grab a list of files and shuffle them so we don't just search Class 0 forever
    all_files = list(WINDOWS_DIR.glob("*_spectrograms.npz"))
    random.shuffle(all_files)
    
    for f in all_files:
        data = np.load(f)
        X, y = data['X'], data['y']
        
        # Look for the classes we still haven't found yet
        for target_class in [0, 1, 2]:
            if examples[target_class] is None:
                # Find indices where this label exists in the array
                indices = np.where(y == target_class)[0]
                if len(indices) > 0:
                    # Grab the very first image of this class that we find
                    examples[target_class] = X[indices[0]]
                    
        # If we successfully found all 3 classes, stop searching!
        if all(v is not None for v in examples.values()):
            print("Successfully found all 3 classes! Generating plot...")
            break
            
    # --- Generate the Visual Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, target_class in enumerate([0, 1, 2]):
        ax = axes[i]
        if examples[target_class] is not None:
            # We select [0] to just show the 1st EEG channel out of the 22 channels
            spectrogram = examples[target_class][0] 
            
            # Draw the spectrogram heat map
            im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='jet')
            ax.set_title(f"Class {target_class}: {class_names[target_class]}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (within 2-second window)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title(f"Class {target_class} Not Found in searched files.")
            
    plt.suptitle("What the CNN Actually Sees (1 Channel of a 2-Second Window)", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_eeg_classes()