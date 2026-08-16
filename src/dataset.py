import pandas as pd
import numpy as np
from pathlib import Path
from config import RAW_DIR, ANNOTATIONS_DIR, WINDOWS_DIR
from preprocessing import load_and_filter_edf
from windowing import create_windows_and_labels
from features import compute_stft

def process_patient(patient_dir):
    """Runs the full pipeline for a single patient and saves the output."""
    patient_id = patient_dir.name
    print(f"\n--- Processing Patient: {patient_id} ---")
    
    # Load patient annotations
    annot_path = ANNOTATIONS_DIR / f"{patient_id}_seizures.csv"
    if annot_path.exists():
        annotations_df = pd.read_csv(annot_path)
    else:
        annotations_df = pd.DataFrame() # Empty DataFrame if no seizures
        
    edf_files = list(patient_dir.glob("*.edf"))
    
    all_spectrograms = []
    all_labels = []
    all_metadata = []
    
    for edf_file in edf_files:
        print(f"Loading {edf_file.name}...")
        
        # 1. Preprocess (Filter & Pad)
        data, _, _ = load_and_filter_edf(edf_file)
        
        # 2. Window & Label
        windows, labels = create_windows_and_labels(data, annotations_df, edf_file.name)
        
        if len(windows) == 0:
            continue
            
        # 3. Extract Features (STFT)
        spectrograms = compute_stft(windows)
        
        all_spectrograms.append(spectrograms)
        all_labels.append(labels)
        
        # Track metadata for debugging/splitting later
        all_metadata.extend([{
            'patient_id': patient_id,
            'filename': edf_file.name,
            'window_index': i,
            'label': labels[i]
        } for i in range(len(labels))])
        
    if not all_spectrograms:
        print(f"No valid data generated for {patient_id}.")
        return
        
    # Concatenate all files for this patient
    patient_spectrograms = np.concatenate(all_spectrograms, axis=0)
    patient_labels = np.concatenate(all_labels, axis=0)
    metadata_df = pd.DataFrame(all_metadata)
    
    # Save optimized numpy arrays
    npz_path = WINDOWS_DIR / f"{patient_id}_spectrograms.npz"
    np.savez_compressed(npz_path, X=patient_spectrograms, y=patient_labels)
    
    # Save index reference
    csv_path = WINDOWS_DIR / f"{patient_id}_index.csv"
    metadata_df.to_csv(csv_path, index=False)
    
    print(f"Saved {len(patient_labels)} windows for {patient_id} to {WINDOWS_DIR.name}/")

if __name__ == "__main__":
    # Loop through all CHB folders in data/raw/
    patient_folders = [d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.startswith('chb')]
    
    for folder in patient_folders:
        process_patient(folder)
        
    print("\nDataset generation complete!")