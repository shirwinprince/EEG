import numpy as np
import pandas as pd
from config import WINDOW_SIZE_SEC, SAMPLING_RATE, PREICTAL_DURATION_SEC

def create_windows_and_labels(data, annotations_df, filename):
    """
    Slices a continuous EEG signal into fixed windows and assigns classification labels.
    
    Args:
        data (np.ndarray): Shape (channels, samples)
        annotations_df (pd.DataFrame): The parsed seizures for the current patient
        filename (str): The specific EDF file name (e.g., 'chb01_03.edf')
        
    Returns:
        X (np.ndarray): Shape (num_windows, channels, window_samples)
        y (np.ndarray): Shape (num_windows,) with labels [0, 1, 2]
    """
    window_samples = int(WINDOW_SIZE_SEC * SAMPLING_RATE)
    total_samples = data.shape[1]
    
    # Filter annotations for this specific EDF file
    file_seizures = annotations_df[annotations_df['filename'] == filename]
    
    windows = []
    labels = []
    
    # Slide through the continuous data without overlap
    for start_idx in range(0, total_samples - window_samples + 1, window_samples):
        end_idx = start_idx + window_samples
        
        start_sec = start_idx / SAMPLING_RATE
        end_sec = end_idx / SAMPLING_RATE
        
        label = 0  # Default to Interictal
        
        # Check against all recorded seizures in this file
        for _, row in file_seizures.iterrows():
            sz_start = row['seizure_start']
            sz_end = row['seizure_end']
            
            # 1. Check Ictal (Does the window overlap with the seizure?)
            if (start_sec < sz_end) and (end_sec > sz_start):
                label = 2
                break
                
            # 2. Check Preictal (Does the window fall in the 15m window before a seizure?)
            preictal_start = max(0, sz_start - PREICTAL_DURATION_SEC)
            if (start_sec >= preictal_start) and (end_sec <= sz_start):
                label = 1
                break
                
        window_data = data[:, start_idx:end_idx]
        windows.append(window_data)
        labels.append(label)
        
    return np.array(windows), np.array(labels)