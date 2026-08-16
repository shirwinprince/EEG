import mne
import numpy as np
from pathlib import Path
from config import TARGET_CHANNELS, FILTER_LOW, FILTER_HIGH, NOTCH_FREQ, RAW_DIR

def load_and_filter_edf(edf_path):
    """
    Loads an EDF file, standardizes channels, and applies filtering.
    Zero-pads missing channels to maintain fixed tensor dimensions for CNN input.
    
    Args:
        edf_path (str or Path): Path to the .edf file.
        
    Returns:
        final_data (np.ndarray): Filtered & padded EEG data array of shape (23, samples).
        sfreq (float): Sampling frequency.
        channels (list): List of target channel names used.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    
    available_channels = raw.ch_names
    channels_to_pick = [ch for ch in TARGET_CHANNELS if ch in available_channels]
    
    if len(channels_to_pick) < len(TARGET_CHANNELS):
        missing = set(TARGET_CHANNELS) - set(channels_to_pick)
        print(f"Warning: {Path(edf_path).name} is missing channels: {missing}. Zero-padding will be applied.")
        
    # FIX: Use pick() instead of the legacy pick_channels()
    raw.pick(picks=channels_to_pick)
    raw.reorder_channels(channels_to_pick)
    
    # 2. Filtering
    raw.notch_filter(freqs=NOTCH_FREQ, verbose='ERROR')
    raw.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, verbose='ERROR')
    
    raw_data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # FIX: Zero-pad missing channels to guarantee a fixed shape of (23, samples)
    final_data = np.zeros((len(TARGET_CHANNELS), raw_data.shape[1]))
    for i, target_ch in enumerate(TARGET_CHANNELS):
        if target_ch in channels_to_pick:
            idx = channels_to_pick.index(target_ch)
            final_data[i, :] = raw_data[idx, :]
            
    return final_data, sfreq, TARGET_CHANNELS

if __name__ == "__main__":
    print("Testing EDF loader and filter...")
    test_files = list(RAW_DIR.rglob("*.edf"))
    
    if test_files:
        test_file = test_files[0]
        print(f"Processing {test_file.name}...")
        
        filtered_data, sfreq, channels = load_and_filter_edf(test_file)
        
        print(f"Success! Data shape: {filtered_data.shape} (Channels x Samples)")
        print(f"Sampling frequency: {sfreq} Hz")
    else:
        print("No .edf files found in data/raw/ to test.")