import numpy as np
from scipy.signal import stft
from config import SAMPLING_RATE

def compute_stft(windows):
    """
    Converts 1D EEG windows into 2D STFT spectrograms.
    
    Args:
        windows (np.ndarray): Shape (num_windows, channels, samples) e.g., (N, 23, 512)
        
    Returns:
        spectrograms (np.ndarray): Magnitude spectrograms, shape (N, channels, freq_bins, time_steps)
    """
    # fs=256, nperseg=128 (0.5s window), noverlap=64 (50% overlap)
    # This yields a good balance of time and frequency resolution for 2s windows
    frequencies, times, Zxx = stft(
        windows, 
        fs=SAMPLING_RATE, 
        nperseg=128, 
        noverlap=64, 
        axis=-1
    )
    
    # We only need the magnitude (absolute value) of the complex STFT output for the CNN
    spectrograms = np.abs(Zxx)
    
    return spectrograms

if __name__ == "__main__":
    print("Feature extraction utility ready.")