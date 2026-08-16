from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = PROCESSED_DIR / "annotations"
WINDOWS_DIR = PROCESSED_DIR / "windows"
SPLITS_DIR = PROCESSED_DIR / "splits"

# Create directories if they don't exist
for d in [ANNOTATIONS_DIR, WINDOWS_DIR, SPLITS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# EEG Parameters (CHB-MIT specific)
SAMPLING_RATE = 256 
# 23 standard channels commonly shared across CHB-MIT patients
TARGET_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'
]

# Signal Processing
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
NOTCH_FREQ = 60.0

# Windowing & Labeling
WINDOW_SIZE_SEC = 2
PREICTAL_DURATION_SEC = 900  # 15 minutes before seizure