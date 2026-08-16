import os
import re
import pandas as pd
from pathlib import Path
from config import RAW_DIR, ANNOTATIONS_DIR

def parse_summary(txt_path):
    """Parses a single CHB-MIT summary text file into a DataFrame."""
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    records = []
    current_file = None
    
    for i, line in enumerate(lines):
        # Match file name
        if line.startswith('File Name:'):
            current_file = line.split(':')[1].strip()
            
        # Match number of seizures in file
        if 'Number of Seizures in File:' in line:
            num_seizures = int(line.split(':')[1].strip())
            if num_seizures > 0:
                # Look ahead for start/end times
                for j in range(1, num_seizures * 2 + 1, 2):
                    start_line = lines[i + j]
                    end_line = lines[i + j + 1]
                    
                    try:
                        start_sec = int(re.search(r'(\d+)\s+seconds', start_line).group(1))
                        end_sec = int(re.search(r'(\d+)\s+seconds', end_line).group(1))
                        records.append({
                            'filename': current_file,
                            'seizure_start': start_sec,
                            'seizure_end': end_sec
                        })
                    except AttributeError:
                        continue
                        
    return pd.DataFrame(records)

def process_all_annotations():
    """Loops through all raw patient folders and parses summaries."""
    print("Parsing annotations...")
    patient_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.startswith('chb')]
    
    for p_dir in patient_dirs:
        summary_file = p_dir / f"{p_dir.name}-summary.txt"
        if summary_file.exists():
            df = parse_summary(summary_file)
            if not df.empty:
                out_path = ANNOTATIONS_DIR / f"{p_dir.name}_seizures.csv"
                df.to_csv(out_path, index=False)
                print(f"Saved: {out_path.name} ({len(df)} seizures)")
            else:
                print(f"No seizures found or parsed in {p_dir.name}")
        else:
            print(f"Summary missing for {p_dir.name}")

if __name__ == "__main__":
    process_all_annotations()