import os
from config import WINDOWS_DIR

# Import your functions directly from your other files
from train_lstm import train_patient_specific
from evaluate import run_evaluation
from visualize import generate_timeline

def process_all_patients():
    # 1. Scan the data directory to find all unique patient IDs
    all_files = list(WINDOWS_DIR.glob("*_spectrograms.npz"))
    
    if not all_files:
        print("No files found. Check your WINDOWS_DIR path!")
        return
        
    # Extracts the "chbXX" part from the file names and removes duplicates
    patients = sorted(list(set([f.name.split('_')[0] for f in all_files])))
    
    print(f"\n🚀 Found {len(patients)} patients to process: {patients}\n")
    
    # 2. Loop through every single patient automatically
    for patient in patients:
        
        # ========================================================
        # SMART RESUME CHECK: Does this patient's model already exist?
        # ========================================================
        expected_model = f"models/transformer_specific_{patient}.pth"
        if os.path.exists(expected_model):
            print(f"⏭️ Skipping {patient}... (Model already exists. Resuming from next patient!)")
            continue
        # ========================================================
            
        print("\n" + "🔥"*25)
        print(f"   STARTING PIPELINE FOR: {patient}")
        print("🔥"*25)
        
        try:
            # Step A: Train the model
            print(f"\n>>> [1/3] Training Transformer for {patient}...")
            train_patient_specific(target_patient=patient, max_epochs=20)
            
            # Step B: Evaluate the model
            print(f"\n>>> [2/3] Running Blending Filter for {patient}...")
            run_evaluation(patient_id=patient, window_size=5)
            
            # Step C: Generate the timeline graph
            print(f"\n>>> [3/3] Generating Timeline Graph for {patient}...")
            generate_timeline(patient_id=patient, window_size=5)
            
        except Exception as e:
            # If one patient's data is corrupted, this stops the whole script from crashing
            print(f"\n❌ ERROR processing {patient}: {e}")
            print("Moving to the next patient...\n")

if __name__ == "__main__":
    process_all_patients()