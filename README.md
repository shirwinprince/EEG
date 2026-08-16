Markdown
# ⚡ NEURO-PULSE | Clinical Seizure Prediction Pipeline

NeuroPulse is a state-of-the-art Deep Learning pipeline and Clinical Telemetry Dashboard designed to predict epileptic seizures using continuous EEG data. By leveraging a **Vision Transformer (ViT) + CNN backbone**, the system analyzes 2D EEG spectrograms to detect microscopic biological warning signs (Preictal states) up to 10 minutes before clinical onset.

## 🌟 Key Features
* **Patient-Specific AI Models:** Generates customized Vision Transformers trained on individual patient biological fingerprints for near 100% preictal recall.
* **Temporal Blending Filter:** A custom 10-second sliding-window algorithmic filter that mathematically suppresses baseline movement/muscle artifacts to eliminate false alarms.
* **Premium Clinical Dashboard:** A Streamlit-powered UI featuring interactive Plotly/Matplotlib analytics and real-time predictive timelines.
* **Multimodal AI Integration:** Embedded Google Gemini 1.5 Vision integration allowing physicians to "chat" directly with the AI's generated EEG diagnostic matrices.

## 📂 Project Architecture
```text
project/
├── data/                  # Raw .edf and processed .npz files (Git Ignored)
├── models/                # Trained Patient-Specific .pth weights (Git Ignored)
├── plots/                 # Clinical Timelines & Confusion Matrices
├── src/
│   ├── preprocessing.py   # Converts 1D EEG signals to 2D Spectrograms
│   ├── train_lstm.py      # ViT Model Architecture & Training Loop
│   ├── evaluate.py        # Temporal Filter & Model Validation
│   ├── visualize.py       # Clinical Timeline Generator
│   └── run_all.py         # Automated Master Pipeline Execution
├── app.py                 # Streamlit UI Dashboard
├── requirements.txt       # Python Dependencies
└── README.md              # Project Documentation
🚀 Installation & Setup
1. Clone the repository

Bash
git clone [https://github.com/YOUR_USERNAME/NeuroPulse.git](https://github.com/YOUR_USERNAME/NeuroPulse.git)
cd NeuroPulse
2. Create a Virtual Environment & Install Dependencies

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
3. Add Data
Place your raw .edf or preprocessed .npz EEG files inside the data/ directory.

🧠 Running the Pipeline
To automatically train the Vision Transformers, evaluate the Temporal Filter, and generate the clinical plots for all patients in your data folder:

Bash
python src/run_all.py
🖥️ Launching the Clinical Dashboard
To launch the NeuroPulse UI for physician review:

Bash
streamlit run app.py
