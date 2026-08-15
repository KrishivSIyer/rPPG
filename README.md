# Remote Photoplethysmography (rPPG) - Contactless Heart Rate Estimation

This repository contains the pipeline for extracting contactless blood volume pulse (rPPG) signals from facial video feeds using facial landmark detection (MediaPipe) and classical/deep signal processing techniques.

---

## 📁 Repository Structure

```text
rPPG/
├── models/
│   └── face_landmarker.task      # Pretrained MediaPipe Face Landmarker model
├── data/
│   └── UBFC-rPPG/               # Benchmark dataset folder
│       └── DATASET1/
│           └── subject1/
│               ├── vid-001.avi   # Subject video file (25 / 30 FPS)
│               └── gtdump.xmp    # Ground truth PPG recording
├── src/
│   ├── download_models.py       # Script to download MediaPipe model files
│   ├── pipeline.py              # Landmark & ROI RGB signal extraction module
│   └── __init__.py
├── results/
│   └── subject1_rgb.csv         # Extracted RGB mean time-series data
├── rPPG_Project_Proposal.md     # Full project design proposal & milestones
├── requirements.txt             # Python package dependencies
└── README.md                    # Quickstart guide & data handoff docs
```

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Model & Pipeline
Make sure the MediaPipe model task file is downloaded:
```bash
python src/download_models.py
```

Run the pipeline test script to extract ROI RGB signals from `subject1`:
```bash
python src/pipeline.py
```
*(This will process `data/UBFC-rPPG/DATASET1/subject1/vid-001.avi` and automatically save the output to `results/subject1_rgb.csv`).*

---

## 📊 Handoff Guide for Signal Processing (CHROM Baseline)

If you are working on **Stage 1: CHROM Implementation & Heart Rate Estimation**, here is where to get the data and how to use it:

### Data Location & Format

#### Option A: Load from Pre-extracted CSV File
If `python src/pipeline.py` has already been run, the frame-by-frame mean RGB values are saved at:
- **Path**: [`results/subject1_rgb.csv`](file:///c:/Users/krish/OneDrive/Documents/rppg/results/subject1_rgb.csv)
- **Format**:
  ```csv
  R,G,B
  138.42,104.15,90.28
  138.51,104.22,90.35
  138.39,104.10,90.22
  ```

#### Option B: Extract Programmatically in Python
```python
from src.pipeline import FaceLandmarkExtractor
import pandas as pd
import numpy as np

# Initialize extractor
extractor = FaceLandmarkExtractor()

# Process video file
video_path = "data/UBFC-rPPG/DATASET1/subject1/vid-001.avi"
rgb_signals, fps = extractor.extract_video_rgb(video_path, save_path="results/subject1_rgb.csv")

extractor.close()

# rgb_signals shape: (T_frames, 3)
# fps: e.g., 30.0
```

---

## 🛠️ Next Steps for CHROM Signal Processing

Using `rgb_signals` (shape `(T, 3)`) and `fps` (30.0):

1. **Chrominance Signal Combination (CHROM)**:
   - Normalize R, G, B by their temporal mean: $\mu_R, \mu_G, \mu_B$.
   - Calculate chrominance signals:
     $$X = 3R_n - 2G_n$$
     $$Y = 1.5R_n + G_n - 1.5B_n$$
   - Bandpass filter $X$ and $Y$ ($0.7 \text{ Hz} - 3.0 \text{ Hz}$, Butterworth 3rd order).
   - Combine $S_{CHROM} = X - \alpha Y$ where $\alpha = \frac{\sigma(X)}{\sigma(Y)}$ over sliding windows.

2. **Heart Rate Estimation**:
   - Compute Welch Power Spectral Density (PSD) or FFT over $S_{CHROM}$.
   - Identify peak frequency $f_{peak}$ in the pulse range ($0.7 - 3.0 \text{ Hz}$).
   - $\text{Heart Rate (BPM)} = f_{peak} \times 60$.

3. **Ground Truth Comparison**:
   - Compare estimated Heart Rate (BPM) against ground truth PPG timestamps in `data/UBFC-rPPG/DATASET1/subject1/`.
