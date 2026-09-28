# 🫀 Remote Photoplethysmography (rPPG) — Contactless Vital Sign Monitoring

A complete open-source framework for extracting contactless blood volume pulse (BVP) signals and estimating Heart Rate (BPM) from standard facial video feeds. This repository implements both **classical chrominance-based signal processing (CHROM)** and a **deep temporal attention neural network (`rPPGNet`)**.

Blood pumping through facial capillaries causes minute, sub-pixel skin color variations (~<1% intensity shift). By isolating facial skin regions (forehead and cheeks) using **MediaPipe Face Landmarker**, this software amplifies and extracts the subtle pulsatile signal to measure physiological heart rate without physical contact or specialized hardware.

---

## ⚙️ How It Works

```
                          ┌───────────────────────────┐
                          │    Input Facial Video     │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  MediaPipe Face Mesh ROI  │  (468 Landmarks)
                          │   Forehead & Cheek Mask   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │    Per-Frame Mean RGB     │  (N, 3) Time Series
                          └─────────────┬─────────────┘
                                        │
               ┌────────────────────────┴────────────────────────┐
               ▼                                                 ▼
┌──────────────────────────────┐                ┌──────────────────────────────┐
│     Classical Baseline       │                │      Deep Learning Model     │
│   (CHROM Signal Extraction)  │                │          (`rPPGNet`)         │
│  Xs = 3R - 2G, Ys = 1.5R+G-1.5B               │  - Spatial 1D Conv Blocks    │
│  S = Xs - α * Ys             │                │  - Temporal Attention Layer  │
└──────────────┬───────────────┘                └──────────────┬───────────────┘
               │                                               │
               └────────────────────────┬──────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │    Butterworth Bandpass   │  (0.75 Hz – 2.5 Hz / 45-150 BPM)
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Welch PSD & Peak FFT    │  Dominant Frequency → Heart Rate
                          └───────────────────────────┘
```

### 1. Facial Landmark & ROI Extraction
Standard video input (e.g., webcam or dataset video) is processed frame-by-frame:
* **MediaPipe Face Mesh** detects 468 3D facial landmarks.
* Dynamic masks isolate clean skin regions over the **forehead** and **left/right cheeks**, ignoring eyes, mouth, and background.
* Spatial averaging over the masked pixels produces a raw RGB time-series signal $(N, 3)$, where $N$ is the frame count.

### 2. Classical Chrominance Signal Extraction (CHROM)
CHROM eliminates illumination drift and head motion artifacts by taking ratios between normalized RGB channels:
$$X_s = 3R_n - 2G_n, \quad Y_s = 1.5R_n + G_n - 1.5B_n$$
$$S_{\text{CHROM}} = X_s - \alpha Y_s \quad \text{where} \quad \alpha = \frac{\sigma(X_s)}{\sigma(Y_s)}$$

### 3. Deep Temporal Attention Neural Network (`rPPGNet`)
* **Spatial 1D CNN:** Extracts high-level spatiotemporal features from windowed RGB time-series inputs.
* **Temporal Attention Module:** Dynamically calculates attention scores across temporal frames, suppressing noisy frames corrupted by motion or lighting shifts.
* **Negative Pearson Correlation Loss:** Optimizes for physiological pulse wave shape rather than absolute MSE.

### 4. Heart Rate Extraction & Frequency Analysis
* **Butterworth Bandpass Filter:** Filters out non-physiological frequencies outside the 45–150 BPM range ($0.75\text{ Hz} - 2.5\text{ Hz}$).
* **Welch Power Spectral Density (PSD):** Finds the dominant frequency peak to compute the heart rate in Beats Per Minute (BPM): $\text{BPM} = f_{\text{peak}} \times 60$.

---

## 📁 Repository Structure

```text
rPPG/
├── models/
│   ├── face_landmarker.task     # Pretrained MediaPipe Face Landmarker model task
│   └── rppgnet_best.pth         # Saved PyTorch checkpoint for rPPGNet
├── data/
│   └── UBFC-rPPG/              # Dataset directory for benchmark video feeds
├── src/
│   ├── pipeline.py             # Facial landmark ROI & RGB signal extractor
│   ├── heart_rate.py           # Bandpass filtering, CHROM extraction & Welch PSD
│   ├── evaluate.py             # Classical baseline HR evaluation script
│   ├── download_models.py      # Downloader for MediaPipe task dependencies
│   └── training/
│       ├── dataset.py          # Sliding window dataset loader (150 frames, 50% stride)
│       ├── model.py            # rPPGNet network architecture
│       ├── train.py            # PyTorch training loop & Pearson correlation loss
│       ├── run_training.py     # Script to train rPPGNet on extracted signals
│       └── evaluate_model.py   # Full model inference & visual evaluation pipeline
├── results/
│   ├── subject1_rgb.csv        # Extracted frame-by-frame mean RGB time-series
│   ├── gtdump.xmp               # Ground truth pulse oximeter recording
│   ├── heart_rate_analysis.png # Generated classical baseline plot
│   └── model_vs_baseline.png   # Comparative evaluation plot
├── rPPG_Roadmap.md             # Development roadmap and stage milestones
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview & usage guide
```

---

## 🛠️ User Guide: How to Try Out the Software

Follow these steps to run contactless heart rate estimation on your own video or benchmark datasets.

### 1. Prerequisites & Installation

Ensure you have **Python 3.10+** installed. Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/KrishivSIyer/rPPG.git
cd rPPG

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required Python dependencies
pip install -r requirements.txt
```

### 2. Download MediaPipe Model Task
Download the pretrained MediaPipe Face Landmarker model:
```bash
python src/download_models.py
```

---

### 3. Step 1: Extract RGB Signals from Video
Run the face landmark pipeline on a facial video file to extract frame-by-frame mean RGB time-series data:

```bash
python src/pipeline.py
```
> **Custom Video:** You can pass your own video file path inside `src/pipeline.py` or programmatically import `FaceLandmarkExtractor` to generate `results/subject1_rgb.csv`.

---

### 4. Step 2: Run Classical Heart Rate Estimation (CHROM)
Estimate heart rate using the classical CHROM method and save a visual analysis plot:

```bash
python src/heart_rate.py --csv results/subject1_rgb.csv --fps 30.0
```
This generates `results/heart_rate_analysis.png` showing the raw RGB channels, filtered pulse wave, and Welch PSD peak.

To evaluate against a ground truth PPG recording:
```bash
python src/evaluate.py --csv results/subject1_rgb.csv --gt results/gtdump.xmp --fps 30.0
```

---

### 5. Step 3: Train the Deep Learning Model (`rPPGNet`)
To train `rPPGNet` on extracted RGB time-series signals and ground truth PPG signals:

```bash
python src/training/run_training.py --rgb results/subject1_rgb.csv --gt results/gtdump.xmp --epochs 50 --save_dir models
```
* Slices continuous signals into 150-frame windows (~5s at 30 FPS) with a 50% overlap.
* Uses early stopping on validation Pearson correlation to prevent overfitting.
* Saves the best model checkpoint to `models/rppgnet_best.pth`.

---

### 6. Step 4: Evaluate Trained `rPPGNet` Model
Compare the learned `rPPGNet` model against the classical CHROM baseline:

```bash
python src/training/evaluate_model.py --rgb results/subject1_rgb.csv --gt results/gtdump.xmp --model models/rppgnet_best.pth
```
This script runs windowed model inference, stitches the full predicted waveform via overlap-add, computes Heart Rate (BPM) & Pearson correlation, and saves the comparison plot to `results/model_vs_baseline.png`.

---

## 📋 Requirements
* Python 3.10+
* `torch >= 2.0.0`
* `opencv-python`
* `mediapipe`
* `numpy`
* `scipy`
* `pandas`
* `matplotlib`

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more details.
