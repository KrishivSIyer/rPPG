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
├── predict.py                   # Single-command end-to-end video inference & HR prediction
├── models/
│   ├── face_landmarker.task     # Pretrained MediaPipe Face Landmarker model task
│   ├── haarcascade_frontalface_default.xml # OpenCV face detection fallback
│   └── rppgnet_best.pth         # Fine-tuned PyTorch checkpoint for rPPGNet
├── data/
│   └── UBFC-rPPG/               # Dataset directory for benchmark video feeds
├── src/
│   ├── pipeline.py              # Facial landmark ROI & RGB signal extractor (MediaPipe + OpenCV fallback)
│   ├── heart_rate.py            # Zero-phase NumPy bandpass filtering, CHROM extraction & PSD
│   ├── evaluate.py              # Classical baseline HR evaluation script
│   ├── download_models.py       # Downloader for MediaPipe task dependencies
│   └── training/
│       ├── dataset.py           # Sliding window dataset loader (150 frames, 50% stride)
│       ├── model.py             # rPPGNet network architecture
│       ├── train.py             # PyTorch training loop, Pearson loss & fine-tuning support
│       ├── run_training.py      # Script to train/fine-tune rPPGNet on extracted signals
│       └── evaluate_model.py    # Full model inference & visual evaluation pipeline
├── results/
│   ├── subject1_rgb.csv         # Extracted frame-by-frame mean RGB time-series
│   ├── gtdump.xmp                # Ground truth pulse oximeter recording
│   ├── heart_rate_analysis.png  # Generated classical baseline plot
│   └── subject49_model_vs_baseline.png # Comparative fine-tuned model plot
├── rPPG_Roadmap.md              # Development roadmap and stage milestones
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview & usage guide
```

---

## 🛠️ User Guide: How to Try Out the Software

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

---

### ⚡ Quickstart: End-to-End Prediction on Any Video

To estimate Heart Rate (BPM) from any facial video feed in **one single command**:

```bash
python predict.py --video path/to/your_video.mp4
```

**What happens automatically:**
1. **Face Tracking**: Extracts face ROIs using MediaPipe Face Mesh (or OpenCV Haar Cascade if MediaPipe is blocked).
2. **Signal Processing**: Computes frame-by-frame RGB time series.
3. **Dual Prediction**: Calculates estimated Heart Rate using both classical **CHROM** baseline and fine-tuned **rPPGNet**.
4. **Visual Report**: Saves a comparison plot of predicted pulse waveforms and PSD frequency spectrum to `results/prediction_output.png`.

---

### Advanced Step-by-Step Pipeline & Fine-Tuning

If you want to run individual stages, train from scratch, or fine-tune on custom/noisy videos:

#### 1. Extract RGB Signals from Custom Video
```bash
python src/pipeline.py --video path/to/video.avi --output results/custom_rgb.csv
```

#### 2. Run Classical CHROM Heart Rate Estimation
```bash
python src/heart_rate.py --csv results/custom_rgb.csv --fps 30.0
```

#### 3. Fine-Tune `rPPGNet` on Custom/Noisy Datasets
Fine-tune the pre-trained weights (`models/rppgnet_best.pth`) on a new dataset with a reduced learning rate (`1e-4`):
```bash
python src/training/run_training.py --rgb results/custom_rgb.csv --gt path/to/ground_truth.txt --weights models/rppgnet_best.pth --lr 1e-4 --epochs 40 --save_dir models
```

#### 4. Evaluate Fine-Tuned Model vs Baseline
```bash
python src/training/evaluate_model.py --rgb results/custom_rgb.csv --gt path/to/ground_truth.txt --model models/rppgnet_best.pth --output results/custom_evaluation.png
```

---

## 📋 Requirements
* Python 3.10+
* `torch >= 1.10.0`
* `torchvision`
* `opencv-python`
* `mediapipe` (MediaPipe Face Mesh with automatic OpenCV Haar Cascade fallback)
* `numpy` (Zero-phase FFT bandpass filtering & PSD calculation)
* `pandas`
* `matplotlib`

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more details.
