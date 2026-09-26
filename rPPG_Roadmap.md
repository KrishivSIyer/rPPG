# rPPG Contactless Vital Sign Monitoring — Full Project Roadmap

## Project Summary
Extract heart rate, breathing rate, and HRV from ordinary facial video with no sensors.
Blood pumping through facial capillaries causes sub-pixel color changes in skin (~<1% intensity shift).
The pipeline detects, amplifies, and analyzes this signal using classical methods first, then replaces the core with a learned deep model.

---

## ✅ Stage 1: Completed Pipeline (Mid-Sem)

### 1.1 Face Landmark Extraction
- **Tool:** MediaPipe Face Mesh (pretrained, 468 landmarks)
- **What it does:** Detects forehead and cheek ROI coordinates per frame
- **Output:** Per-frame convex hull mask over clean skin regions

### 1.2 ROI Mean RGB Extraction
- **What it does:** Averages all pixel values inside the mask per frame
- **Output:** `rgb_signals` — `np.ndarray` of shape `(N, 3)` where N = number of valid frames

### 1.3 Ground Truth PPG Loading
- **Source:** UBFC-rPPG dataset — space-separated `.txt` file alongside the video
- **What it does:** Loads GT PPG waveform, resamples to match video frame count N
- **Output:** `ppg_signal` — `np.ndarray` of shape `(N,)`

### 1.4 Classical Baseline — CHROM + Bandpass + FFT
- **CHROM:** Chrominance-based signal extraction — takes ratios between RGB channels to cancel illumination drift
  ```
  Xs = 3R - 2G
  Ys = 1.5R + G - 1.5B
  α  = std(Xs) / std(Ys)
  rPPG_raw = Xs - α * Ys
  ```
- **Bandpass filter:** Butterworth filter (0.7–3 Hz) — removes everything outside plausible heart rate range (42–180 bpm)
- **FFT:** Dominant frequency in 0.7–3 Hz → heart rate in BPM
- **Output:** Estimated heart rate (scalar, BPM)

### 1.5 Heart Rate Estimation & Evaluation
- Compare estimated BPM vs. ground truth BPM from GT PPG
- Metrics: MAE (beats per minute), Pearson correlation on waveform

### 1.6 Visualization
Three stacked subplots:
1. Raw R, G, B channels over time
2. Extracted rPPG waveform (after CHROM)
3. Ground truth PPG waveform

---

## 🔵 Stage 2: Deep Learning Model

### 2.1 Dataset Windowing

**Why windowing:** The model needs fixed-length inputs. The full video is too long to feed at once — slice it into overlapping chunks.

**Approach:**
- Window size: **150 frames** (~5 seconds at 30fps)
- Stride: **75 frames** (50% overlap — doubles dataset size)
- Each window: input `(150, 3)` RGB → target `(150,)` GT PPG waveform segment

**Normalization (per window before training):**
- RGB: zero mean, unit std per channel
- PPG: zero mean, unit std

**Train/Val split:**
- 80% train, 20% validation
- Split on windows (not subjects — only 1 subject at this stage)

**Dataset class:** `rPPGDataset(rgb_signals, ppg_signal, window_size=150, stride=75)`
- Returns `(3, 150)` RGB tensor (channels first for Conv1d) and `(150,)` PPG tensor per sample

---

### 2.2 Model Architecture — `rPPGNet`

```
Input: (B, 3, T)        ← batch of windowed RGB signals, T=150 frames

        ↓
┌─────────────────────┐
│     SpatialCNN      │  3 × Conv1d blocks (3→16→32→64 channels)
│  (Feature Extractor)│  BatchNorm + ELU after each block
│                     │  Kernel size 3, same padding → preserves time length
└─────────────────────┘
        ↓ (B, 64, T)

┌─────────────────────┐
│  TemporalAttention  │  Self-attention over the time axis
│                     │  Q, K, V projections via 1×1 Conv1d
│                     │  Attention map: (B, T, T) softmax scores
│                     │  Learns to suppress motion/lighting corrupted frames
│                     │  Residual connection: output + input
└─────────────────────┘
        ↓ (B, 64, T)

┌─────────────────────┐
│  Decoder            │  1×1 Conv1d: 64 channels → 1 channel
└─────────────────────┘
        ↓ (B, 1, T)

        squeeze → (B, T)

Output: (B, T)          ← predicted rPPG waveform
```

**Parameter count:** ~50K–80K (deliberately lightweight — you have 1 subject, overfitting is a real risk with a large model)

---

### 2.3 Loss Function — Negative Pearson Correlation

**Why not MSE:** MSE cares about absolute values. You care about waveform *shape* — the periodic rise and fall matching the heartbeat. Two identical waveforms with a DC offset would have high MSE but perfect Pearson correlation.

```python
def pearson_loss(pred, target):
    # pred, target: (B, T)
    pred   = pred   - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num    = (pred * target).sum(dim=1)
    denom  = pred.norm(dim=1) * target.norm(dim=1) + 1e-8
    return -corr.mean()   # negate → minimize = maximize correlation
```

---

### 2.4 Training Loop

**Setup:**
- Optimizer: Adam, lr=1e-3
- Scheduler: ReduceLROnPlateau (patience=5) — halves LR if val loss plateaus
- Epochs: 50 (early stopping at patience=10 on val loss)
- Batch size: 16

**Per epoch:**
1. Forward pass → predicted waveform `(B, T)`
2. Compute Pearson loss vs. GT PPG window
3. Backward + optimizer step
4. Validation: run without gradients, log val loss and val Pearson r

**Logging:** Track train loss, val loss, val Pearson r per epoch → plot learning curves

**Checkpointing:** Save best model weights by val Pearson r (not val loss — they're equivalent here but Pearson r is more interpretable)

---

### 2.5 Heart Rate Extraction from Learned Waveform

Same pipeline as CHROM baseline — the model replaces the CHROM step only:
1. Run model on test windows → predicted rPPG waveform
2. Stitch overlapping windows back (average overlapping regions)
3. Bandpass filter (0.7–3 Hz)
4. FFT → dominant frequency → BPM

---

### 2.6 Evaluation — Model vs. Baseline

Run both CHROM and `rPPGNet` on the same held-out test windows:

| Metric | CHROM (Baseline) | rPPGNet (Learned) |
|--------|-----------------|-------------------|
| Heart Rate MAE (BPM) | — | — |
| Pearson r (waveform) | — | — |

**Plots:**
- Predicted waveform vs. GT PPG (CHROM vs. model side by side)
- Learning curves (train loss vs. val loss over epochs)
- FFT spectrum comparison (CHROM vs. model vs. GT)

---

## 🔴 Stage 3: Motion Robustness Ablation

### 3.1 Stress Test on Protocol 2 (Realistic)
- Download UBFC-rPPG Protocol 2 for the same subject (head motion + lighting variation)
- Run both CHROM and rPPGNet on it without retraining
- Compare MAE and Pearson r — expect both to degrade, model less so

### 3.2 Ablation: Disable Attention
- Create `rPPGNet_NoAttn` — identical architecture but remove `TemporalAttention` block
- Retrain from scratch on Protocol 1
- Compare on Protocol 2:

| Metric | CHROM | rPPGNet (No Attn) | rPPGNet (With Attn) |
|--------|-------|-------------------|---------------------|
| MAE (BPM) | — | — | — |
| Pearson r | — | — | — |

**This is your key contribution:** quantitatively showing that temporal attention recovers accuracy under motion — not just claiming it helps.

### 3.3 Frame-Level Attention Visualization
- Extract attention weights from `TemporalAttention` for a test window
- Plot attention weight per frame alongside head motion magnitude (optical flow or landmark displacement)
- Expect: attention goes down when motion goes up — visualizing the model "knowing" to ignore bad frames

---

## 📁 File Structure

```
rPPG_Project/
├── data/
│   └── UBFC-rPPG/
│       └── DATASET1/
│           └── subject1/
│               ├── vid-001.avi
│               └── ground_truth.txt
├── models/
│   └── face_landmarker.task
├── src/
│   ├── landmark_extractor.py   ✅ Done — MediaPipe ROI extraction
│   ├── signal_processing.py   ✅ Done — CHROM + bandpass + FFT
│   ├── gt_loader.py           ✅ Done — GT PPG loading + resampling
│   ├── evaluate.py            ✅ Done — MAE, Pearson r, plots
│   ├── dataset.py             🔵 Next — windowing + DataLoader
│   ├── model.py               🔵 Next — rPPGNet architecture
│   ├── train.py               🔵 Next — training loop
│   └── ablation.py            🔴 Final — robustness ablation
├── results/
│   ├── subject1_rgb.csv
│   ├── figures/
│   │   ├── raw_signal_plot.png
│   │   ├── waveform_comparison.png
│   │   ├── learning_curves.png
│   │   ├── fft_spectrum.png
│   │   └── attention_weights.png
│   └── metrics_summary.csv
├── notebooks/
│   ├── 01_baseline_demo.ipynb
│   └── 02_model_training.ipynb
├── README.md
└── requirements.txt
```

---

## 📦 Dependencies

```
torch>=2.0
numpy
scipy
opencv-python
mediapipe
matplotlib
pandas
```

---

## 📊 Course Outcome Mapping

| CO | How This Project Covers It |
|----|---------------------------|
| CO1 — Apply DL fundamentals | CNN, attention, backprop, Pearson loss, Adam optimizer |
| CO2 — DL in Python | Full PyTorch implementation: model, training loop, evaluation |
| CO3 — Signal analysis | CHROM, bandpass filter, FFT, HRV metrics on extracted waveform |
| CO4 — Image analysis | Spatiotemporal feature extraction from video frames + transfer learning backbone |

---

## 🗓 Remaining Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| Week 3 | `dataset.py` — windowing + DataLoader | Smoke-tested dataset class |
| Week 3 | `model.py` — rPPGNet architecture | Forward pass verified |
| Week 4 | `train.py` — training loop | Trained model checkpoint |
| Week 4 | Evaluation — model vs. CHROM | Comparison table + plots |
| Week 5 | Protocol 2 stress test | Robustness degradation results |
| Week 5 | Ablation — with vs. without attention | Ablation table + attention weight visualization |
| Week 6 | Final writeup + repo polish | Report + clean codebase |
