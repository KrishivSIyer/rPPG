# Contactless Vital Sign Monitoring from Facial Video (rPPG)

## Project Title
**Remote Photoplethysmography (rPPG): Deep Learning for Contactless Heart Rate and HRV Estimation**

---

## 1. Problem Statement & Motivation

Current vital sign monitoring (heart rate, respiratory rate, heart-rate variability) requires wearable sensors or direct contact. This limits deployment in telehealth, continuous ICU monitoring, and driver-drowsiness detection scenarios where contact-based sensing is impractical or intrusive.

**Research Gap:** Remote photoplethysmography (rPPG) — extracting pulse from facial video alone — is a solved problem in classical signal processing (CHROM, POS methods), but deep learning approaches that are robust to motion and lighting variation remain under-developed. Most published models break when the subject moves their head or lighting changes abruptly.

**Our Contribution:** Implement and compare a spatiotemporal attention-based CNN against a classical baseline, with a dedicated ablation study on **motion robustness** — the key differentiator between lab-demo and deployable systems.

---

## 2. Technical Approach

### 2.1 Pipeline Architecture

```
Video Input
    ↓
[Face Detection & Landmark Extraction] — MediaPipe Face Mesh (pretrained)
    ↓
[ROI Spatial Extraction] — Forehead/cheek patches per frame
    ↓
[Signal Extraction & Baseline] 
    ├─ Stage 1 (Mid-sem review): Classical method (CHROM or green-channel + bandpass + FFT)
    └─ Stage 2 (Final): CNN + spatiotemporal attention network
    ↓
[Post-processing]
    ├─ Bandpass filtering (0.7–3 Hz for heart rate)
    ├─ FFT for dominant frequency → heart rate (bpm)
    └─ Peak detection for HRV metrics
    ↓
[Evaluation]
    ├─ Heart rate MAE vs. ground-truth PPG
    ├─ Waveform correlation (Pearson r)
    └─ Motion-robustness ablation (performance degradation under head movement)
```

### 2.2 Model Architecture (Final Stage)

- **Spatial branch:** CNN backbone (lightweight ResNet or MobileNet, pretrained on face recognition tasks) extracts per-frame features from ROI patches
- **Temporal branch:** LSTM or temporal attention layers (TS-CAN / PhysFormer style) to isolate periodic pulse signal from motion/noise
- **Fusion:** Spatial + temporal features combined with learned attention weights that suppress motion-corrupted frames
- **Output:** Raw rPPG waveform (1D signal) for downstream FFT and heart-rate computation

---

## 3. Data & Methodology

### 3.1 Datasets

**Primary:** UBFC-rPPG (public benchmark)
- **Protocol 1** (Mid-sem): Simple, frontal, minimal motion, controlled lighting — for baseline validation
- **Protocol 2** (Post-review): Realistic motion and lighting variation — for robustness ablation

**Initial Scope:** 1 subject (mid-sem review), scale to 3–5 subjects for final submission

**Ground Truth:** PPG waveforms sampled at 25 Hz (provided with dataset) and ECG-matched reference

### 3.2 Methodology

**Stage 1 (Mid-Sem Review, Due in 2 days):**
1. Load UBFC Protocol 1 video + ground-truth PPG signal
2. Extract face landmarks (MediaPipe) → ROI patches
3. Compute raw RGB time series (averaged across patch pixels)
4. Apply classical baseline (CHROM: chrominance-based signal extraction)
5. Bandpass filter (0.7–3 Hz) → FFT → estimate heart rate
6. Compute MAE and correlation vs. ground truth
7. **Deliverable:** Working pipeline demo + quantitative baseline result

**Stage 2 (Post-Review, 4 weeks):**
1. Train CNN + attention model on UBFC Protocol 1 (80/20 train/val split)
2. Compare learned model against CHROM baseline (same test set)
3. Stress-test both models on UBFC Protocol 2 (head motion + lighting)
4. Measure performance degradation: baseline vs. motion-aware attention
5. Ablation: disable attention → show accuracy drop
6. **Deliverable:** Comparison plots, robustness analysis, trained model checkpoints

---

## 4. Expected Outputs

### Mid-Sem Review (This Week)
- ✅ Data pipeline notebook (load video → extract ROI → plot signal)
- ✅ Classical baseline implementation (CHROM) with one quantitative result
- ✅ Pipeline diagram (visual) + architecture slide
- ✅ Clear "next steps" roadmap showing deep learning model

### Final Submission (6 weeks total)
- ✅ Trained CNN + attention model
- ✅ Comparison table: CHROM vs. learned model (Protocol 1 + Protocol 2)
- ✅ Motion-robustness ablation study with plots
- ✅ Waveform examples (side-by-side extracted vs. ground truth)
- ✅ Discussion of failure modes and limitations
- ✅ Code repo with reproducible pipeline

---

## 5. File Structure & Deliverables

```
rPPG_Project/
├── data/
│   └── UBFC-rPPG/
│       └── subject_01/
│           ├── vid.avi (or .mp4)
│           └── ppg.txt (ground truth)
├── notebooks/
│   ├── 01_data_loading.ipynb          [Mid-sem]
│   ├── 02_classical_baseline.ipynb    [Mid-sem]
│   ├── 03_deep_model_training.ipynb   [Post-review]
│   └── 04_robustness_ablation.ipynb   [Final]
├── src/
│   ├── pipeline.py          (ROI extraction, signal processing)
│   ├── models.py            (CNN + attention architecture)
│   ├── evaluate.py          (MAE, correlation, ablation logic)
│   └── utils.py             (plotting, data loaders)
├── results/
│   ├── baseline_results.csv
│   ├── learned_model_results.csv
│   └── figures/
│       ├── waveform_comparison.png
│       ├── robustness_plot.png
│       └── architecture_diagram.png
├── README.md                (Overview + usage instructions)
└── requirements.txt         (Dependencies: opencv, mediapipe, torch, scipy, numpy)
```

---

## 6. Timeline & Milestones

| Phase | Duration | Deliverable | Status |
|-------|----------|-------------|--------|
| **Stage 1: Baseline** | 2 days | Data loading + CHROM + 1 quantitative result | 🔴 In Progress |
| **Mid-Sem Review** | — | Demo + slides + roadmap | 🟡 Scheduled |
| **Stage 2: Deep Learning** | 3 weeks | Trained model + comparison vs. baseline | 🔵 Planned |
| **Stage 3: Ablation** | 1 week | Motion-robustness analysis + writeup | 🔵 Planned |
| **Final Submission** | 6 weeks total | Complete codebase + report | 🔵 Planned |

---

## 7. Course Alignment

This project directly addresses **all four course outcomes:**

- **CO1 (Apply fundamentals of DL):** Understand CNN + LSTM/attention principles
- **CO2 (DL algorithms in code):** Implement CNN + temporal attention in PyTorch
- **CO3 (Signal analysis):** Extract + analyze time-series PPG waveform (FFT, bandpass filtering, HRV metrics)
- **CO4 (Image analysis):** Spatiotemporal feature extraction from video frames + transfer learning (pretrained face recognition backbone)

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| UBFC download slow | Use local webcam recording as fallback (same pipeline logic) |
| Model doesn't converge | Classical baseline is fully working and defensible as a standalone result |
| Limited compute time | Use lightweight models (MobileNet backbone, smaller attention heads) |
| Motion robustness is hard | Explicitly frame it as an **open research problem**, show partial improvements via ablation |

---

## 9. References & Baselines

- **CHROM (2013):** Wang et al., "Algorithmic Principles of Remote Photoplethysmographic Imaging"
- **TS-CAN (2019):** Lu et al., "Unsupervised Learning Relationship for Efficient Robust Skin Tone Detection"
- **PhysFormer (2022):** Zhu et al., "PhysFormer: A Physics-Guided Transformer for Remote Sensing Change Detection" (spatiotemporal attention principles)
- **UBFC-rPPG Dataset:** Bobbia et al., "Unsupervised skin tissue segmentation for remote photoplethysmography"

---

## Contact & Questions

All code, data logs, and results will be tracked in a shared repo with clear commit messages and reproducible notebooks for each stage.

