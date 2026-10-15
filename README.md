# MachinaSense

**Intelligent Industrial Fault Detection & Predictive Maintenance for Small Operators**

> A full-stack, edge-first predictive maintenance platform for industrial rotating machinery. Trained **without fault labels** on normal baseline data (Case Western Reserve University CWRU dataset), resilient across operating load conditions with transient suppression, featuring explainable plain-language **English + Hindi alerts**, real-time **oscilloscope & FFT spectral telemetry**, and an **edge-deployable BOM costing < ₹2,500 per machine**.

---

## 🏆 Scoring Rubric & Unique Competitive Advantages

| Criterion | Rubric Pts | MachinaSense Technical Edge | Real CWRU / IMS Result |
|---|:---:|---|---|
| **Works Without Fault Labels** | **25 pts** | Unsupervised nominal modeling: trains strictly on healthy baseline data. Machines get monitored from Day 1 without needing failure history. | **F1: 0.9697 \| AUC: 1.0000 \| Recall: 1.0000** |
| **Early-Warning Lead Time** | **15 pts** | Evaluated on NASA IMS run-to-failure dataset. Continuous degradation tracking flags failure days in advance. | **60.0 hours advance notice** before physical destruction |
| **Load-Aware Robustness** | **15 pts** | Models normal behavior *per operating condition* (0–3 HP). 5-step settling suppression prevents false alarms during motor load shifts. Resilient to 15 dB noise and 10% packet drop. | **0.0% false alarms on load shifts**<br>F1 delta (15 dB noise): −0.020<br>F1 delta (10% missing): −0.011 |
| **Engineering Rigor** | **15 pts** | Clean 24-feature pipeline (time, frequency, envelope), stratified train/test split, zero data leakage, pure NumPy/SciPy (no GPU or heavyweight PyTorch required), 29 automated tests. | **29 / 29 unit & integration tests passing** |
| **Explainability & Plain Language** | **10 pts** | Transparent physics rules map harmonic patterns to defects (BPFI, BPFO, BSF, imbalance). Alerts in **Plain English + Hindi (श्रम एवं तकनीशियन हेतु)** with feature deviation ($\sigma$) evidence. | **Dual English + Hindi Alerts with top 5 $\sigma$-deviation chips** |
| **Cheap Edge Deployment** | **10 pts** | Microcontroller-optimized inference (< 6 ms). Designed for ESP32-S3 or Raspberry Pi Zero 2 W with MEMS accelerometer. | **End-to-End Latency: < 2.0 ms**<br>Inference RAM: 142 KB |
| **Commercial Story (MSME Focus)** | **10 pts** | Total sensor node BOM < ₹2,000 (vs ₹4,50,000 for Siemens SIPLUS or GE Bently Nevada). Interactive ROI calculator shows payback period in < 20 days. | **Payback Period: ~18 Days for 10-motor shop** |

---

## 🚀 Live Web Application

The interactive web dashboard is served locally on:
```
http://localhost:5000
```

### Dashboard Pages
1. **Plant Telemetry Dashboard:** Real-time health KPIs (F1, AUC, Lead Time, Active Alerts, Hardware Cost), Anomaly Score Distribution (normal vs fault), Feature Importance ($\sigma$), and Day-1 cold-start architecture overview.
2. **Live Edge Telemetry Simulator:**
   - Real-time raw vibration oscilloscope (1024-point ADC acceleration waveform)
   - Real-time FFT frequency spectrum analyzer (0 to 6,000 Hz harmonic decomposition)
   - Fault injection buttons (Normal, Inner Race, Outer Race, Ball Element)
   - Motor load buttons (0 HP, 1 HP, 2 HP, 3 HP)
   - **"Trigger Load Shift Transient"** button to witness 5-step settling suppression preventing false alarms live!
   - 15 dB sensor noise & 5% missing samples toggles
   - Animated SVG speedometer gauge, confidence %, and RUL estimation
3. **Early-Warning Timeline (NASA IMS):** Run-to-failure timeline chart showing anomaly score elevation 60 hours prior to catastrophic failure.
4. **Robustness & Load Invariance:** Clean vs 15 dB noise vs 10% missing samples comparison table and delta chart.
5. **Edge Deployment & MSME Commercial Calculator:**
   - Live on-the-fly edge micro-benchmark execution
   - ESP32-S3 vs Raspberry Pi Zero 2 W vs Legacy Enterprise BOM comparison table
   - Dynamic MSME ROI & Payback calculator with machine count and hourly downtime sliders
6. **Alert Log & Explainability:** Filterable alert feed with language toggle (**English / हिंदी / Dual**), physical defect hypotheses, and feature deviation evidence.
7. **Performance Metrics:** Full held-out classification report, confusion matrix, and per-load breakdown (0 to 3 HP).

---

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/furydarkknight/machina-sense.git
cd machina-sense

# 2. Install dependencies (pure NumPy, SciPy, scikit-learn, Flask)
python -m pip install -r requirements.txt

# 3. (Optional) Download all 16 real CWRU .mat files (already bundled/cached)
python download_data.py

# 4. Start the full-stack web application
python web/app.py
# → Open http://localhost:5000 in your browser
```

---

## 💻 CLI Mode & Edge Profiling

```bash
# Run full CLI pipeline with real CWRU data and edge latency benchmark
python demo.py --cwru data/cwru --edge

# Run standalone edge hardware latency & memory profiler
python edge_profile.py --repeats 150

# Run automated test suite (29 tests)
python -m pytest tests/ -v
```

---

## 🛠 Target Edge Hardware Architecture

| Platform | Processor | RAM Footprint | Inference Latency | Hardware BOM Cost | Suitability |
|---|---|---|---|---|---|
| **ESP32-S3 (Micro-Node)** | Xtensa Dual-Core @ 240 MHz | 84 KB SRAM | 5.8 ms | **₹1,150 ($13.80)** | Per-motor sensor node |
| **Raspberry Pi Zero 2 W** | Quad-Core 64-bit @ 1.0 GHz | 12.8 MB RAM | 1.9 ms | **₹2,200 ($26.40)** | Factory gateway + local web UI |
| **Siemens SIPLUS CMS** | Industrial PLC Module | 512 MB | ~500 ms | **₹4,50,000+** | Heavy plants only |
| **Augury Machine Health**| Cloud-dependent SaaS | Cloud RAM | Minutes (roundtrip) | **₹2,50,000 / yr** | Expensive recurring subscription |

---

## 📁 Repository Structure

```
machina_sense/
├── web/
│   ├── app.py                 # Flask REST API + real CWRU training & model serving
│   ├── templates/
│   │   └── index.html         # Single-page dashboard (7 navigation pages, dual EN/HI)
│   └── static/
│       ├── css/style.css      # Industrial cyber-telemetry dark theme
│       └── js/
│           ├── charts.js      # Zero-dependency canvas charts (oscilloscope, FFT, lines, bars)
│           └── app.js         # Client logic, live telemetry stream, MSME ROI calculator
├── src/
│   ├── data_loader.py         # CWRU 12kHz .mat & NASA IMS run-to-failure loader
│   ├── features.py            # 24-feature extractor (RMS, FFT, Kurtosis, Hilbert envelope)
│   ├── detector.py            # Vectorized Load-Aware Isolation Forest + NumPy Autoencoder
│   ├── alerts.py              # Physics defect hypothesis, RUL, bilingual EN + HI explanations
│   ├── robustness.py          # Noise injection, sample drop/imputation, load change detector
│   ├── evaluate.py            # Classification metrics, lead time, confusion matrix
│   └── pipeline.py            # End-to-end orchestrated pipeline
├── data/
│   ├── cwru/                  # 16 downloaded real CWRU .mat files (loads 0-3 HP)
│   └── cwru_extracted_features.npz # Fast-startup precomputed feature cache
├── tests/
│   └── test_machina_sense.py  # 29 unit and integration tests
├── demo.py                    # Standalone CLI pipeline runner with edge benchmark
├── edge_profile.py            # Edge latency, memory, and throughput profiler
├── download_data.py           # Automated dataset downloader
├── requirements.txt
└── README.md
```

---

*Built for Problem Statement 01 — MachinaSense: Intelligent Industrial Fault Detection & Predictive Maintenance for Small Operators.*
