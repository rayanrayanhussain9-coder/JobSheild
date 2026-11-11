# 🛡️ JobShield — Explainable Job-Advertisement Fraud Screening

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Pytest-30%20Passed-brightgreen.svg)]()
[![Inference Latency](https://img.shields.io/badge/p95%20Latency-6.4ms-success.svg)]()

> **AI Applications Lab Project (First-Year B.Tech)**  
> A functionality-first, explainable fraud-screening tool that combines a calibrated linear classification model with a deterministic rule engine to screen job advertisements for recruitment scams, upfront payment requests, and suspicious language patterns.

---

## 🌟 Key Highlights

1. **Dual-Engine Detection:**
   - **Deterministic Rule Engine:** Catches explicit red flags (registration fees, training deposits, cryptocurrency demands, sensitive data harvesting) while contextually suppressing false alarms (e.g. *"We never charge recruitment fees"*).
   - **Explainable ML Pipeline:** Calibrated Linear Support Vector Classifier with n-gram TF-IDF, tuned on validation precision-recall trade-offs (F1: **83.5%**, Precision: **90.2%**, PR-AUC: **89.2%** on locked final test set).
2. **Mathematically Faithful Explanations:**
   - Exact local linear feature attributions ($x_i \cdot w_i$).
   - Explicitly shows which words pushed the ad toward or away from the suspicious class, accounting for the intercept base rate.
3. **Strict Data Leakage Prevention:**
   - Group-stratified partitioning ensures that 2,494 duplicate text templates never cross train, validation, or test boundaries. Overlap is verified **0**.
4. **Honest Scoring & Input Adequacy:**
   - Gating rejects empty, overly brief, or gibberish input without fabricating scores.
   - Scores are honestly labelled as calibrated historical pattern similarities, not "universal safety guarantees."
5. **Interactive Streamlit Web Interface:**
   - Multi-tab application supporting single-ad screening, side-by-side job comparison, live performance metrics & confusion matrix, and downloadable sanitized HTML/JSON audit reports.
6. **Sub-10ms CPU Performance:**
   - Median warm inference latency: **5.75 ms** (p95: **6.37 ms**), running 100% locally on CPU without paid APIs or cloud dependencies.

---

## 📁 Repository Structure

```
jobshield/
├── data/
│   ├── raw/
│   │   └── fake_job_postings.csv           # Authentic raw EMSCAD benchmark dataset (17,880 rows)
│   └── processed/
│       ├── train.csv                       # 70% Train partition (12,516 rows)
│       ├── val.csv                         # 15% Validation partition (2,682 rows)
│       ├── test.csv                        # 15% Locked Final Test partition (2,682 rows)
│       ├── split_manifest.json             # Exact counts, seeds, and leakage-free proofs
│       └── data_quality_report.json        # Missing values, duplicate rates, class balance
├── artifacts/
│   ├── model.joblib                        # Trained serialized pipeline
│   ├── metadata.json                       # Hyperparameters, threshold (0.38), feature counts
│   ├── evaluation_metrics.json             # Val vs Test metrics (Precision, Recall, F1, PR-AUC)
│   ├── baseline_metrics.json               # Comparison across all 6 candidate experiments
│   ├── confusion_matrix.json               # TN, FP, FN, TP breakdowns
│   └── calibration_curve.json              # Empirical vs predicted probability points
├── src/
│   ├── __init__.py
│   ├── config.py                           # Paths, limits, threshold constants
│   ├── text_cleaner.py                     # HTML stripping, normalization, adequacy validation
│   ├── data_processor.py                   # Schema verification, template deduplication, splitting
│   ├── rules_engine.py                     # Deterministic rule engine with negation handling
│   ├── explainer.py                        # Exact linear feature attribution (x_i · w_i)
│   ├── predictor.py                        # End-to-end inference orchestrator & disagreement resolver
│   ├── report_generator.py                 # Sanitized HTML & JSON report exporter
│   └── train.py                            # Candidate training, threshold tuning & artifact generation
├── tests/
│   ├── __init__.py
│   ├── test_data_processor.py              # Schema, label, and split isolation tests
│   ├── test_rules_engine.py                # Payment, disclaimer, benefit, and urgency tests
│   ├── test_predictor_and_explainer.py     # Input adequacy, math fidelity, corrupt artifact tests
│   ├── test_challenge_set.py               # Documented challenge cases & edge cases
│   ├── test_streamlit_app.py               # Streamlit AppTest UI interaction tests
│   └── test_latency.py                     # 50-iteration warm latency benchmark (<2000ms target)
├── app.py                                  # Multi-tab Streamlit web application
├── requirements.txt                        # Pinned dependencies
├── DATASET_ASSESSMENT.md                   # Provenance, licensing, and candidate analysis
├── MODEL_CARD.md                           # Formal model card documentation
├── EXPERIMENT_LOG.md                       # Comprehensive experiment and activity register
├── ARCHITECTURE.md                         # Detailed software architecture & design choices
└── VIVA_PREPARATION.md                     # 10 essential viva defense questions & answers
```

---

## 🚀 Quickstart & Setup Instructions

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14 on macOS arm64).
- Terminal access (bash / zsh).

### 2. Virtual Environment Setup
```bash
# Clone or navigate to the project directory
cd /Users/rayan/.gemini/antigravity/scratch/jobshield

# Create isolated virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Reproducible Data Preparation & Model Training

### 1. Ingest & Prepare Data (Group-Aware Deduplication)
```bash
PYTHONPATH=. python src/data_processor.py
```
*Validates schema, detects duplicate text templates, enforces group isolation, and exports `data/processed/split_manifest.json`.*

### 2. Train Candidates & Evaluate Winner on Locked Test Set
```bash
PYTHONPATH=. python src/train.py
```
*Trains 6 candidate configurations, sweeps thresholds on validation data, locks the best model, evaluates once on the untouched test partition, and saves all evaluation artifacts to `artifacts/`.*

---

## 🧪 Running Automated Tests

Run the complete 30-test suite with verbose output and latency benchmarking:
```bash
PYTHONPATH=. pytest -v -s tests/
```
*Executes unit tests, integration tests, challenge sets, AppTest UI checks, and latency benchmarks in under 2 seconds.*

---

## 🖥️ Launching the Application

Start the interactive Streamlit server:
```bash
PYTHONPATH=. streamlit run app.py
```
Once started, open your browser at `http://localhost:8501`.

### Main Application Tabs
1. **🔍 Analyze Job:** Paste a job title, employer context, and description. Click **Screen Advertisement** to inspect the model's suspicion score, triggered rules, local feature attributions, and download full HTML/JSON reports.
2. **⚖️ Compare Jobs:** Screen two opportunities side-by-side to compare risk profiles, rule flags, and linguistic characteristics.
3. **📊 Model Performance:** Explore live validation vs locked test metrics, interactive confusion matrix heatmaps, candidate comparisons, and calibration plots loaded directly from real evaluation artifacts.
4. **📖 About & Methodology:** Review dataset provenance, technical explanations, and student viva notes.

---

## 📊 Summary of Final Measured Results

- **Trained Pipeline:** CalibratedLinearSVC with sublinear word n-gram TF-IDF (1–2)
- **Decision Threshold:** `0.38` (tuned strictly on validation data)
- **Test Precision (Fraud Class):** `90.18%` (Only 11 false alarms out of 2,552 legitimate ads)
- **Test Recall (Fraud Class):** `77.69%` (Detected 101 of 130 genuine scams)
- **Test F1 Score (Fraud Class):** `83.47%`
- **Test PR-AUC (Average Precision):** `89.16%`
- **Median Inference Latency:** `5.75 ms` (p95: `6.37 ms`)
- **Automated Tests:** 30 passed, 0 failed.

---

## 🛡️ Product Boundary & Disclaimer
JobShield analyzes advertisement text for historical fraud characteristics and known deceptive red flags. It does **not** verify corporate registration numbers, validate recruiter DNS records, or guarantee the absolute safety or legitimacy of an employer. Users must always conduct independent diligence before accepting offers or sharing personal data.
