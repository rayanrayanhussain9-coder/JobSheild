# JobShield — Experiment & Activity Log

## Project Summary
- **Target**: Explainable Job-Advertisement Fraud Screening
- **Environment**: macOS Apple Silicon (Darwin 27.0.0, arm64), Python 3.14.5 (.venv in project directory)
- **Primary Dependencies**: scikit-learn 1.9.1, pandas 3.0.6, streamlit 1.66.1, plotly 7.1.0, joblib 1.6.0, pytest 9.1.1
- **Dataset**: University of the Aegean Employment Scam Aegean Dataset (EMSCAD) / Real or Fake Job Postings
  - File: `data/raw/fake_job_postings.csv`
  - Size: 50,061,601 bytes
  - SHA-256: `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`
  - Total records: 17,880 (17,014 Legitimate, 866 Fraudulent ~ 4.84%)
  - Unique Text Templates: 15,749 (2,494 duplicate rows identified and group-isolated)

---

## Partitioning Manifest (Group-Stratified Split)

| Split | Total Records | Legitimate (0) | Fraudulent (1) | Fraud Rate (%) | Unique Template Groups | Cross-Split Overlap |
|---|---|---|---|---|---|---|
| **Train** | 12,516 | 11,910 | 606 | 4.84% | 11,058 | 0 (Strictly isolated) |
| **Validation** | 2,682 | 2,552 | 130 | 4.85% | 2,410 | 0 (Strictly isolated) |
| **Final Test** | 2,682 | 2,552 | 130 | 4.85% | 2,411 | 0 (Locked & untouched) |
| **Total** | 17,880 | 17,014 | 866 | 4.84% | 15,749 | **0 (Leakage-Free)** |

---

## Experiment Registry (Max Budget: 12 candidate configurations)

| Experiment ID | Model Description | Tuned Val Threshold | Val Precision | Val Recall | Val F1 Score | Val PR-AUC | Val ROC-AUC | Outcome / Selection Status |
|---|---|---|---|---|---|---|---|---|
| **EXP-00** | Majority Class Baseline (Always predicts 0) | 0.50 | 0.0000 | 0.0000 | 0.0000 | 0.0485 | 0.5000 | Reference baseline (0 recall) |
| **EXP-01** | Word TF-IDF (1-2 gram) + Logistic Regression (Unweighted, C=1.0) | 0.19 | 0.9167 | 0.6769 | 0.7788 | 0.8448 | 0.9634 | High precision but lower recall |
| **EXP-02** | Word TF-IDF (1-2 gram) + Balanced Logistic Regression (C=1.0) | 0.69 | 0.9159 | 0.7538 | 0.8270 | 0.8741 | 0.9672 | Balanced weights significantly boost recall |
| **EXP-03** | Word TF-IDF (1-2 gram) + Weighted Logistic Regression (class_weight 3.5x, C=2.0) | 0.44 | 0.9333 | 0.7538 | 0.8340 | 0.8804 | 0.9681 | Controlled weight boost improves F1 |
| **EXP-04** | **Word TF-IDF (1-2 gram) + Calibrated LinearSVC (Platt Sigmoid Calibration, cv=3)** | **0.38** | **0.9043** | **0.8000** | **0.8490** | **0.8846** | **0.9683** | **WINNER — Highest validation F1 & PR-AUC** |
| **EXP-05** | Word + Sublinear TF-IDF + Logistic Regression (C=3.0, weight 2.5x) | 0.34 | 0.9174 | 0.7692 | 0.8368 | 0.8864 | 0.9664 | Competitive, but slightly lower F1 than EXP-04 |

---

## Final Locked Test Evaluation (Evaluated ONCE on Untouched Test Set)

- **Selected Candidate**: `tfidf_linearsvc_calibrated` (EXP-04)
- **Frozen Operating Threshold**: 0.38
- **Test Precision**: **0.9018** (90.18%)
- **Test Recall**: **0.7769** (77.69%)
- **Test F1 Score**: **0.8347** (83.47%)
- **Test PR-AUC (Average Precision)**: **0.8916**
- **Test ROC-AUC**: **0.9865**
- **Test Confusion Matrix**:
  - True Negatives (TN): 2,541 (Legitimate correctly identified)
  - False Positives (FP): 11 (Legitimate incorrectly flagged as suspicious)
  - False Negatives (FN): 29 (Fraudulent ads missed by statistical model)
  - True Positives (TP): 101 (Fraudulent ads correctly caught)

---

## Inference Latency Benchmark (Measured over 50 Warm Runs)
- **Platform**: Apple Silicon M-series (Darwin 27.0.0, arm64)
- **Input Text Size**: 885 characters
- **Min Latency**: 5.55 ms
- **Median Latency**: **5.75 ms**
- **p95 Latency**: **6.37 ms**
- **p99 Latency**: 6.51 ms
- **Max Latency**: 6.52 ms
- **Target**: < 2,000 ms (Met with substantial margin: 314x faster than target).
