# JobShield Model Card

## 1. Model Details

- **Model Name:** JobShield Linear Fraud Screener
- **Version:** `1.0.0`
- **Model Type:** Calibrated Linear Support Vector Classifier (`CalibratedClassifierCV` wrapping `LinearSVC`) with Sublinear Word N-gram TF-IDF (`TfidfVectorizer(ngram_range=(1, 2))`)
- **Release Date:** October 22, 2026
- **Developer:** AI Applications Lab Project (First-Year B.Tech)
- **Frameworks:** Python 3.14.5, Scikit-learn 1.9.1, Joblib 1.6.0
- **Serialization:** Local Joblib binary artifact (`artifacts/model.joblib`)
- **Associated Config & Metadata:** `artifacts/metadata.json`

---

## 2. Intended Use & Target Users

### Intended Applications
- Screening job advertisement text pasted by students, fresh graduates, and job seekers.
- Identifying high-confidence deterministic red flags (upfront training/registration fees, wire transfers, unverified placement guarantees).
- Providing transparent, mathematical feature attributions ($x_i \cdot w_i$) explaining why particular words contribute toward or away from a suspicious score.
- Side-by-side comparison of ambiguous opportunities to highlight differing risk profiles.

### Explicitly Out-of-Scope Uses
- Automated recruiter bans or automated employment rejections without human review.
- Employer identity verification or corporate authenticity guarantees.
- Replacement for background checks, legal investigations, or official regulatory reporting.
- Guaranteeing that a low-risk advertisement is safe to accept.

---

## 3. Training & Evaluation Data

- **Benchmark Dataset:** Employment Scam Aegean Dataset (EMSCAD).
- **Source:** Vidros et al., *Future Internet*, 2017 (University of the Aegean).
- **Total Records:** 17,880 verified real-world recruitment postings.
- **Natural Class Distribution:** 17,014 Legitimate (95.16%), 866 Fraudulent (4.84%).
- **Data Hygiene & Anti-Leakage:**
  - Group-stratified 70% / 15% / 15% partitioning.
  - Template hashing grouped 2,494 duplicate postings strictly into the same partition.
  - Zero cross-partition template leakage.
  - TF-IDF vectorizer and calibrated scaling fitted strictly on training data.
  - Final test set locked until model hyperparameters and decision thresholds were frozen.

---

## 4. Quantitative Performance

### Metrics on Locked Final Test Set (Threshold = 0.38)

| Metric | Measured Value | Notes |
|---|---|---|
| **Precision (Fraud Class)** | **90.18%** | Only 11 legitimate ads falsely flagged out of 2,552 |
| **Recall (Fraud Class)** | **77.69%** | Caught 101 of 130 genuine fraud postings |
| **F1 Score (Fraud Class)** | **83.47%** | Optimized on validation precision-recall trade-off |
| **Average Precision (PR-AUC)** | **89.16%** | High area under the precision-recall curve |
| **ROC-AUC** | **98.65%** | Strong separation across all thresholds |
| **Warm Inference Latency (p95)**| **6.37 ms** | Sub-10ms local CPU execution |

### Confusion Matrix on Test Set
- **True Negatives (TN):** 2,541
- **False Positives (FP):** 11
- **False Negatives (FN):** 29
- **True Positives (TP):** 101

---

## 5. Explainability & Mathematical Fidelity

- The model uses exact linear feature contributions: $\text{contribution}_i = x_i \cdot w_i$, where $x_i$ is the TF-IDF weight of word/n-gram $i$, and $w_i$ is the model's learned coefficient.
- Features with positive contributions push the prediction toward the suspicious class.
- Features with negative contributions push the prediction toward the legitimate class.
- The base rate (intercept: $-0.6703$) is explicitly factored into the calculation.
- Explanations are mathematically exact and reflect historical statistical associations, not causal claims.

---

## 6. Risks, Limitations & Ethical Mitigations

1. **Adversarial Scammers:** Sophisticated scammers can rephrase listings to omit suspicious keywords (e.g. omitting "fee" and redirecting to Telegram).  
   *Mitigation:* JobShield pairs statistical modeling with deterministic rules and clearly flags missing employer information.
2. **False Negatives:** A low score does not guarantee legitimacy.  
   *Mitigation:* JobShield explicitly labels low scores as "Low Risk Text Pattern" rather than "Safe Employer" and includes persistent disclaimer banners.
3. **Class Imbalance:** Fraud represents <5% of data. Raw accuracy (95%) is misleading.  
   *Mitigation:* We optimized specifically for fraud-class F1 and Average Precision, setting the threshold on validation data.
