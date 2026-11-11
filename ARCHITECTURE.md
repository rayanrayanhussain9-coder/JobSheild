# JobShield — System Architecture & Design Documentation

## 1. High-Level System Architecture

JobShield employs a **hybrid dual-engine architecture** that decouples deterministic rule screening from statistical machine-learning pattern recognition.

```
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|                      Streamlit Multi-Tab Web Application                          |
|         [🔍 Analyze Job]   [⚖️ Compare Jobs]   [📊 Performance]   [📖 About]      |
+------------------------------------------+----------------------------------------+
                                           |
                                           | JobAdvertisementInput
                                           v
+-----------------------------------------------------------------------------------+
|                             INPUT ADEQUACY GATEWAY                                |
|  - Length Validator (40 to 30,000 characters)                                     |
|  - Vocabulary Diversity Analyzer (Catches repetitive spam)                         |
|  - Character Distribution Check (Rejects non-alphanumeric gibberish)              |
|  * If inadequate: stops immediately without fabricating misleading neutral scores |
+------------------------------------------+----------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+----------------------------------------+  +---------------------------------------+
|       RULE-BASED WARNING ENGINE        |  |        EXPLAINABLE ML PIPELINE        |
| - Upfront fee & deposit detection      |  | - HTML unescaping & normalization     |
| - Unconventional payment methods       |  | - Word & subword TF-IDF n-grams (1-2) |
|   (cryptocurrency, wire transfers)     |  | - Calibrated Linear Classifier        |
| - Unrealistic hiring guarantees        |  | - Tuned Decision Threshold (0.38)     |
| - Sensitive credential harvesting      |  | - Platt Sigmoid Probability           |
| - Contextual negation & disclaimer     |  |   Calibration                         |
|   guard (e.g. "we never charge fees")  |  |                                       |
+-------------------+--------------------+  +-------------------+-------------------+
                    |                                           |
                    | RuleWarnings                              | Prediction & Logits
                    +----------------------+--------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                              SYNTHESIS & EXPLAINER                                |
|  - Exact Linear Feature Attribution: c_i = x_i · w_i                              |
|  - Partitions vocabulary into Suspect vs Legitimate contributors                   |
|  - Missing Information Profiler (Omitted salary, employer, or prerequisites)      |
|  - Disagreement Resolver (Exposes conflicts between Rule Red Flags and ML scores) |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                              REPORTING & EXPORT                                   |
|  - Interactive Plotly Feature Bar Charts & Confusion Heatmaps                     |
|  - Exportable Self-Contained Sanitized HTML Audit Reports                         |
|  - Machine-Readable JSON Export Files                                             |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Breakdown

| Module | Location | Responsibility |
|---|---|---|
| `config.py` | `src/config.py` | Centralizes file paths, threshold defaults, length boundaries, and random seeds. |
| `text_cleaner.py` | `src/text_cleaner.py` | Handles HTML stripping, spacing normalization, and input adequacy validation. |
| `data_processor.py`| `src/data_processor.py` | Validates raw CSV schemas, clusters duplicate templates via SHA-256 hashes, executes leakage-free group-stratified splits, and writes data quality reports. |
| `rules_engine.py` | `src/rules_engine.py` | Regex-driven deterministic screening for upfront payments, wire transfers, and identity demands with negation and reimbursement discernment. |
| `explainer.py` | `src/explainer.py` | Computes exact local linear feature attributions ($x_i \cdot w_i$), calculates raw logits, and surfaces top positive and negative vocabulary features. |
| `predictor.py` | `src/predictor.py` | Orchestrates the end-to-end pipeline, evaluates missing information, handles borderlines, and detects rule-model disagreements. |
| `report_generator.py`| `src/report_generator.py` | Exports sanitised, self-contained HTML reports and machine-readable JSON files. |
| `train.py` | `src/train.py` | Trains candidate pipelines, sweeps thresholds on validation data, evaluates the winner on the locked test set, and writes binary/JSON artifacts. |
| `app.py` | `app.py` | Interactive Streamlit multi-tab user interface with side-by-side comparison and evaluation dashboards. |

---

## 3. Key Design Decisions

### A. Dual Engine vs Single End-to-End Model
- **Problem:** If a recruiter demands a $100 registration fee via wire transfer in an otherwise standard job ad, a pure statistical bag-of-words model might assign a low suspicion score because 95% of the words look corporate.
- **Solution:** The separate rule engine checks for non-negotiable red flags. If a red flag is found, JobShield highlights it immediately, even if the general statistical score is low.

### B. Group-Stratified Template Deduplication
- **Problem:** In job board datasets, duplicate recruiters repost the exact same ad dozens of times. Random splitting leaks identical templates between training and testing sets, producing artificially inflated metrics.
- **Solution:** We compute a SHA-256 hash of normalized text templates and constrain all duplicate copies to the exact same partition (`StratifiedGroupKFold`). Overlap between train, validation, and test is strictly 0.

### C. Linear Classifiers vs Black-Box Deep Learning
- **Problem:** Complex models (like LSTMs or large Transformers) require heavy compute and rely on post-hoc surrogate explainers (like LIME or SHAP) which are approximate, slow, and can produce contradictory explanations.
- **Solution:** A well-tuned LinearSVC with Platt calibration achieves an F1 of 83.5% and PR-AUC of 89.2% on the locked test set, runs in **5.7 milliseconds** on CPU, and enables **exact, mathematically faithful local feature attributions** ($x_i \cdot w_i$).
