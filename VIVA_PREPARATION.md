# JobShield — Viva Preparation & Oral Defense Notes

## Quick Project Summary for the Examiner
> *"JobShield is an explainable job-advertisement fraud-screening application. It combines a Calibrated Linear Classifier trained on the academic EMSCAD benchmark dataset (17,880 real ads) with a separate deterministic rule engine that detects critical red flags such as upfront fees and wire transfer demands. The system provides exact mathematical feature attributions ($x_i \cdot w_i$), operates at sub-10ms CPU latency, and enforces strict group-aware deduplication to guarantee zero data leakage."*

---

## 10 Core Viva Questions & Model Answers

### Q1: Why did you build a hybrid system with both a rule engine and a machine learning model?
**Answer:**  
Machine learning and rule-based systems solve complementary problems:
- **Machine Learning:** Excels at recognizing diffuse, subtle statistical vocabulary patterns across thousands of words that humans might miss.
- **Rule Engine:** Captures non-negotiable, explicit red flags (e.g. *"candidates must pay a $150 registration fee"* or *"wire transfer required"*).  
If we blended these into an opaque single score, an ad with 95% polite corporate phrasing might dilute the fee demand and produce a low score. Keeping them separate guarantees that critical red flags are immediately surfaced and exposed to the user.

---

### Q2: How did you handle the severe 95:5 class imbalance in the dataset?
**Answer:**  
In the EMSCAD dataset, only 4.84% of postings are fraudulent. If a model simply predicts "legitimate" for every ad, it achieves 95.16% raw accuracy but catches zero fraud (0% recall).  
To address this:
1. We ignored raw accuracy and optimized for **fraud-class F1 score** and **Precision-Recall AUC (Average Precision)**.
2. We compared unweighted, balanced, and class-weighted candidates.
3. We performed **decision threshold tuning** on validation data, finding that an operating threshold of **0.38** (rather than default 0.50) achieved the optimal balance of high precision (90.2%) and high recall (77.7%) on the held-out test set.

---

### Q3: What is "template leakage" and how did you prevent it?
**Answer:**  
Recruiters frequently repost identical or near-identical job listings across different cities and dates. In EMSCAD, out of 17,880 rows, there were **2,494 duplicate rows** sharing the same text template.  
If we had used a naive random split, identical templates would have appeared in both the training set and the test set, allowing the model to simply memorize specific templates and reporting artificially inflated performance.  
To prevent this, we generated a SHA-256 hash of normalized text templates and used `StratifiedGroupKFold` so that **all duplicate copies of a template were strictly assigned to the same partition**. We mathematically verified that cross-partition group overlap is **zero**.

---

### Q4: Why did you choose a Calibrated Linear Classifier instead of Deep Learning (BERT / LLMs)?
**Answer:**  
For three sound engineering and scientific reasons:
1. **Explainability:** With a linear model, local feature attributions ($c_i = x_i \cdot w_i$) are mathematically exact and un-approximated. Post-hoc explainers like LIME or SHAP on deep networks are sampling approximations that can be inconsistent and slow.
2. **Computational Efficiency:** Our linear model executes inference in **5.7 milliseconds** on a standard CPU with zero GPU requirements, zero cloud dependencies, and zero inference costs.
3. **Competitive Performance:** Calibrated LinearSVC achieved an F1 of 83.5% and PR-AUC of 89.2% on the locked test set, which is competitive with published academic baselines on this dataset.

---

### Q5: How does your rule engine avoid false alarms when an employer states "We never charge recruitment fees"?
**Answer:**  
The rule engine does not perform naive single-word matching. It implements **contextual negation filtering**:
1. It splits text into individual sentences and clauses.
2. It screens each clause against negation and anti-scam patterns (e.g., *"never charge"*, *"do not ask for payment"*, *"scam alert"*, *"tuition reimbursement"*).
3. If an explicit negation precedes or surrounds the fee keyword, the rule warning is suppressed. We wrote explicit automated tests verifying that legitimate disclaimers and employee tuition benefits do not trigger false alarms.

---

### Q6: How are your local feature explanations calculated?
**Answer:**  
Because our classification pipeline uses TF-IDF and a linear decision function, the pre-activation logit is:
$$\text{Logit} = \text{Intercept} + \sum_{i=1}^V x_i \cdot w_i$$
Where $x_i$ is the TF-IDF weight of word $i$, and $w_i$ is the learned model coefficient.  
For any input advertisement, we compute $c_i = x_i \cdot w_i$ for all non-zero terms in the text:
- Terms with $c_i > 0$ push the logit toward suspicion (e.g. `data entry`, `wire transfer`, `earn`).
- Terms with $c_i < 0$ push the logit toward legitimacy (e.g. `python`, `pto`, `mentorship`, `software`).  
The user sees the exact terms and weights that drove the model's decision.

---

### Q7: Why did you lock the final test set?
**Answer:**  
To prevent **p-hacking and evaluation leakage**. If an engineer inspects test set errors and modifies preprocessing, thresholds, or hyperparameters based on test feedback, the test set is no longer an unbiased measure of generalization.  
In JobShield, the final test set (2,682 records) was locked prior to experiment sweeps. Model selection and threshold tuning were conducted strictly on the validation partition, and the winning model was evaluated only once on the untouched test partition.

---

### Q8: What does the "Model Suspicion Score" actually mean?
**Answer:**  
It is a **calibrated probability** reflecting how closely the input advertisement's vocabulary resembles the historical fraudulent advertisements in our training dataset.  
We intentionally do NOT call it a "Trust Score" or "Real-World Fraud Probability" because text screening cannot verify whether a physical company exists or whether a recruiter's email domain has been spoofed. Honest labeling prevents users from developing a false sense of security.

---

### Q9: What happens if a user inputs empty or nonsensical text?
**Answer:**  
JobShield implements an **Input Adequacy Guard** before any inference occurs. If the input is empty, under 40 characters, under 8 words, over 30,000 characters, repetitive spam (low vocabulary diversity), or symbol-heavy gibberish, the application halts with an `INADEQUATE_INPUT` status and refuses to fabricate a neutral or misleading model score.

---

### Q10: What are the main limitations of this system and what would you do next?
**Answer:**  
1. **Limitation:** Text screening alone cannot catch sophisticated scams that copy-paste genuine job descriptions from legitimate companies and redirect applicants to external messaging apps (e.g. Telegram/WhatsApp).  
2. **Next Steps:**
   - Integrate domain authentication checks (verifying if the contact email domain matches official MX/DNS records).
   - Expand the dataset to include multilingual Indian regional employment scam patterns.
   - Implement active optical character recognition (OCR) to screen screenshot images of advertisements.
