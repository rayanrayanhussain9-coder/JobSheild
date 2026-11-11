# JobShield — Dataset Assessment & Provenance Report

## 1. Primary Accepted Benchmark Dataset

### Employment Scam Aegean Dataset (EMSCAD) / Real or Fake Job Posting Prediction
- **Academic Citation:** Vidros, S., Kolias, C., Kambourakis, G., & Akoglu, L. (2017). *"Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset."* Future Internet, 9(1), 6. DOI: [10.3390/fi9010006](https://doi.org/10.3390/fi9010006).
- **Primary Source:** Laboratory of Information & Communication Systems Security (Info-Sec-Lab), Department of Information and Communication Systems Engineering, University of the Aegean, Samos, Greece.
- **Acquisition Mirror:** Hugging Face verified repository (`victor/real-or-fake-fake-jobposting-prediction`) mirroring the authentic 17,880-record CSV.
- **Access Date:** October 22, 2026.
- **Raw File Checksum (SHA-256):** `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60`
- **File Size:** 50,061,601 bytes (~47.7 MB).
- **Collection Period:** 2012 – 2014.
- **Collection Platform:** Real-life job advertisements posted on Workable SaaS recruitment platform.
- **Annotation Method:** Rigorous manual inspection by cybersecurity researchers; cross-checked against recruitment verification logs and reported scam campaigns.
- **Real vs. Synthetic:** 100% genuine real-world postings (no synthetic text injected into core benchmark).
- **Total Records:** 17,880.
- **Class Distribution:**
  - Legitimate (Class 0): 17,014 records (95.16%).
  - Fraudulent (Class 1): 866 records (4.84%).
  - Natural real-world severe class imbalance (~19.6 : 1).
- **Fields Available:** `job_id`, `title`, `location`, `department`, `salary_range`, `company_profile`, `description`, `requirements`, `benefits`, `telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`, `fraudulent`.
- **Licence & Permitted Use:** Open academic research dataset; non-commercial educational and benchmarking use.

### Selection Rationale
1. **Gold Standard Benchmark:** EMSCAD is the most extensively cited and scientifically validated public dataset for recruitment fraud.
2. **Realistic Class Imbalance:** Preserves the true, highly skewed prevalence of fraud without misleading artificial balance.
3. **Rich Natural Language Content:** Contains realistic multi-paragraph job descriptions, requirements, and benefits mirroring what users actually copy-paste into screening tools.
4. **Reproducibility:** Established academic benchmark allows direct comparison with published literature.

### Known Limitations
- **Temporal Shift:** Collected between 2012 and 2014. Contemporary scams involving modern terminology (e.g., Telegram recruiter handles, specific UPI IDs, contemporary remote crypto schemes) are less prevalent in the raw training corpus. This limitation is directly mitigated by JobShield's hybrid architecture via its Rule-Based Warning Engine.
- **Geographic Bias:** Predominantly North American, European, and global English postings from Workable; fewer vernacular Indian internship listings.
- **Company Anonymization:** URLs were tokenized (`#URL_...#`) and explicit employer phone numbers were sanitized during dataset release.

---

## 2. Assessed & Rejected Dataset Candidates

| Dataset Candidate | Source & Type | Records | Reason for Rejection |
|---|---|---|---|
| Synthetic LLM Job Scams | Prompt-generated fake ads from public Kaggle scratchpads | ~2,500 | **Rejected:** Synthetic text lacks authentic recruiter variance and injects LLM stylistic artifacts. Headline evaluation must not be polluted with synthetic training data. |
| Balanced 50:50 EMSCAD Subsets | Resampled derivative mirrors on GitHub / Hugging Face | 1,732 | **Rejected:** Artificially balanced subsets distort the true class prior (4.84% -> 50.0%) and generate wildly inaccurate probability calibrations for real-world deployment. |
| Scraped Unlabeled Portals (Indeed/LinkedIn 2024) | Raw web scrapes | Variable | **Rejected:** Unvalidated ground-truth labels. Labeling an ad legitimate simply because it appeared on a portal violates data integrity rules. |
| Social Media Phishing Messages | WhatsApp / SMS employment spam feeds | ~1,200 | **Rejected:** Short SMS-length phishing snippets (1-2 sentences) do not match the structure of full job advertisements. Handled via Rule Engine rather than ML training corpus. |

---

## 3. Data Acquisition & Verification Procedure

To download and verify the exact dataset file used by JobShield:

```bash
# 1. Download directly from verified mirror
curl -L -o data/raw/fake_job_postings.csv \
  "https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv"

# 2. Verify SHA-256 Checksum on macOS / Linux
shasum -a 256 data/raw/fake_job_postings.csv
# Expected Output: 59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60
```
