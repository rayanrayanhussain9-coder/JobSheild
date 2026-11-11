"""
JobShield Configuration Module
Defines paths, random seeds, input constraints, and default decision thresholds.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "fake_job_postings.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "split_manifest.json"
DATA_QUALITY_REPORT_PATH = PROCESSED_DATA_DIR / "data_quality_report.json"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
EVAL_METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
BASELINE_METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.json"
CALIBRATION_CURVE_PATH = ARTIFACTS_DIR / "calibration_curve.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.json"
CHALLENGE_RESULTS_PATH = ARTIFACTS_DIR / "challenge_results.json"

# Random Seeds & Splits
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Input Adequacy Constraints
MIN_INPUT_CHARS = 40
MIN_INPUT_WORDS = 8
MAX_INPUT_CHARS = 30000
MIN_VOCAB_DIVERSITY_RATIO = 0.25

# Default Decision Threshold for Fraud Alert (tuned on validation set)
DEFAULT_DECISION_THRESHOLD = 0.40

# Review Band Margins (Around threshold)
REVIEW_BAND_LOWER = 0.30
REVIEW_BAND_UPPER = 0.50

# Text columns used for model input
TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
TARGET_COLUMN = "fraudulent"
