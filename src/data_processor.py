"""
JobShield Data Processor
Handles dataset schema validation, missing-value profiling,
template-aware deduplication grouping, and leakage-free stratified splitting.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    SPLIT_MANIFEST_PATH,
    DATA_QUALITY_REPORT_PATH,
    RANDOM_SEED,
    TEXT_COLUMNS,
    TARGET_COLUMN,
)
from src.text_cleaner import normalize_text_for_modeling

logger = logging.getLogger(__name__)

EXPECTED_RAW_COLUMNS = [
    "job_id",
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "fraudulent",
]


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validates presence and types of core columns in raw DataFrame."""
    missing_cols = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        return False, f"Missing required columns in dataset: {missing_cols}"

    if TARGET_COLUMN not in df.columns:
        return False, f"Target column '{TARGET_COLUMN}' is missing."

    valid_labels = set(df[TARGET_COLUMN].dropna().unique())
    if not valid_labels.issubset({0, 1}):
        return False, f"Target labels must be binary {0, 1}, found: {valid_labels}"

    return True, "Schema is valid."


def create_combined_clean_text(df: pd.DataFrame) -> pd.Series:
    """
    Concatenates available text fields into a single unified clean string.
    Missing optional fields are gracefully handled as empty strings.
    """
    combined = (
        df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("company_profile", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("requirements", pd.Series("", index=df.index)).fillna("").astype(str)
        + " "
        + df.get("benefits", pd.Series("", index=df.index)).fillna("").astype(str)
    )
    return combined.apply(normalize_text_for_modeling)


def generate_template_group_ids(clean_texts: pd.Series) -> pd.Series:
    """
    Assigns an identical group_id (SHA-256 hash) to identical or normalized template texts.
    Ensures all duplicate / template copies are grouped strictly together.
    """
    def text_hash(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    return clean_texts.apply(text_hash)


def generate_data_quality_report(df: pd.DataFrame, clean_texts: pd.Series, group_ids: pd.Series) -> Dict[str, Any]:
    """Compiles thorough data profiling and data hygiene metrics."""
    total_records = len(df)
    null_counts = df.isnull().sum().to_dict()
    null_percentages = (df.isnull().mean() * 100).round(2).to_dict()

    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    class_proportions = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict()

    total_groups = group_ids.nunique()
    exact_duplicate_rows = total_records - total_groups

    # Check for ambiguous / conflicting labels within template groups
    grouped_labels = df.groupby(group_ids)[TARGET_COLUMN].nunique()
    conflicting_groups = int((grouped_labels > 1).sum())

    report = {
        "dataset_name": "EMSCAD (Employment Scam Aegean Dataset)",
        "total_records": total_records,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "target_column": TARGET_COLUMN,
        "class_distribution": {
            "legitimate_count": int(class_counts.get(0, 0)),
            "fraudulent_count": int(class_counts.get(1, 0)),
            "legitimate_pct": float(class_proportions.get(0, 0.0)),
            "fraudulent_pct": float(class_proportions.get(1, 0.0)),
        },
        "missing_values": {
            col: {
                "count": int(null_counts[col]),
                "pct": float(null_percentages[col]),
            }
            for col in df.columns
        },
        "duplicate_analysis": {
            "unique_text_templates": total_groups,
            "duplicate_records_detected": exact_duplicate_rows,
            "conflicting_label_groups": conflicting_groups,
            "grouping_strategy": "Normalized SHA-256 text template hash to prevent cross-partition leakage",
        },
    }
    return report


def split_data_group_stratified(
    df: pd.DataFrame,
    clean_texts: pd.Series,
    group_ids: pd.Series,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Performs a 70% Train, 15% Validation, 15% Test group-stratified split.
    All duplicate/template group members remain strictly in the same split.
    Final test partition is isolated and protected.
    """
    df_prepared = df.copy()
    df_prepared["clean_text"] = clean_texts
    df_prepared["group_id"] = group_ids

    # Step 1: Divide into 70% Train vs 30% Temp (Val + Test) using StratifiedGroupKFold
    # With n_splits=10, 7 folds can form train (70%) and 3 folds form temp (30%).
    # Then split the 30% temp into 15% Val and 15% Test.
    sgkf_outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    
    # We choose fold assignment
    folds = np.zeros(len(df_prepared), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(
        sgkf_outer.split(df_prepared, df_prepared[TARGET_COLUMN], groups=df_prepared["group_id"])
    ):
        folds[test_idx] = fold_idx

    # Assign folds 0..6 to Train (70%), folds 7..9 to Temp (30%)
    train_mask = folds < 7
    temp_mask = folds >= 7

    df_train = df_prepared[train_mask].copy()
    df_temp = df_prepared[temp_mask].copy()

    # Step 2: Split temp (30%) into 15% Val (folds 7 and half of 8/9) or using a 2-fold StratifiedGroupKFold on temp
    sgkf_inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_fold_assignments = np.zeros(len(df_temp), dtype=int)
    for inner_idx, (_, test_idx) in enumerate(
        sgkf_inner.split(df_temp, df_temp[TARGET_COLUMN], groups=df_temp["group_id"])
    ):
        val_fold_assignments[test_idx] = inner_idx

    df_val = df_temp[val_fold_assignments == 0].copy()
    df_test = df_temp[val_fold_assignments == 1].copy()

    # Leakage check: Verify group intersection between splits is strictly empty
    train_groups = set(df_train["group_id"])
    val_groups = set(df_val["group_id"])
    test_groups = set(df_test["group_id"])

    assert train_groups.isdisjoint(val_groups), "Leakage detected: Train and Val share groups!"
    assert train_groups.isdisjoint(test_groups), "Leakage detected: Train and Test share groups!"
    assert val_groups.isdisjoint(test_groups), "Leakage detected: Val and Test share groups!"

    manifest = {
        "random_seed": seed,
        "split_method": "StratifiedGroupKFold (Template-Aware Grouping)",
        "train": {
            "total_records": len(df_train),
            "legitimate": int((df_train[TARGET_COLUMN] == 0).sum()),
            "fraudulent": int((df_train[TARGET_COLUMN] == 1).sum()),
            "fraud_rate_pct": round(float((df_train[TARGET_COLUMN] == 1).mean() * 100), 2),
            "unique_groups": int(df_train["group_id"].nunique()),
        },
        "val": {
            "total_records": len(df_val),
            "legitimate": int((df_val[TARGET_COLUMN] == 0).sum()),
            "fraudulent": int((df_val[TARGET_COLUMN] == 1).sum()),
            "fraud_rate_pct": round(float((df_val[TARGET_COLUMN] == 1).mean() * 100), 2),
            "unique_groups": int(df_val["group_id"].nunique()),
        },
        "test": {
            "total_records": len(df_test),
            "legitimate": int((df_test[TARGET_COLUMN] == 0).sum()),
            "fraudulent": int((df_test[TARGET_COLUMN] == 1).sum()),
            "fraud_rate_pct": round(float((df_test[TARGET_COLUMN] == 1).mean() * 100), 2),
            "unique_groups": int(df_test["group_id"].nunique()),
        },
        "total_records": len(df_prepared),
        "leakage_verification": {
            "train_val_overlap_count": len(train_groups.intersection(val_groups)),
            "train_test_overlap_count": len(train_groups.intersection(test_groups)),
            "val_test_overlap_count": len(val_groups.intersection(test_groups)),
            "is_leakage_free": True,
        },
    }

    return df_train, df_val, df_test, manifest


def process_and_save_data(raw_csv_path: Path = RAW_DATA_PATH) -> Dict[str, Any]:
    """
    Orchestrates the entire ingestion, validation, deduplication,
    splitting, and artifact generation process.
    """
    logger.info("Loading raw data from %s", raw_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_csv_path}")

    raw_sha256 = compute_file_sha256(raw_csv_path)
    df_raw = pd.read_csv(raw_csv_path)

    is_valid, msg = validate_schema(df_raw)
    if not is_valid:
        raise ValueError(f"Dataset schema validation failed: {msg}")

    # Generate combined text and template group IDs
    clean_texts = create_combined_clean_text(df_raw)
    group_ids = generate_template_group_ids(clean_texts)

    # Generate and save Data Quality Report
    quality_report = generate_data_quality_report(df_raw, clean_texts, group_ids)
    quality_report["raw_file_sha256"] = raw_sha256

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_QUALITY_REPORT_PATH, "w") as f:
        json.dump(quality_report, f, indent=2)

    # Perform group-aware stratified split
    df_train, df_val, df_test, manifest = split_data_group_stratified(
        df_raw, clean_texts, group_ids, seed=RANDOM_SEED
    )
    manifest["raw_file_sha256"] = raw_sha256

    with open(SPLIT_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # Save split datasets
    df_train.to_csv(TRAIN_DATA_PATH, index=False)
    df_val.to_csv(VAL_DATA_PATH, index=False)
    df_test.to_csv(TEST_DATA_PATH, index=False)

    logger.info("Processed data successfully saved to %s", PROCESSED_DATA_DIR)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = process_and_save_data()
    print("Data processing complete. Manifest:\n", json.dumps(res, indent=2))
