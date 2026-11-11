"""
JobShield Data Processor Tests
Validates schema checks, label constraints, deduplication group isolation,
and preprocessing consistency.
"""
import pytest
import pandas as pd
import numpy as np

from src.data_processor import (
    validate_schema,
    create_combined_clean_text,
    generate_template_group_ids,
    split_data_group_stratified,
    EXPECTED_RAW_COLUMNS,
)
from src.text_cleaner import normalize_text_for_modeling


def test_schema_validation_valid():
    """Synthetic test fixture: valid dataframe structure."""
    df_valid = pd.DataFrame({col: ["test"] for col in EXPECTED_RAW_COLUMNS})
    df_valid["fraudulent"] = [0]
    is_valid, msg = validate_schema(df_valid)
    assert is_valid is True
    assert "Schema is valid" in msg


def test_schema_validation_missing_columns():
    """Synthetic test fixture: missing required columns."""
    df_invalid = pd.DataFrame({"title": ["Software Engineer"], "fraudulent": [0]})
    is_valid, msg = validate_schema(df_invalid)
    assert is_valid is False
    assert "Missing required columns" in msg


def test_schema_validation_invalid_labels():
    """Synthetic test fixture: non-binary target labels."""
    df_bad_labels = pd.DataFrame({col: ["test"] for col in EXPECTED_RAW_COLUMNS})
    df_bad_labels["fraudulent"] = [2]  # Invalid class
    is_valid, msg = validate_schema(df_bad_labels)
    assert is_valid is False
    assert "Target labels must be binary" in msg


def test_template_grouping_and_leakage_free_split():
    """
    Synthetic test fixture: verifies duplicate templates are assigned identical group IDs
    and strictly kept within the same partition without leakage.
    """
    # 20 ad templates, repeated multiple times
    templates = [
        f"Unique Job Description Template Number {i} with specific responsibilities."
        for i in range(20)
    ]
    data = []
    for i, t in enumerate(templates):
        # Repeat each template 3 times (exact duplicate copies)
        for _ in range(3):
            data.append({
                "job_id": len(data) + 1,
                "title": f"Title {i}",
                "company_profile": "Standard company text",
                "description": t,
                "requirements": "Standard requirements",
                "benefits": "Standard benefits",
                "fraudulent": 1 if i % 5 == 0 else 0,
            })
    for col in EXPECTED_RAW_COLUMNS:
        if col not in data[0]:
            for row in data:
                row[col] = "N/A"

    df_synth = pd.DataFrame(data)
    clean_texts = create_combined_clean_text(df_synth)
    group_ids = generate_template_group_ids(clean_texts)

    # All identical copies of the same template must have the same group ID
    assert group_ids.iloc[0] == group_ids.iloc[1] == group_ids.iloc[2]
    assert group_ids.nunique() == 20

    train_df, val_df, test_df, manifest = split_data_group_stratified(
        df_synth, clean_texts, group_ids, seed=42
    )

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert manifest["leakage_verification"]["is_leakage_free"] is True


def test_preprocessing_consistency():
    """Verifies that text normalization produces identical output for identical raw content."""
    raw_html = "<p>Join our team &amp; build software! <br>Immediate opening.</p>"
    clean_1 = normalize_text_for_modeling(raw_html)
    clean_2 = normalize_text_for_modeling(raw_html)
    assert clean_1 == clean_2
    assert "<p>" not in clean_1
    assert "&amp;" not in clean_1
    assert "Join our team & build software! Immediate opening." == clean_1
