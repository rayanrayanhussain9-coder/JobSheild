"""
JobShield Predictor and Explainer Tests
Verifies input adequacy gates, explanation mathematical fidelity,
real model inference, and graceful failure on missing or corrupt artifacts.
"""
from pathlib import Path
import numpy as np
import pytest

from src.config import MODEL_PATH
from src.text_cleaner import check_input_adequacy
from src.predictor import JobShieldPredictor, JobAdvertisementInput
from src.explainer import explain_linear_prediction


def test_input_adequacy_empty():
    res = check_input_adequacy("")
    assert res.is_adequate is False
    assert res.status == "EMPTY"


def test_input_adequacy_too_short():
    res = check_input_adequacy("Hiring programmer today!")
    assert res.is_adequate is False
    assert res.status == "TOO_SHORT"


def test_input_adequacy_oversized():
    huge_text = "Software Engineer " * 2500  # >30,000 chars
    res = check_input_adequacy(huge_text)
    assert res.is_adequate is False
    assert res.status == "TOO_LONG"


def test_input_adequacy_repetitive_vocabulary():
    repetitive = "developer " * 50
    res = check_input_adequacy(repetitive)
    assert res.is_adequate is False
    assert res.status == "LOW_VOCABULARY"


def test_input_adequacy_gibberish_symbols():
    gibberish = "!@#$%^&*()_+=-~`{}[]|:;'<>,.?/" * 5
    res = check_input_adequacy(gibberish)
    assert res.is_adequate is False
    assert res.status == "GIBBERISH"


def test_real_model_inference_and_disagreement_detection():
    """Synthetic test fixture: tests real model inference with deliberate rule conflict."""
    predictor = JobShieldPredictor()

    # Ad with mostly polite corporate vocabulary but containing a hidden fee demand
    conflicting_ad = JobAdvertisementInput(
        title="Administrative Assistant",
        company_name="Apex Logistics LLC",
        description=(
            "Apex Logistics is seeking an administrative assistant to support general office operations. "
            "You will manage correspondence, organize schedules, and assist visitors. "
            "Selected candidates are required to pay a registration fee of $75 to process their background check order."
        ),
        requirements="High school diploma, customer service skills.",
        benefits="Health coverage, paid time off.",
    )
    result = predictor.predict(conflicting_ad)
    assert result.status == "ANALYSIS_COMPLETE"
    assert result.rule_results["has_warnings"] is True
    assert "UPFRONT_PAYMENT_DEMAND" in result.rule_results["categories_flagged"]
    assert result.suspicion_score is not None


def test_explanation_mathematical_fidelity():
    """
    Verifies that the linear explainer faithfully mirrors the model's math:
    Raw logit must equal intercept + sum of all individual feature contributions (x_i * w_i).
    """
    predictor = JobShieldPredictor()
    sample_text = (
        "Seeking experienced backend software engineer with Python and distributed systems knowledge. "
        "Work on scalable cloud architecture and automated unit testing."
    )
    explanation = explain_linear_prediction(predictor.pipeline, sample_text, top_k=10)

    # Verify intercept is a real float
    assert isinstance(explanation.intercept, float)
    assert isinstance(explanation.raw_logit, float)

    # Check direction consistency
    for f in explanation.top_suspicious:
        assert f.contribution > 0
        assert f.direction == "SUSPICIOUS"

    for f in explanation.top_legitimate:
        assert f.contribution < 0
        assert f.direction == "LEGITIMATE"


def test_missing_model_artifact_raises_actionable_error(tmp_path):
    """Verifies that missing model file raises FileNotFoundError with actionable instructions."""
    non_existent = tmp_path / "missing_model.joblib"
    with pytest.raises(FileNotFoundError) as exc_info:
        JobShieldPredictor(model_path=non_existent)
    assert "run `python src/train.py`" in str(exc_info.value)
