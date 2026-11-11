"""
JobShield Inference Engine
Orchestrates input adequacy gating, rule-based screening, linear model inference,
honest scoring, local explanations, and missing-information analysis.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np

from src.config import (
    MODEL_PATH,
    METADATA_PATH,
    DEFAULT_DECISION_THRESHOLD,
    REVIEW_BAND_LOWER,
    REVIEW_BAND_UPPER,
)
from src.text_cleaner import check_input_adequacy, normalize_text_for_modeling, AdequacyResult
from src.rules_engine import screen_advertisement_rules, RuleEngineResult, RuleWarning
from src.explainer import explain_linear_prediction, ExplanationResult

logger = logging.getLogger(__name__)


@dataclass
class JobAdvertisementInput:
    """Represents the raw user inputs for a job advertisement screening request."""
    title: str = ""
    company_name: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""
    telecommuting: bool = False

    def build_full_text(self) -> str:
        """Constructs unified text for analysis."""
        parts = []
        if self.title.strip():
            parts.append(f"Title: {self.title.strip()}")
        if self.company_name.strip():
            parts.append(f"Company: {self.company_name.strip()}")
        if self.description.strip():
            parts.append(self.description.strip())
        if self.requirements.strip():
            parts.append(f"Requirements: {self.requirements.strip()}")
        if self.benefits.strip():
            parts.append(f"Benefits: {self.benefits.strip()}")
        return "\n\n".join(parts)


@dataclass
class MissingInformationObservation:
    field_name: str
    severity: str
    observation: str


@dataclass
class JobShieldResult:
    """Full structured screening response."""
    status: str  # "ANALYSIS_COMPLETE", "INADEQUATE_INPUT", "ERROR"
    input_adequacy: Dict[str, Any]
    model_version: str
    operating_threshold: float

    # ML Model Outputs
    model_prediction: Optional[str] = None  # "SUSPICIOUS_PATTERN", "NEEDS_REVIEW", "LOW_RISK_PATTERN"
    suspicion_score: Optional[float] = None
    score_label: str = "Model Suspicion Score (Calibrated historical pattern similarity)"

    # Explanations
    explanation: Optional[Dict[str, Any]] = None

    # Rule Engine Outputs
    rule_results: Optional[Dict[str, Any]] = None

    # Missing Information
    missing_info_observations: List[Dict[str, Any]] = field(default_factory=list)

    # Synthesis & Disagreement
    synthesis_summary: str = ""
    disagreement_notice: Optional[str] = None
    limitations_disclaimer: str = (
        "JobShield screens advertisement text for historical scam patterns and red-flag phrases. "
        "It does NOT verify company registration, authenticate recruiters, or guarantee safety."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "input_adequacy": self.input_adequacy,
            "model_version": self.model_version,
            "operating_threshold": self.operating_threshold,
            "model_prediction": self.model_prediction,
            "suspicion_score": self.suspicion_score,
            "score_label": self.score_label,
            "explanation": self.explanation,
            "rule_results": self.rule_results,
            "missing_info_observations": self.missing_info_observations,
            "synthesis_summary": self.synthesis_summary,
            "disagreement_notice": self.disagreement_notice,
            "limitations_disclaimer": self.limitations_disclaimer,
        }


class JobShieldPredictor:
    """Thread-safe, state-isolated predictor loaded from trusted local artifacts."""

    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata = {}
        self.threshold = DEFAULT_DECISION_THRESHOLD
        self._load_artifacts()

    def _load_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Please run `python src/train.py` to generate the trained model."
            )
        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Corrupt or incompatible model artifact at {self.model_path}: {e}")

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path) as f:
                    self.metadata = json.load(f)
                    self.threshold = self.metadata.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
            except Exception as e:
                logger.warning("Could not read metadata at %s: %s", self.metadata_path, e)
                self.metadata = {"model_version": "1.0.0"}

    def detect_missing_information(self, ad: JobAdvertisementInput) -> List[MissingInformationObservation]:
        """Identifies omitted contextual fields that frequently appear in authentic postings."""
        observations = []
        combined_text = (ad.description + " " + ad.requirements + " " + ad.benefits).lower()

        if not ad.company_name.strip():
            observations.append(
                MissingInformationObservation(
                    field_name="Company Name",
                    severity="MEDIUM",
                    observation="No employer or organization name was provided for context.",
                )
            )

        if not ad.requirements.strip() and "requirement" not in combined_text and "qualification" not in combined_text:
            observations.append(
                MissingInformationObservation(
                    field_name="Job Requirements",
                    severity="LOW",
                    observation="No explicit educational or skill prerequisites are detailed.",
                )
            )

        # Check for contact/location specifics
        has_contact = any(k in combined_text for k in ["email", "contact", "apply at", "http", "phone", "office"])
        if not has_contact:
            observations.append(
                MissingInformationObservation(
                    field_name="Application / Contact Route",
                    severity="LOW",
                    observation="No explicit company email, office address, or application portal was mentioned.",
                )
            )

        return observations

    def predict(self, ad: JobAdvertisementInput) -> JobShieldResult:
        """Executes full screening pipeline on a job advertisement input."""
        full_text = ad.build_full_text()

        # Step 1: Input Adequacy Gate
        adequacy = check_input_adequacy(full_text)
        if not adequacy.is_adequate:
            return JobShieldResult(
                status="INADEQUATE_INPUT",
                input_adequacy={
                    "is_adequate": False,
                    "status": adequacy.status,
                    "message": adequacy.message,
                    "metrics": adequacy.metrics,
                },
                model_version=self.metadata.get("model_version", "1.0.0"),
                operating_threshold=self.threshold,
                synthesis_summary=(
                    f"Screening stopped: {adequacy.message} "
                    "A model score cannot be responsibly generated without adequate text."
                ),
            )

        # Step 2: Separate Rule-Based Warning Screening
        rule_res = screen_advertisement_rules(full_text)

        # Step 3: Missing Information Checks
        missing_obs = self.detect_missing_information(ad)

        # Step 4: ML Model Inference
        clean_text_norm = normalize_text_for_modeling(full_text)
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([clean_text_norm])[0]
            suspicion_score = round(float(probs[1]), 4)
        elif hasattr(self.pipeline, "decision_function"):
            raw_logit = self.pipeline.decision_function([clean_text_norm])[0]
            suspicion_score = round(float(1.0 / (1.0 + np.exp(-raw_logit))), 4)
        else:
            pred_class = self.pipeline.predict([clean_text_norm])[0]
            suspicion_score = 1.0 if pred_class == 1 else 0.0

        # Categorize prediction based on tuned threshold & review band
        review_margin = 0.08
        if suspicion_score >= self.threshold:
            prediction_label = "SUSPICIOUS_PATTERN"
        elif suspicion_score >= (self.threshold - review_margin):
            prediction_label = "NEEDS_REVIEW"
        else:
            prediction_label = "LOW_RISK_PATTERN"

        # Step 5: Faithful Local Linear Feature Explanations
        explanation = explain_linear_prediction(self.pipeline, clean_text_norm, top_k=8)

        # Step 6: Synthesis and Disagreement Analysis
        disagreement_notice = None
        if rule_res.has_warnings and prediction_label == "LOW_RISK_PATTERN":
            disagreement_notice = (
                "Important Disagreement: The ML model assigned a low suspicion score based on general vocabulary, "
                "but the deterministic rule engine flagged critical red flags (such as upfront payment demands). "
                "Rule-based red flags MUST take precedence in your diligence."
            )
        elif not rule_res.has_warnings and prediction_label == "SUSPICIOUS_PATTERN":
            disagreement_notice = (
                "Notice: The ML model detected subtle linguistic and structural patterns commonly associated "
                "with fraudulent job postings in historical data, even though no explicit payment or banking rule was tripped."
            )

        # Build synthesis summary
        if rule_res.has_warnings and prediction_label == "SUSPICIOUS_PATTERN":
            synthesis = (
                "HIGH CAUTION: Both the machine learning model and rule engine identified strong indicators of recruitment fraud. "
                "Do NOT transfer funds, share banking credentials, or accept offers without independent company verification."
            )
        elif rule_res.has_warnings:
            synthesis = (
                "CAUTION: Explicit warning rules triggered red flags (e.g. fee demands or untraceable payment methods), "
                "despite general text looking standard. Exercise extreme caution."
            )
        elif prediction_label == "SUSPICIOUS_PATTERN":
            synthesis = (
                "ELEVATED RISK: The advertisement's vocabulary and phrasing closely match historical fraudulent patterns. "
                "Review the top contributing features and verify the employer through official channels."
            )
        elif prediction_label == "NEEDS_REVIEW":
            synthesis = (
                "INCONCLUSIVE / BORDERLINE: Model score lies near the operating threshold. "
                "Inspect the missing information and conduct secondary verification before applying."
            )
        else:
            synthesis = (
                "LOW TEXT RISK: No explicit scam rules were triggered, and phrasing aligns with typical legitimate postings. "
                "Remember that text screening alone cannot guarantee employer authenticity or legitimacy."
            )

        return JobShieldResult(
            status="ANALYSIS_COMPLETE",
            input_adequacy={
                "is_adequate": True,
                "status": "ADEQUATE",
                "message": adequacy.message,
                "metrics": adequacy.metrics,
            },
            model_version=self.metadata.get("model_version", "1.0.0"),
            operating_threshold=self.threshold,
            model_prediction=prediction_label,
            suspicion_score=suspicion_score,
            score_label="Model Suspicion Score (Calibrated historical pattern similarity)",
            explanation=explanation.to_dict(),
            rule_results=rule_res.to_dict(),
            missing_info_observations=[
                {"field": obs.field_name, "severity": obs.severity, "observation": obs.observation}
                for obs in missing_obs
            ],
            synthesis_summary=synthesis,
            disagreement_notice=disagreement_notice,
        )
