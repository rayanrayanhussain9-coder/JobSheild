"""
JobShield Faithful Local Linear Explainer
Computes exact local feature attributions (x_i * w_i) from the trained
linear classification pipeline, transparently exposing intercept,
features pushing toward suspiciousness, and features pushing toward legitimacy.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class FeatureContribution:
    feature: str
    contribution: float  # x_i * w_i
    tfidf_value: float  # x_i
    model_weight: float  # w_i
    direction: str  # "SUSPICIOUS" or "LEGITIMATE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "contribution": round(float(self.contribution), 4),
            "tfidf_value": round(float(self.tfidf_value), 4),
            "model_weight": round(float(self.model_weight), 4),
            "direction": self.direction,
        }


@dataclass
class ExplanationResult:
    intercept: float
    raw_logit: float
    total_feature_contribution: float
    top_suspicious: List[FeatureContribution] = field(default_factory=list)
    top_legitimate: List[FeatureContribution] = field(default_factory=list)
    method: str = "Exact Linear Feature Attribution (x_i * w_i)"
    disclaimer: str = (
        "These contributions represent exact mathematical components (x_i * w_i) of the "
        "trained linear model. They reflect historical statistical associations in the training "
        "data, not causal proof of fraud."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intercept": round(float(self.intercept), 4),
            "raw_logit": round(float(self.raw_logit), 4),
            "total_feature_contribution": round(float(self.total_feature_contribution), 4),
            "top_suspicious": [f.to_dict() for f in self.top_suspicious],
            "top_legitimate": [f.to_dict() for f in self.top_legitimate],
            "method": self.method,
            "disclaimer": self.disclaimer,
        }


def get_pipeline_components(pipeline):
    """Extracts the TF-IDF vectorizer and the linear estimator from the pipeline."""
    # Find TfidfVectorizer
    vectorizer = None
    classifier = None

    if hasattr(pipeline, "named_steps"):
        for name, step in pipeline.named_steps.items():
            if hasattr(step, "transform") and hasattr(step, "vocabulary_"):
                vectorizer = step
            if hasattr(step, "coef_") or hasattr(step, "calibrated_classifiers_"):
                classifier = step
    elif hasattr(pipeline, "steps"):
        for name, step in pipeline.steps:
            if hasattr(step, "transform") and hasattr(step, "vocabulary_"):
                vectorizer = step
            if hasattr(step, "coef_") or hasattr(step, "calibrated_classifiers_"):
                classifier = step

    return vectorizer, classifier


def explain_linear_prediction(pipeline, text: str, top_k: int = 8) -> ExplanationResult:
    """
    Computes exact local feature attribution for a single text input.
    Supports standard linear classifiers (LogisticRegression, CalibratedClassifierCV with linear base).
    """
    vectorizer, classifier = get_pipeline_components(pipeline)
    if vectorizer is None or classifier is None:
        raise ValueError("Pipeline must contain a TF-IDF vectorizer and a linear classifier.")

    # Extract weights and intercept
    if hasattr(classifier, "coef_"):
        coef = classifier.coef_[0]
        intercept = float(classifier.intercept_[0]) if hasattr(classifier, "intercept_") else 0.0
    elif hasattr(classifier, "calibrated_classifiers_"):
        # For CalibratedClassifierCV, average the weights across calibrated folds
        coefs = []
        intercepts = []
        for cc in classifier.calibrated_classifiers_:
            base_est = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
            if hasattr(base_est, "coef_"):
                coefs.append(base_est.coef_[0])
                if hasattr(base_est, "intercept_"):
                    intercepts.append(base_est.intercept_[0])
        if coefs:
            coef = np.mean(coefs, axis=0)
            intercept = float(np.mean(intercepts)) if intercepts else 0.0
        else:
            raise ValueError("Calibrated classifier base estimator has no coef_.")
    else:
        raise ValueError(f"Classifier type {type(classifier)} does not expose linear coef_.")

    feature_names = vectorizer.get_feature_names_out()
    x = vectorizer.transform([text])
    nonzero_indices = x.nonzero()[1]

    contributions = []
    for idx in nonzero_indices:
        tfidf_val = x[0, idx]
        weight = coef[idx]
        contrib = tfidf_val * weight
        feat_name = feature_names[idx]

        direction = "SUSPICIOUS" if contrib > 0 else "LEGITIMATE"
        contributions.append(
            FeatureContribution(
                feature=feat_name,
                contribution=contrib,
                tfidf_value=tfidf_val,
                model_weight=weight,
                direction=direction,
            )
        )

    # Sort suspicious by largest positive contribution
    suspicious = sorted(
        [c for c in contributions if c.contribution > 0],
        key=lambda c: c.contribution,
        reverse=True,
    )[:top_k]

    # Sort legitimate by largest negative contribution (most negative first)
    legitimate = sorted(
        [c for c in contributions if c.contribution < 0],
        key=lambda c: c.contribution,
    )[:top_k]

    total_feature_contrib = sum(c.contribution for c in contributions)
    raw_logit = intercept + total_feature_contrib

    return ExplanationResult(
        intercept=intercept,
        raw_logit=raw_logit,
        total_feature_contribution=total_feature_contrib,
        top_suspicious=suspicious,
        top_legitimate=legitimate,
    )
