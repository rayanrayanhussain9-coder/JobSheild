"""
JobShield Model Training and Evaluation Pipeline
Trains baseline and candidate linear models, performs group-disciplined validation,
tunes the operating decision threshold on validation data, evaluates the frozen
winner on the locked final test set, and exports all required artifacts.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    SPLIT_MANIFEST_PATH,
    ARTIFACTS_DIR,
    MODEL_PATH,
    METADATA_PATH,
    EVAL_METRICS_PATH,
    BASELINE_METRICS_PATH,
    CALIBRATION_CURVE_PATH,
    CONFUSION_MATRIX_PATH,
    RANDOM_SEED,
    DEFAULT_DECISION_THRESHOLD,
)

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics for the positive (fraud) class."""
    y_pred = (y_probs >= threshold).astype(int)

    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    ap = average_precision_score(y_true, y_probs)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": round(float(threshold), 3),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "average_precision": round(float(ap), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Sweeps candidate decision thresholds on validation probabilities
    to find the threshold that maximizes fraud-class F1 score.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_threshold = 0.50
    best_f1 = -1.0
    best_metrics = {}

    for t in thresholds:
        metrics = evaluate_predictions(y_true, y_probs, threshold=t)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = t
            best_metrics = metrics

    return round(float(best_threshold), 2), best_metrics


def build_candidate_pipelines() -> Dict[str, Pipeline]:
    """Defines baseline and candidate model pipelines."""
    candidates = {}

    # Candidate 1 (Baseline 2): Word TF-IDF + Logistic Regression (unweighted)
    candidates["tfidf_logreg_unweighted"] = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            min_df=3,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED,
        )),
    ])

    # Candidate 2: Word TF-IDF + Logistic Regression (balanced class weight)
    candidates["tfidf_logreg_balanced"] = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            min_df=3,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )),
    ])

    # Candidate 3: Word TF-IDF + Logistic Regression (mild fraud boost class weight {0: 1, 1: 3})
    candidates["tfidf_logreg_weighted_3x"] = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            min_df=3,
        )),
        ("clf", LogisticRegression(
            C=2.0,
            class_weight={0: 1.0, 1: 3.5},
            max_iter=1000,
            random_state=RANDOM_SEED,
        )),
    ])

    # Candidate 4: Word TF-IDF + LinearSVC (Calibrated with Sigmoid Platt Scaling)
    candidates["tfidf_linearsvc_calibrated"] = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            min_df=3,
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED),
            method="sigmoid",
            cv=3,
        )),
    ])

    # Candidate 5: Word + Char n-gram TF-IDF + Logistic Regression
    candidates["tfidf_word_char_logreg"] = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            max_features=30000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=3.0,
            class_weight={0: 1.0, 1: 2.5},
            max_iter=1000,
            random_state=RANDOM_SEED,
        )),
    ])

    return candidates


def run_training_and_evaluation() -> Dict[str, Any]:
    """Executes the full experiment suite, selects winner, evaluates on test, and writes artifacts."""
    logger.info("Loading train, val, test datasets...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    X_train = df_train["clean_text"].fillna("").astype(str)
    y_train = df_train["fraudulent"].values

    X_val = df_val["clean_text"].fillna("").astype(str)
    y_val = df_val["fraudulent"].values

    X_test = df_test["clean_text"].fillna("").astype(str)
    y_test = df_test["fraudulent"].values

    logger.info("Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    # Baseline 0: Majority Class Classifier (predicts 0 always)
    maj_probs_val = np.zeros(len(y_val))
    baseline_0_metrics = evaluate_predictions(y_val, maj_probs_val, threshold=0.5)
    baseline_0_metrics["candidate_name"] = "majority_class_baseline"
    baseline_0_metrics["description"] = "Always predicts legitimate (class 0)"

    candidates = build_candidate_pipelines()
    experiment_results = [baseline_0_metrics]

    best_candidate_name = None
    best_candidate_pipeline = None
    best_val_f1 = -1.0
    best_val_metrics = None
    best_threshold = DEFAULT_DECISION_THRESHOLD

    for name, pipe in candidates.items():
        logger.info("Fitting candidate: %s", name)
        pipe.fit(X_train, y_train)

        # Predict validation probabilities
        if hasattr(pipe, "predict_proba"):
            val_probs = pipe.predict_proba(X_val)[:, 1]
        elif hasattr(pipe, "decision_function"):
            raw_scores = pipe.decision_function(X_val)
            val_probs = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            val_probs = pipe.predict(X_val).astype(float)

        opt_thresh, opt_metrics = find_optimal_threshold(y_val, val_probs)
        opt_metrics["candidate_name"] = name
        logger.info(
            "Candidate %s -> Val F1: %.4f (Precision: %.4f, Recall: %.4f, Threshold: %.2f, PR-AUC: %.4f)",
            name,
            opt_metrics["f1"],
            opt_metrics["precision"],
            opt_metrics["recall"],
            opt_thresh,
            opt_metrics["average_precision"],
        )
        experiment_results.append(opt_metrics)

        # Selection criterion: Highest fraud-class F1 on validation set
        if opt_metrics["f1"] > best_val_f1:
            best_val_f1 = opt_metrics["f1"]
            best_candidate_name = name
            best_candidate_pipeline = pipe
            best_val_metrics = opt_metrics
            best_threshold = opt_thresh

    logger.info(
        "Winning candidate selected on validation: %s (Val F1: %.4f, Threshold: %.2f)",
        best_candidate_name,
        best_val_f1,
        best_threshold,
    )

    # Final Locked Test Evaluation
    logger.info("Evaluating selected frozen candidate on locked test set...")
    if hasattr(best_candidate_pipeline, "predict_proba"):
        test_probs = best_candidate_pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(best_candidate_pipeline, "decision_function"):
        raw_scores = best_candidate_pipeline.decision_function(X_test)
        test_probs = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        test_probs = best_candidate_pipeline.predict(X_test).astype(float)

    final_test_metrics = evaluate_predictions(y_test, test_probs, threshold=best_threshold)
    logger.info(
        "Final Test Evaluation -> Precision: %.4f, Recall: %.4f, F1: %.4f, PR-AUC: %.4f",
        final_test_metrics["precision"],
        final_test_metrics["recall"],
        final_test_metrics["f1"],
        final_test_metrics["average_precision"],
    )

    # Compute Calibration Curve on Validation Data
    val_probs_winner = best_candidate_pipeline.predict_proba(X_val)[:, 1]
    prob_true, prob_pred = calibration_curve(y_val, val_probs_winner, n_bins=8, strategy="uniform")
    calibration_data = {
        "binned_true_probabilities": [round(float(p), 4) for p in prob_true],
        "binned_mean_predicted_probabilities": [round(float(p), 4) for p in prob_pred],
        "interpretation": (
            "Points close to the diagonal y=x indicate that predicted scores accurately match "
            "the empirical rate of fraudulent postings in historical data."
        ),
    }

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Trained model pipeline
    joblib.dump(best_candidate_pipeline, MODEL_PATH)

    # 2. Raw data hash & split manifest info
    with open(SPLIT_MANIFEST_PATH) as f:
        split_manifest = json.load(f)

    vectorizer_step = best_candidate_pipeline.named_steps["tfidf"]
    clf_step = best_candidate_pipeline.named_steps["clf"]

    safe_params = {}
    for k, v in clf_step.get_params().items():
        try:
            json.dumps(v)
            safe_params[k] = v
        except (TypeError, OverflowError):
            safe_params[k] = str(v)

    metadata = {
        "model_version": "1.0.0",
        "model_type": type(clf_step).__name__,
        "pipeline_name": best_candidate_name,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "raw_data_sha256": split_manifest.get("raw_file_sha256", ""),
        "decision_threshold": best_threshold,
        "feature_count": len(vectorizer_step.vocabulary_),
        "vocabulary_size": len(vectorizer_step.vocabulary_),
        "ngram_range": list(vectorizer_step.ngram_range),
        "split_summary": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "classifier_hyperparameters": safe_params,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. Evaluation metrics (Validation vs Final Test)
    eval_metrics = {
        "selected_model": best_candidate_name,
        "operating_threshold": best_threshold,
        "validation_metrics": best_val_metrics,
        "final_test_metrics": final_test_metrics,
        "note": "Final test set remained completely untouched until hyperparameter and threshold selection were frozen.",
    }
    with open(EVAL_METRICS_PATH, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # 4. Baseline metrics comparison
    with open(BASELINE_METRICS_PATH, "w") as f:
        json.dump(experiment_results, f, indent=2)

    # 5. Confusion matrix summary
    cm_summary = {
        "operating_threshold": best_threshold,
        "validation_confusion_matrix": best_val_metrics["confusion_matrix"],
        "final_test_confusion_matrix": final_test_metrics["confusion_matrix"],
    }
    with open(CONFUSION_MATRIX_PATH, "w") as f:
        json.dump(cm_summary, f, indent=2)

    # 6. Calibration curve data
    with open(CALIBRATION_CURVE_PATH, "w") as f:
        json.dump(calibration_data, f, indent=2)

    logger.info("All model artifacts and evaluation summaries written to %s", ARTIFACTS_DIR)
    return {
        "selected_candidate": best_candidate_name,
        "operating_threshold": best_threshold,
        "val_metrics": best_val_metrics,
        "test_metrics": final_test_metrics,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_training_and_evaluation()
    print("Training and evaluation completed successfully:")
    print(json.dumps(results, indent=2))
