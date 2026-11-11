"""
JobShield Text Cleaner and Input Adequacy Validator
Handles text hygiene, HTML tag stripping, and input adequacy gating.
"""
import re
import html
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from src.config import (
    MIN_INPUT_CHARS,
    MIN_INPUT_WORDS,
    MAX_INPUT_CHARS,
    MIN_VOCAB_DIVERSITY_RATIO,
)


@dataclass
class AdequacyResult:
    """Represents the outcome of input adequacy validation."""
    is_adequate: bool
    status: str  # "ADEQUATE", "TOO_SHORT", "TOO_LONG", "EMPTY", "LOW_VOCABULARY", "GIBBERISH"
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)


def clean_html_and_entities(text: str) -> str:
    """Removes HTML markup and unescapes HTML entities."""
    if not text:
        return ""
    # Unescape entities like &amp; &lt; &gt;
    text = html.unescape(text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_for_modeling(text: str) -> str:
    """
    Standard cleaning pipeline for raw text before vectorization.
    Removes HTML, unescapes entities, and standardizes spacing.
    Case folding is handled by the scikit-learn vectorizer.
    """
    if not text or not isinstance(text, str):
        return ""
    text = clean_html_and_entities(text)
    # Strip non-printable / control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    return text.strip()


def check_input_adequacy(text: str) -> AdequacyResult:
    """
    Conservatively checks if an advertisement text contains sufficient,
    usable content for machine learning inference.
    """
    if text is None:
        return AdequacyResult(
            is_adequate=False,
            status="EMPTY",
            message="Input advertisement text is empty.",
            metrics={"char_count": 0, "word_count": 0},
        )

    cleaned = normalize_text_for_modeling(text)
    char_count = len(cleaned)

    if char_count == 0:
        return AdequacyResult(
            is_adequate=False,
            status="EMPTY",
            message="No readable text detected after removing formatting and whitespace.",
            metrics={"char_count": 0, "word_count": 0},
        )

    if char_count > MAX_INPUT_CHARS:
        return AdequacyResult(
            is_adequate=False,
            status="TOO_LONG",
            message=f"Text exceeds the maximum allowed length of {MAX_INPUT_CHARS:,} characters (received {char_count:,}).",
            metrics={"char_count": char_count, "max_allowed": MAX_INPUT_CHARS},
        )

    # Check for excessive punctuation / non-alphanumeric ratio first if text has length
    alpha_chars = sum(1 for c in cleaned if c.isalnum() or c.isspace())
    alpha_ratio = alpha_chars / char_count if char_count > 0 else 0.0
    if char_count >= MIN_INPUT_CHARS and alpha_ratio < 0.35:
        return AdequacyResult(
            is_adequate=False,
            status="GIBBERISH",
            message="Text contains excessive punctuation or unreadable symbols, lacking sufficient natural language content.",
            metrics={"char_count": char_count, "alpha_ratio": round(alpha_ratio, 3)},
        )

    words = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", cleaned)
    word_count = len(words)

    if char_count < MIN_INPUT_CHARS or word_count < MIN_INPUT_WORDS:
        return AdequacyResult(
            is_adequate=False,
            status="TOO_SHORT",
            message=f"Text is too brief for reliable evaluation (requires at least {MIN_INPUT_CHARS} characters and {MIN_INPUT_WORDS} words; received {char_count} chars, {word_count} words).",
            metrics={"char_count": char_count, "word_count": word_count},
        )

    # Check for vocabulary diversity (repetitive strings or spam repetition)
    unique_words = set(w.lower() for w in words)
    diversity_ratio = len(unique_words) / word_count if word_count > 0 else 0.0

    if word_count >= 15 and diversity_ratio < MIN_VOCAB_DIVERSITY_RATIO:
        return AdequacyResult(
            is_adequate=False,
            status="LOW_VOCABULARY",
            message="Text exhibits unnaturally low vocabulary diversity (repetitive words or spam pattern).",
            metrics={
                "char_count": char_count,
                "word_count": word_count,
                "unique_words": len(unique_words),
                "diversity_ratio": round(diversity_ratio, 3),
            },
        )

    return AdequacyResult(
        is_adequate=True,
        status="ADEQUATE",
        message="Input text meets length and structural quality requirements.",
        metrics={
            "char_count": char_count,
            "word_count": word_count,
            "unique_words": len(unique_words),
            "diversity_ratio": round(diversity_ratio, 3),
        },
    )
