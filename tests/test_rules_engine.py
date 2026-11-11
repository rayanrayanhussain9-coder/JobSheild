"""
JobShield Rule Engine Tests
Verifies deterministic fraud rules, contextual negation handling,
reimbursement discernment, and separation from machine learning scores.
"""
import pytest
from src.rules_engine import screen_advertisement_rules


def test_positive_registration_fee():
    """Synthetic test fixture: upfront registration fee."""
    text = "Shortlisted candidates are required to pay a registration fee of $100 before the final interview."
    res = screen_advertisement_rules(text)
    assert res.has_warnings is True
    assert "UPFRONT_PAYMENT_DEMAND" in res.categories_flagged
    assert any(w.rule_id == "RULE_PAY_01" for w in res.warnings)


def test_positive_training_deposit():
    """Synthetic test fixture: refundable training deposit."""
    text = "Successful applicants must pay a refundable training fee to secure their workstation software."
    res = screen_advertisement_rules(text)
    assert res.has_warnings is True
    assert "UPFRONT_PAYMENT_DEMAND" in res.categories_flagged


def test_positive_cryptocurrency_payment():
    """Synthetic test fixture: payment via crypto."""
    text = "Please send payment through bitcoin to confirm your background check order."
    res = screen_advertisement_rules(text)
    assert res.has_warnings is True
    assert "UNCONVENTIONAL_PAYMENT_METHOD" in res.categories_flagged


def test_negated_recruitment_fee_disclaimer():
    """Synthetic test fixture: employer explicitly disclaiming recruitment fees."""
    text = "Global Tech Corp does not charge any application or recruitment fee at any stage of hiring."
    res = screen_advertisement_rules(text)
    assert res.has_warnings is False
    assert len(res.warnings) == 0


def test_negated_warning_in_company_statement():
    """Synthetic test fixture: fraud alert in job posting footer."""
    text = "Scam alert: Beware of fraudulent agencies. We never ask for money or processing charges to apply."
    res = screen_advertisement_rules(text)
    assert res.has_warnings is False
    assert len(res.warnings) == 0


def test_benefit_tuition_reimbursement():
    """Synthetic test fixture: tuition reimbursement benefit must not trigger fee warning."""
    text = "We provide health benefits, 401(k), and comprehensive tuition fee reimbursement for graduate degrees."
    res = screen_advertisement_rules(text)
    assert res.has_warnings is False
    assert len(res.warnings) == 0


def test_guaranteed_placement_without_interview():
    """Synthetic test fixture: unrealistic guarantee without screening."""
    text = "Direct hiring with 100% placement guarantee without an interview upon form submission."
    res = screen_advertisement_rules(text)
    assert res.has_warnings is True
    assert "UNREALISTIC_GUARANTEE" in res.categories_flagged


def test_sensitive_banking_credentials_request():
    """Synthetic test fixture: demanding banking credentials."""
    text = "To process your employment contract, send online banking password and ATM card PIN to hr@temp.com."
    res = screen_advertisement_rules(text)
    assert res.has_warnings is True
    assert "SENSITIVE_INFO_HARVESTING" in res.categories_flagged
