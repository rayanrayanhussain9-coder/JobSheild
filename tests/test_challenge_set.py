"""
JobShield Challenge Set Test Suite
Evaluates nuanced, adversarial, and edge-case job advertisement scenarios:
1. Fee disclaimers embedded in genuine postings.
2. Camouflaged upfront fees.
3. Vague low-content ads.
4. Unconventional payment methods (cryptocurrency, gift cards).
5. Contradictory signals between deterministic rules and diffuse statistical vocabulary.
"""
import pytest
from src.predictor import JobShieldPredictor, JobAdvertisementInput

CHALLENGE_CASES = [
    {
        "id": "CHALLENGE_01_DISCLAIMER_PRESERVATION",
        "name": "Legitimate Ad with Anti-Scam Disclaimer",
        "description": "Standard corporate ad that contains the words 'fee' and 'scam' solely in an anti-fraud notice.",
        "input": JobAdvertisementInput(
            title="Senior Java Developer",
            company_name="Enterprise Systems Inc.",
            description=(
                "Enterprise Systems Inc. is hiring a Senior Java Developer for our core payment services team. "
                "Candidates will design microservices, ensure PCI-DSS compliance, and participate in on-call rotations. "
                "NOTICE: Enterprise Systems Inc. never charges any application, training, or recruitment fee. "
                "All hiring communications occur exclusively through official @enterprisesystems.com emails."
            ),
            requirements="7+ years Java, Spring Boot, PostgreSQL, Docker.",
            benefits="Medical, dental, vision, 401(k) matching, flexible PTO.",
        ),
        "expected_rules_fired": False,
        "expected_prediction": "LOW_RISK_PATTERN",
    },
    {
        "id": "CHALLENGE_02_CAMOUFLAGED_EQUIPMENT_FEE",
        "name": "Camouflaged Workstation Purchase Demand",
        "description": "Scam listing cloaked in formal language demanding upfront software/equipment purchase.",
        "input": JobAdvertisementInput(
            title="Remote Administrative Specialist",
            company_name="Global Operations Direct",
            description=(
                "We are seeking a reliable Remote Administrative Specialist to handle customer data and documentation. "
                "Prior to joining, candidates must purchase software and materials from our vendor to qualify for onboarding. "
                "This ensures standardized security configurations."
            ),
            requirements="Good communication skills, attention to detail.",
            benefits="Flexible hours, competitive hourly compensation.",
        ),
        "expected_rules_fired": True,
        "expected_rule_category": "UPFRONT_PAYMENT_DEMAND",
    },
    {
        "id": "CHALLENGE_03_CRYPTO_WIRE_TRANSFER",
        "name": "Cryptocurrency Background Check Fee",
        "description": "Unconventional payment demand via crypto for background check verification.",
        "input": JobAdvertisementInput(
            title="Virtual Assistant",
            company_name="Apex Online Works",
            description=(
                "Exciting work-from-home position for immediate hire! Earn top hourly rates assisting executives. "
                "Selected applicants must pay via cryptocurrency for their initial background check clearance."
            ),
            requirements="Internet access, basic computer literacy.",
            benefits="Weekly payouts.",
        ),
        "expected_rules_fired": True,
        "expected_rule_category": "UNCONVENTIONAL_PAYMENT_METHOD",
    },
    {
        "id": "CHALLENGE_04_VAGUE_MINIMAL_POSTING",
        "name": "Extremely Vague Job Posting",
        "description": "Very short and vague posting that passes minimum character threshold but has low detail.",
        "input": JobAdvertisementInput(
            title="Online Worker Needed",
            company_name="FastWork",
            description=(
                "Immediate opening for online workers. Need someone to help with daily tasks online. "
                "Flexible hours, good pay, start immediately. Please apply with contact number."
            ),
            requirements="",
            benefits="",
        ),
        "expected_rules_fired": False,  # No explicit fee, but missing critical company & requirements info
        "expected_missing_info": True,
    },
]


@pytest.fixture(scope="module")
def predictor():
    return JobShieldPredictor()


@pytest.mark.parametrize("case", CHALLENGE_CASES, ids=[c["id"] for c in CHALLENGE_CASES])
def test_challenge_case_execution(predictor, case):
    """Executes documented challenge case and verifies expected safety and rule behavior."""
    res = predictor.predict(case["input"])

    assert res.status == "ANALYSIS_COMPLETE"

    if "expected_rules_fired" in case:
        assert res.rule_results["has_warnings"] is case["expected_rules_fired"]

    if "expected_rule_category" in case:
        assert case["expected_rule_category"] in res.rule_results["categories_flagged"]

    if "expected_prediction" in case:
        assert res.model_prediction == case["expected_prediction"]

    if "expected_missing_info" in case and case["expected_missing_info"]:
        assert len(res.missing_info_observations) > 0
