"""
JobShield Rule-Based Warning Engine
Detects deterministic red-flag indicators (upfront fees, guaranteed hiring,
urgent payment pressure, sensitive financial disclosures) while explicitly
accounting for negation and legitimate employer disclaimers.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RuleWarning:
    """Represents a discrete rule violation detected in advertisement text."""
    category: str
    rule_id: str
    severity: str  # "HIGH", "MEDIUM", "INFO"
    matched_text: str
    explanation: str
    guidance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "matched_text": self.matched_text,
            "explanation": self.explanation,
            "guidance": self.guidance,
        }


@dataclass
class RuleEngineResult:
    """Aggregate result from rule engine screening."""
    has_warnings: bool
    warnings: List[RuleWarning] = field(default_factory=list)
    warning_count: int = 0
    categories_flagged: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_warnings": self.has_warnings,
            "warning_count": self.warning_count,
            "categories_flagged": self.categories_flagged,
            "warnings": [w.to_dict() for w in self.warnings],
        }


# Negation / Legitimate Disclaimer Patterns
# If any of these patterns occur within the same sentence/clause, the warning is suppressed.
NEGATION_PATTERNS = [
    r"\b(?:never|do\s+not|does\s+not|don'?t|won'?t|will\s+not)\s+(?:ask|charge|request|require|demand|collect)\b",
    r"\b(?:no|without\s+any)\s+(?:fees?|charges?|cost|deposit|payment)\b",
    r"\b(?:free\s+of\s+charge|at\s+no\s+cost|100%\s+free|completely\s+free)\b",
    r"\b(?:beware\s+of|scam\s+alert|fraud\s+warning|we\s+never\s+solicit)\b",
    r"\b(?:reimburse|reimbursement|covers?|covered\s+by\s+company)\b.*?\b(?:fees?|costs?|expenses?)\b",
    r"\b(?:tuition|certification|exam|visa)\s+reimbursement\b",
]

# Rule Definitions: (category, rule_id, severity, regex_pattern, explanation, guidance)
RULES = [
    # 1. Upfront Payment / Training Fee Requests
    (
        "UPFRONT_PAYMENT_DEMAND",
        "RULE_PAY_01",
        "HIGH",
        r"\b(?:pay|deposit|transfer|send|wire)\b[^\.\n;]{0,40}\b(?:registration|application|processing|interview|training|onboarding|equipment|laptop|security|clearance)\s+(?:fee|deposit|charge|amount|cost)\b",
        "The advertisement appears to request an upfront payment or deposit for application, interview, equipment, or training.",
        "Legitimate employers never charge candidates money to apply, interview, or undergo standard onboarding.",
    ),
    (
        "UPFRONT_PAYMENT_DEMAND",
        "RULE_PAY_02",
        "HIGH",
        r"\b(?:refundable|mandatory|compulsory)\s+(?:security\s+deposit|training\s+fee|onboarding\s+fee|registration\s+fee)\b",
        "A mandatory or 'refundable' deposit is specified.",
        "Fraudulent recruiters frequently claim deposits are 'refundable' after selection to trick candidates into transferring money.",
    ),
    (
        "UPFRONT_PAYMENT_DEMAND",
        "RULE_PAY_03",
        "HIGH",
        r"\b(?:purchase|buy)\b[^\.\n;]{0,35}\b(?:software|tools?|materials?|kit|uniform)\b[^\.\n;]{0,35}\b(?:from\s+us|from\s+our\s+vendor|prior\s+to\s+joining|to\s+qualify)\b",
        "Candidate is instructed to purchase software, tools, or uniforms from a designated vendor as a prerequisite.",
        "Reputable employers provide company tools directly or license them without demanding upfront applicant cash.",
    ),
    (
        "UNCONVENTIONAL_PAYMENT_METHOD",
        "RULE_PAY_04",
        "HIGH",
        r"\b(?:pay\s+via|send\s+via|payment\s+through)\b[^\.\n;]{0,30}\b(?:cryptocurrency|bitcoin|usdt|crypto|gift\s*card|apple\s+gift\s*card|steam\s*card|western\s+union|moneygram)\b",
        "The posting mentions payment through untraceable methods such as cryptocurrency, gift cards, or wire transfer services.",
        "Immediate red flag: legitimate companies never conduct payroll, fees, or hiring logistics via gift cards or crypto transfers.",
    ),

    # 2. Guaranteed Placement Without Legitimate Evaluation
    (
        "UNREALISTIC_GUARANTEE",
        "RULE_GUAR_01",
        "HIGH",
        r"\b(?:100%|direct|instant|guaranteed)\b[^\.\n;]{0,35}\b(?:selection|placement|job\s+offer|hiring|guarantee)\b[^\.\n;]{0,35}\b(?:without\s+(?:an?\s+)?interview|no\s+interview|without\s+screening)\b",
        "The ad claims guaranteed employment or selection without any interview or skill verification.",
        "Genuine hiring processes always require mutual evaluation. 'Guaranteed hiring without an interview' is a hallmark scam tactic.",
    ),

    # 3. High Pressure Urgency / Time Coercion
    (
        "URGENCY_PRESSURE",
        "RULE_URG_01",
        "MEDIUM",
        r"\b(?:act\s+immediately\s+or\s+lose|pay\s+within\s+(?:24|12|48)\s+hours|limited\s+slots?\s+pay\s+now|immediate\s+wire\s+transfer\s+required)\b",
        "Artificial urgency coercing the applicant to act or pay within a narrow, urgent deadline.",
        "Scammers create false panic to bypass candidates' critical thinking and diligence checks.",
    ),

    # 4. Premature Sensitive Financial / Identity Requests
    (
        "SENSITIVE_INFO_HARVESTING",
        "RULE_INFO_01",
        "HIGH",
        r"\b(?:send|submit|provide|email)\b[^\.\n;]{0,35}\b(?:bank\s+account\s+details|online\s+banking\s+password|netbanking|credit\s+card\s+number|cvv|pin\s+number|atm\s+card)\b",
        "The listing requests sensitive financial credentials (banking credentials, credit card details, CVV, or PIN) during application.",
        "Never disclose banking passwords, CVV, or full card details in response to a job advertisement or application form.",
    ),
    (
        "SENSITIVE_INFO_HARVESTING",
        "RULE_INFO_02",
        "MEDIUM",
        r"\b(?:attach|send|upload)\b[^\.\n;]{0,35}\b(?:scanned\s+copy\s+of\s+passport|original\s+certificates?|aadhaar\s+card\s+copy|ssn\s+card\s+copy)\b[^\.\n;]{0,35}\b(?:before\s+interview|for\s+shortlisting|to\s+apply)\b",
        "Demanding sensitive identity cards or original documents merely to submit an application or be shortlisted.",
        "Identity verification documents are only appropriate after a formal job offer has been vetted and accepted.",
    ),
]


def is_negated_or_disclaimer(sentence: str) -> bool:
    """Checks if a sentence or clause contains an explicit negation or anti-fraud disclaimer."""
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    return False


def split_into_clauses(text: str) -> List[str]:
    """Splits text into sentences and clause segments for local context checking."""
    # Split on sentence boundaries and bullet points
    raw_clauses = re.split(r"[\n\r\.\?!;•\*\t]+", text)
    clauses = [c.strip() for c in raw_clauses if len(c.strip()) > 5]
    return clauses


def screen_advertisement_rules(text: str) -> RuleEngineResult:
    """
    Evaluates advertisement text against high-confidence fraud rules.
    Respects negative contexts (e.g. 'We never ask for fees').
    """
    if not text or not isinstance(text, str):
        return RuleEngineResult(has_warnings=False, warnings=[], warning_count=0, categories_flagged=[])

    clauses = split_into_clauses(text)
    detected_warnings: List[RuleWarning] = []
    seen_rule_ids = set()

    for clause in clauses:
        # If this clause is a disclaimer (e.g. "We never charge fees"), skip it
        if is_negated_or_disclaimer(clause):
            continue

        for category, rule_id, severity, pattern, explanation, guidance in RULES:
            if rule_id in seen_rule_ids:
                continue

            match = re.search(pattern, clause, re.IGNORECASE)
            if match:
                matched_span = match.group(0)
                # Double-check that the immediate matched window isn't preceded by negation
                window_start = max(0, match.start() - 30)
                preceding_text = clause[window_start:match.start()]
                if re.search(r"\b(?:never|not|no|don'?t|won'?t)\b", preceding_text, re.IGNORECASE):
                    continue

                warning = RuleWarning(
                    category=category,
                    rule_id=rule_id,
                    severity=severity,
                    matched_text=matched_span,
                    explanation=explanation,
                    guidance=guidance,
                )
                detected_warnings.append(warning)
                seen_rule_ids.add(rule_id)

    unique_categories = sorted(list(set(w.category for w in detected_warnings)))

    return RuleEngineResult(
        has_warnings=len(detected_warnings) > 0,
        warnings=detected_warnings,
        warning_count=len(detected_warnings),
        categories_flagged=unique_categories,
    )
