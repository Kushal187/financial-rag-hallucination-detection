"""Hallucination category taxonomy for the detection stage.

The primary signal is binary — ``supported: bool`` (is the answer grounded in the
retrieved evidence?) — so the LLM judge and Member 1's rule-based verifier can be
scored head-to-head. ``category`` is the analytical layer: *why* an unsupported
answer is unsupported. Five categories, kept disjoint so a judge doesn't have three
buckets to put the same case in (``fabrication`` is folded into ``out_of_context``).
"""

__all__ = ["CATEGORY_LIST", "CATEGORIES", "DEFAULT_CATEGORY", "SUPPORTED_CATEGORY"]

# Order matters only for display; categories are mutually exclusive per verdict.
CATEGORY_LIST = [
    "numeric_error",
    "entity_error",
    "unsupported_claim",
    "out_of_context",
    "abstention",
]

CATEGORIES = {
    "numeric_error": (
        "The answer is a number but its value, units, or underlying arithmetic is "
        "wrong relative to the evidence."
    ),
    "entity_error": (
        "The answer attributes a fact to the wrong entity — a different year, company, "
        "line item, or unit than the evidence states."
    ),
    "unsupported_claim": (
        "A specific claim in the answer is not directly backed by any of the evidence."
    ),
    "out_of_context": (
        "The answer is unrelated to the question or evidence, or invents information "
        "not present anywhere (fabrication)."
    ),
    "abstention": (
        "The model declined to answer or said it could not determine the answer. This "
        "is NOT a hallucination — it is the safe failure mode."
    ),
}

DEFAULT_CATEGORY = "unsupported_claim"      # used when an unsupported verdict has no category
SUPPORTED_CATEGORY = "supported"            # category label when supported == True
