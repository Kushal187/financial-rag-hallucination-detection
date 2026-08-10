"""Hallucination categories.

The primary signal is binary (``supported``) so the two verifiers can be scored
head-to-head; ``category`` records why. These descriptions are fed to the LLM judge.
"""

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

DEFAULT_CATEGORY = "unsupported_claim"
SUPPORTED_CATEGORY = "supported"
