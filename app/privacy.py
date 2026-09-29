"""
privacy.py
Best-effort redaction of obvious personal identifiers from text BEFORE it is
sent to a cloud LLM. Regex-based, so it catches common patterns (emails,
phone numbers, card/SSN-like numbers, API keys) but is not a guarantee.
For truly sensitive material, use LLM_PROVIDER=ollama so nothing leaves
your machine at all.
"""

import re

_PATTERNS = [
    (re.compile(r"\b(?:sk|ghp|gsk|AKIA)[A-Za-z0-9_\-]{16,}"), "[SECRET]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD]"),
    (re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d"), "[PHONE]"),
]


def redact(text):
    for pattern, label in _PATTERNS:
        text = pattern.sub(label, text)
    return text
