from __future__ import annotations
import re
from .base import Extraction

PHONE_RE = re.compile(r"(\+49\s?\d[\d\s\-\/]{6,}\d|\b0\d[\d\s\-\/]{6,}\d\b)")

def extract_phone(text: str) -> Extraction:
    m = PHONE_RE.search(text)
    if not m:
        return Extraction("telephone", None, "missing", None, 0.0, ["phone_not_found"])
    val = re.sub(r"\s+", " ", m.group(1)).strip()
    return Extraction("telephone", val, "rule", m.group(1), 0.60, ["phone_regex_match"])
