from __future__ import annotations
import re
from .base import Extraction
from ..validate import validate_iban, normalize_iban

IBAN_RE = re.compile(r"\bDE\d{2}(?:\s?\d{4}){4,5}\b", re.IGNORECASE)

def extract_iban(text: str) -> Extraction:
    m = IBAN_RE.search(text)
    if not m:
        return Extraction("iban", None, "missing", None, 0.0, ["iban_not_found"])

    raw = m.group(0)
    iban = normalize_iban(raw)
    ok = validate_iban(iban)
    if ok:
        return Extraction("iban", iban, "rule", raw, 0.95, ["iban_regex_match", "iban_checksum_valid"])
    return Extraction("iban", iban, "rule", raw, 0.40, ["iban_regex_match", "iban_checksum_failed"])
