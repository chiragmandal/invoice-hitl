from __future__ import annotations
import re
from .base import Extraction
from ..utils import extract_lines, safe_lower, normalize_whitespace

LEGAL_SUFFIXES = ["gmbh", "ag", "ug", "kg", "gbr", "e.k", "ek", "mbh"]
BANK_HINTS = ["bank", "sparkasse", "volksbank", "iban", "bic", "kreditinstitut"]
PLZ_RE = re.compile(r"\b\d{5}\b")

def extract_company_name_heuristic(text: str) -> Extraction:
    lines = extract_lines(text)
    # top 8 lines often include header/company
    top = lines[:8]
    for ln in top:
        low = safe_lower(ln)
        if any(suf in low for suf in LEGAL_SUFFIXES):
            return Extraction("company_name", normalize_whitespace(ln), "rule", ln, 0.55, ["company_name_legal_suffix_header"])

    # fallback: longest of top lines (but not too long)
    if top:
        best = max(top, key=lambda s: len(s))
        if 6 <= len(best) <= 70:
            return Extraction("company_name", normalize_whitespace(best), "rule", best, 0.35, ["company_name_fallback_long_header"])
    return Extraction("company_name", None, "missing", None, 0.0, ["company_name_not_found"])

def extract_company_address_heuristic(text: str) -> Extraction:
    lines = extract_lines(text)
    # find line containing PLZ; include neighboring line
    for i, ln in enumerate(lines[:30]):
        if PLZ_RE.search(ln):
            parts = [ln]
            if i > 0:
                parts.insert(0, lines[i-1])
            addr = normalize_whitespace(" | ".join(parts))
            return Extraction("company_address", addr, "rule", addr, 0.50, ["address_plz_found"])
    return Extraction("company_address", None, "missing", None, 0.0, ["address_not_found"])

def extract_bank_name_heuristic(text: str) -> Extraction:
    lines = extract_lines(text)
    for ln in lines:
        low = safe_lower(ln)
        if any(h in low for h in BANK_HINTS):
            # likely bank line; attempt to capture bank-like phrase
            if "iban" in low or "bic" in low:
                continue
            if 4 <= len(ln) <= 80:
                return Extraction("bank_name", normalize_whitespace(ln), "rule", ln, 0.45, ["bank_hint_line"])
    return Extraction("bank_name", None, "missing", None, 0.0, ["bank_name_not_found"])
