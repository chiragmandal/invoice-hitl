from __future__ import annotations

import re
from .base import Extraction
from ..utils import extract_lines, safe_lower

# Anchors (German + English + common OCR variants)
ANCHORS = [
    "rechnungsnummer", "rechnungs nr", "rechnung nr", "rechnung-nr", "rechnungnr", "rg-nr", "rg nr", "rg.-nr",
    "re-nr", "re nr", "rechn.-nr", "rechn nr", "rechnungsnr", "rechnungs-nr", "rechnungsnr."
    "invoice no", "invoice number", "inv no", "inv#", "invoice #", "invoice nr",
    "belegnummer", "beleg nr", "beleg-nr", "belegnr",
]

# Lines/phrases that often appear near "numbers" but are NOT invoice numbers
NEGATIVE_CONTEXT = [
    "steuernummer", "stnr", "ust-id", "ustid", "vat", "mwst", "tax", "kundennummer", "kunden nr", "customer no",
    "plz", "postleitzahl", "iban", "bic", "swift", "konto", "kontonummer", "telefon", "tel", "fax",
]

# Candidate invoice id token (start with alnum; allow -, /; length >= 4)
# Example: RE-2017-MAI-11-0003, 2021-00012, INV/12345
ID_RE = re.compile(r"\b([A-Z0-9][A-Z0-9\-\/]{3,})\b", re.IGNORECASE)

# Patterns like "Rechnung #123" or "Invoice # 123"
HASH_RE = re.compile(r"(?:rechnung|invoice)\s*#\s*([A-Z0-9][A-Z0-9\-\/]{2,})", re.IGNORECASE)

# Quick date detector to avoid picking "29.07" or "11.05" etc.
DATE_LIKE = re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b")

def _has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)

def _looks_like_date_token(s: str) -> bool:
    return bool(DATE_LIKE.fullmatch(s.strip()))

def _bad_by_context(line_low: str) -> bool:
    return any(neg in line_low for neg in NEGATIVE_CONTEXT)

def _clean_token(tok: str) -> str:
    return tok.strip().strip(".,;:()[]{}")

def _is_plz_like(tok: str) -> bool:
    # German PLZ 5 digits
    t = tok.strip()
    return bool(re.fullmatch(r"\d{5}", t))

def _is_too_generic(tok: str) -> bool:
    # reject overly generic OCR header words
    low = tok.lower()
    return low in {"rechnung", "invoice", "angebot", "order", "nr", "no"}

def _pick_best_from_line(line: str) -> str | None:
    """
    Extract invoice id from a single anchor line.
    Priority:
      1) after ':' / '='
      2) after '#'
      3) best ID_RE match in the line
    """
    # 1) after ':' or '='
    for sep in [":", "=", "#"]:
        if sep in line:
            tail = line.split(sep, 1)[1].strip()
            m = ID_RE.search(tail)
            if m:
                cand = _clean_token(m.group(1))
                return cand

    # 2) explicit hash pattern
    mh = HASH_RE.search(line)
    if mh:
        return _clean_token(mh.group(1))

    # 3) any token in line
    m2 = ID_RE.search(line)
    if m2:
        return _clean_token(m2.group(1))

    return None

def _valid_invoice_id(tok: str, line_low: str) -> bool:
    if not tok:
        return False
    if _is_too_generic(tok):
        return False
    if not _has_digit(tok):
        return False
    if _looks_like_date_token(tok):
        return False
    if _is_plz_like(tok):
        return False
    # avoid selecting tax/customer/etc if line context indicates those
    if _bad_by_context(line_low):
        return False
    # avoid extremely short pure numbers like "12" (too ambiguous)
    if re.fullmatch(r"\d{1,3}", tok):
        return False
    return True

def extract_invoice_number(text: str) -> Extraction:
    lines = extract_lines(text)

    # 1) Anchor-based extraction (most reliable)
    for ln in lines:
        low = safe_lower(ln)
        if any(a in low for a in ANCHORS):
            cand = _pick_best_from_line(ln)
            if cand and _valid_invoice_id(cand, low):
                return Extraction("invoice_number", cand, "rule", ln, 0.82, ["invoice_number_anchor_match"])

    # 2) Secondary pattern: "Rechnung Nr. <id>" / "Invoice No <id>" across line
    # This avoids the old "first ID-like token in full text" mistake.
    pat = re.compile(
        r"(?:rechnung|invoice)\s*(?:nr|nr\.|no|no\.|number|#)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-\/]{3,})",
        re.IGNORECASE
    )
    for ln in lines:
        low = safe_lower(ln)
        m = pat.search(ln)
        if m:
            cand = _clean_token(m.group(1))
            if _valid_invoice_id(cand, low):
                return Extraction("invoice_number", cand, "rule", ln, 0.75, ["invoice_number_pattern_match"])

    # 3) If nothing, return missing (NO global-ID fallback; that caused "Paul", "logoipsum", etc.)
    return Extraction("invoice_number", None, "missing", None, 0.0, ["invoice_number_not_found"])
