from __future__ import annotations

import re
from .base import Extraction
from ..utils import extract_lines, safe_lower
from ..validate import parse_eur_amount

# Stronger anchors (total / amount due)
TOTAL_ANCHORS_STRONG = [
    "gesamtbetrag", "rechnungsbetrag", "endbetrag", "zu zahlen", "zahlbetrag", "betrag fällig",
    "invoice total", "amount due", "total due", "grand total", "total amount",
    "rechnungssumme", "summe brutto", "gesamt brutto"
]

# Weaker anchors (may include subtotals etc.)
TOTAL_ANCHORS_WEAK = ["gesamt", "summe", "brutto", "total", "end", "betrag"]

CURRENCY_MARKERS = ["€", "eur", "euro"]

# Matches amounts like:
# 1.234,56  | 1234,56 | 1234.56 | 2.072,00
AMOUNT_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:\.\d{3})*,\d{2}|\d{1,6},\d{2}|\d{1,6}\.\d{2})(?!\d)"
)

# Detect date fragments like 29.07 or 11.05 or 02.12
DATE_FRAGMENT = re.compile(r"^(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])$")

def _has_currency(line_low: str) -> bool:
    return any(m in line_low for m in CURRENCY_MARKERS)

def _is_date_like_amount(raw: str, line_low: str) -> bool:
    """
    Reject '29.07' being parsed as 29.07 EUR (date fragment).
    Keep if currency marker exists AND line is clearly a total line.
    """
    s = raw.strip()
    if DATE_FRAGMENT.match(s):
        # allow ONLY if currency marker AND strong anchors present
        if _has_currency(line_low) and any(a in line_low for a in TOTAL_ANCHORS_STRONG):
            return False
        return True
    return False

def _score_line(line_low: str) -> float:
    """
    Higher score = more likely to be the final total.
    """
    score = 0.0
    if any(a in line_low for a in TOTAL_ANCHORS_STRONG):
        score += 2.0
    if any(a in line_low for a in TOTAL_ANCHORS_WEAK):
        score += 0.8
    if _has_currency(line_low):
        score += 1.2
    # penalize typical subtotal signals
    if "netto" in line_low:
        score -= 0.4
    if "zwischensumme" in line_low or "subtotal" in line_low:
        score -= 0.6
    return score

def extract_total(text: str) -> Extraction:
    lines = extract_lines(text)

    candidates: list[tuple[float, str, float]] = []  # (value, evidence_line, score)

    # 1) Anchor-guided candidates
    for ln in lines:
        low = safe_lower(ln)
        if any(a in low for a in TOTAL_ANCHORS_STRONG) or any(a in low for a in TOTAL_ANCHORS_WEAK):
            for m in AMOUNT_RE.finditer(ln):
                raw = m.group(1)
                if _is_date_like_amount(raw, low):
                    continue
                val = parse_eur_amount(raw)
                if val is not None and val > 0:
                    candidates.append((val, ln, _score_line(low)))

    if candidates:
        # Sort:
        #  - Prefer higher line score (strong anchors + currency)
        #  - Then prefer higher amount
        candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
        best_val, best_ev, _ = candidates[0]
        reasons = ["total_anchor_match"]
        if len(candidates) > 1:
            reasons.append("multiple_total_candidates")
        return Extraction("total", f"{best_val:.2f} EUR", "rule", best_ev, 0.82, reasons)

    # 2) Fallback: scan bottom third for currency amounts and take the max
    # This avoids picking dates/phones/random IDs.
    bottom_start = int(len(lines) * 0.66)
    bottom_lines = lines[bottom_start:] if lines else []

    fb_candidates: list[tuple[float, str, float]] = []
    for ln in bottom_lines:
        low = safe_lower(ln)
        # require currency marker in fallback mode to avoid phone/date tokens
        if not _has_currency(low):
            continue
        for m in AMOUNT_RE.finditer(ln):
            raw = m.group(1)
            if _is_date_like_amount(raw, low):
                continue
            val = parse_eur_amount(raw)
            if val is not None and val > 0:
                fb_candidates.append((val, ln, _score_line(low)))

    if fb_candidates:
        fb_candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
        best_val, best_ev, _ = fb_candidates[0]
        return Extraction("total", f"{best_val:.2f} EUR", "rule", best_ev, 0.60, ["total_fallback_bottom_currency_max"])

    # 3) No total found
    return Extraction("total", None, "missing", None, 0.0, ["total_not_found"])
