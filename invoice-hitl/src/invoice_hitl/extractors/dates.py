from __future__ import annotations
import re
from datetime import date
from dateutil import parser as dateparser
from .base import Extraction
from ..utils import extract_lines, safe_lower
from ..validate import normalize_date_iso

DATE_RE = re.compile(r"\b(\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4})\b")

INVOICE_ANCHORS = ["rechnungsdatum", "datum", "invoice date"]
DUE_ANCHORS = ["fälligkeitsdatum", "faelligkeitsdatum", "zahlbar bis", "fällig bis", "due date"]

def _find_date_near_anchor(lines: list[str], anchors: list[str]) -> tuple[str, str] | None:
    # returns (value_raw, evidence_line)
    for i, ln in enumerate(lines):
        low = safe_lower(ln)
        if any(a in low for a in anchors):
            # try date on same line
            m = DATE_RE.search(ln)
            if m:
                return m.group(1), ln
            # else check next 2 lines
            for j in range(i+1, min(i+3, len(lines))):
                m2 = DATE_RE.search(lines[j])
                if m2:
                    return m2.group(1), lines[j]
    return None

def _parse_date(d: str) -> date | None:
    try:
        # German-style day-first
        dt = dateparser.parse(d, dayfirst=True)
        if not dt:
            return None
        return dt.date()
    except Exception:
        return None

def extract_invoice_date(text: str) -> Extraction:
    lines = extract_lines(text)
    found = _find_date_near_anchor(lines, INVOICE_ANCHORS)
    if not found:
        # fallback: first plausible date anywhere
        m = DATE_RE.search(text)
        if not m:
            return Extraction("invoice_date", None, "missing", None, 0.0, ["invoice_date_not_found"])
        raw = m.group(1)
        dt = _parse_date(raw)
        if not dt:
            return Extraction("invoice_date", None, "rule", raw, 0.25, ["invoice_date_parse_failed"])
        return Extraction("invoice_date", normalize_date_iso(dt), "rule", raw, 0.55, ["invoice_date_fallback_first_date"])

    raw, ev = found
    dt = _parse_date(raw)
    if not dt:
        return Extraction("invoice_date", None, "rule", ev, 0.25, ["invoice_date_parse_failed"])
    return Extraction("invoice_date", normalize_date_iso(dt), "rule", ev, 0.75, ["invoice_date_anchor_match"])

def extract_due_date(text: str) -> Extraction:
    lines = extract_lines(text)
    found = _find_date_near_anchor(lines, DUE_ANCHORS)
    if not found:
        return Extraction("due_date", None, "missing", None, 0.0, ["due_date_not_found"])

    raw, ev = found
    dt = _parse_date(raw)
    if not dt:
        return Extraction("due_date", None, "rule", ev, 0.25, ["due_date_parse_failed"])
    return Extraction("due_date", normalize_date_iso(dt), "rule", ev, 0.70, ["due_date_anchor_match"])
