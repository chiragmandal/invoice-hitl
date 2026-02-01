from __future__ import annotations
import re
from datetime import date

def normalize_iban(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()

def validate_iban(iban: str) -> bool:
    # IBAN mod-97
    iban = normalize_iban(iban)
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", iban):
        return False
    # Move first 4 chars to end
    rearranged = iban[4:] + iban[:4]
    # Replace letters with numbers A=10..Z=35
    digits = ""
    for ch in rearranged:
        if ch.isdigit():
            digits += ch
        else:
            digits += str(ord(ch) - 55)
    # Compute mod 97
    mod = 0
    for c in digits:
        mod = (mod * 10 + int(c)) % 97
    return mod == 1

def normalize_date_iso(d: date) -> str:
    return d.isoformat()

def parse_eur_amount(s: str) -> float | None:
    # German format: 1.234,56
    s = s.strip()
    # remove spaces
    s = re.sub(r"\s+", "", s)
    # if comma exists, treat as decimal separator
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None
