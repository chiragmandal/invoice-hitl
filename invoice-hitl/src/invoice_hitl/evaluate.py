from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from rapidfuzz import fuzz

def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip() if ch.isalnum() or ch.isspace())

def exact_match(pred: str, gt: str) -> bool:
    return _norm(pred) == _norm(gt)

def fuzzy_score(pred: str, gt: str) -> float:
    return fuzz.token_set_ratio(_norm(pred), _norm(gt)) / 100.0

def amount_close(pred: str, gt: str, tol: float = 0.01) -> bool:
    # expects "12.34 EUR" or similar
    try:
        p = float(pred.split()[0])
        g = float(gt.split()[0]) if " " in gt else float(gt)
        return abs(p - g) <= tol
    except Exception:
        return False

@dataclass
class EvalRow:
    field: str
    ok: bool
    score: float

def evaluate_one(pred_fields: dict[str, Any], gt_fields: dict[str, Any]) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for field, gt in gt_fields.items():
        if gt is None or gt == "":
            continue
        pred = pred_fields.get(field, {}).get("value")
        if pred is None:
            rows.append(EvalRow(field, False, 0.0))
            continue

        if field in ("iban", "invoice_number", "invoice_date", "due_date", "telephone"):
            ok = exact_match(str(pred), str(gt))
            rows.append(EvalRow(field, ok, 1.0 if ok else 0.0))
        elif field == "total":
            ok = amount_close(str(pred), str(gt))
            rows.append(EvalRow(field, ok, 1.0 if ok else 0.0))
        else:
            score = fuzzy_score(str(pred), str(gt))
            ok = score >= 0.85
            rows.append(EvalRow(field, ok, score))
    return rows
