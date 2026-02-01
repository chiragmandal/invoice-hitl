from __future__ import annotations

import os
import re
from typing import Any
from PIL import Image

from datasets import load_dataset
from rapidfuzz import fuzz

from .config import load_settings, TARGET_FIELDS
from .utils import ensure_dir, write_jsonl
from .preprocess import preprocess_pil
from .ocr import run_tesseract, OCRResult
from .llm import OllamaClient, build_llm_prompt
from .validate import validate_iban, parse_eur_amount
from .confidence import combine_confidence
from .hitl import route_field, route_invoice
from .evaluate import evaluate_one

from .extractors import (
    extract_iban,
    extract_invoice_date,
    extract_due_date,
    extract_total,
    extract_invoice_number,
    extract_phone,
    extract_company_name_heuristic,
    extract_company_address_heuristic,
    extract_bank_name_heuristic,
)

# ----------------------------
# Guardrails / Sanitizers
# ----------------------------
_PLACEHOLDER_PAT = re.compile(r"\[[^\]]+\]")  # e.g. [Adresse]
_BAD_TOKENS = ["pos", "gesamtbetrag", "rechnung", "ust", "umsatzsteuer", "summe", "kundennr", "rechnungsnr"]


def _looks_like_placeholder(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    if _PLACEHOLDER_PAT.search(s):
        return True
    low = s.lower()
    # common template placeholders
    if "unternehmensname" in low or low == "adresse" or "bankname" in low:
        return True
    return False


def _too_long_for_entity(s: str, max_chars: int = 140) -> bool:
    return len(s.strip()) > max_chars


def _looks_like_ocr_blob(s: str) -> bool:
    low = s.lower()
    hits = sum(1 for t in _BAD_TOKENS if t in low)
    return hits >= 2


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _evidence_matches(text: str, evidence: str) -> bool:
    # allow whitespace/punctuation drift
    t = _normalize_spaces(text).lower()
    e = _normalize_spaces(evidence).lower()
    if not e:
        return False
    if e in t:
        return True
    # fuzzy fallback (works well for OCR minor differences)
    return fuzz.partial_ratio(e, t) >= 92


def _reject_garbage_entity(extr_obj: Any, reason: str) -> None:
    """
    extr_obj is your ExtractResult-like object with:
      .value, .evidence, .raw_score, .method, .reasons
    """
    extr_obj.reasons.append(reason)
    extr_obj.value = None
    extr_obj.evidence = None
    extr_obj.raw_score = min(getattr(extr_obj, "raw_score", 0.0), 0.2)
    extr_obj.method = "missing"


def _find_image(sample: dict[str, Any]) -> Image.Image:
    # Common HF patterns: "image" column is PIL Image
    for k, v in sample.items():
        if hasattr(v, "size") and hasattr(v, "mode"):
            return v  # PIL image
    raise ValueError(
        "Could not find an image column in dataset sample. Inspect dataset columns and update loader."
    )


def _find_labels(sample: dict[str, Any]) -> dict[str, Any]:
    # Try common keys
    for key in ["labels", "label", "annotation", "annotations", "ground_truth", "gt"]:
        v = sample.get(key)
        if isinstance(v, dict):
            return v
    return {}


def _stringify_labels(labels: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in labels.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (int, float, str)):
            out[k] = str(v)
        else:
            out[k] = str(v)
    return out


def _ocr_evidence_conf(ocr: OCRResult, evidence: str | None) -> float:
    if not evidence:
        return ocr.avg_conf
    ev_low = evidence.lower()
    matched = [t.conf for t in ocr.tokens if t.text.lower() in ev_low]
    if matched:
        return sum(matched) / len(matched)
    return ocr.avg_conf


def main() -> None:
    settings = load_settings()
    ensure_dir(settings.debug_dir)
    ensure_dir(os.path.dirname(settings.output_path) or ".")

    ds = load_dataset(settings.hf_dataset_id, split=settings.hf_split)
    limit = min(settings.limit, len(ds))

    # LLM ON
    llm = OllamaClient(settings.ollama_host, settings.ollama_model) if settings.use_llm else None

    outputs: list[dict[str, Any]] = []
    eval_rows_all: list[dict[str, Any]] = []

    for idx in range(limit):
        sample = ds[idx]
        img = _find_image(sample)
        labels = _stringify_labels(_find_labels(sample))
        invoice_id = str(sample.get("id", idx))

        pre = preprocess_pil(img, max_width=settings.max_width, do_threshold=settings.do_threshold)

        # Save debug preprocessed image (optional)
        try:
            pre.save(os.path.join(settings.debug_dir, f"{invoice_id}_pre.png"))
        except Exception:
            pass

        ocr = run_tesseract(pre, lang=settings.ocr_lang)
        with open(os.path.join(settings.debug_dir, f"{invoice_id}_ocr.txt"), "w", encoding="utf-8") as f:
            f.write(ocr.full_text)

        text = ocr.full_text

        # ---------- RULE EXTRACTION ----------
        extr: dict[str, Any] = {}
        extr["iban"] = extract_iban(text)
        extr["invoice_date"] = extract_invoice_date(text)
        extr["due_date"] = extract_due_date(text)
        extr["total"] = extract_total(text)
        extr["invoice_number"] = extract_invoice_number(text)
        extr["telephone"] = extract_phone(text)

        extr["company_name"] = extract_company_name_heuristic(text)
        extr["company_address"] = extract_company_address_heuristic(text)
        extr["bank_name"] = extract_bank_name_heuristic(text)

        # ---------- HARD GUARDRAILS (pre-LLM) ----------
        # Reject OCR-blob and placeholders for entity-like fields
        for f in ["company_name", "company_address", "bank_name"]:
            v = extr[f].value
            if isinstance(v, str):
                if _looks_like_placeholder(v):
                    _reject_garbage_entity(extr[f], "rejected_placeholder")
                elif _looks_like_ocr_blob(v):
                    _reject_garbage_entity(extr[f], "rejected_ocr_blob")
                elif _too_long_for_entity(v):
                    _reject_garbage_entity(extr[f], "rejected_too_long")

        # Reject totals that look like date fragments like "29.07 EUR"
        tv = extr["total"].value
        if isinstance(tv, str):
            if re.fullmatch(r"\d{1,2}\.\d{1,2}\s*EUR", tv.strip()):
                extr["total"].reasons.append("rejected_total_looks_like_date")
                extr["total"].value = None
                extr["total"].evidence = None
                extr["total"].raw_score = 0.2
                extr["total"].method = "missing"

        # Invoice number must contain at least one digit
        invv = extr["invoice_number"].value
        if isinstance(invv, str):
            if not any(ch.isdigit() for ch in invv):
                extr["invoice_number"].reasons.append("rejected_invoice_number_no_digit")
                extr["invoice_number"].value = None
                extr["invoice_number"].evidence = None
                extr["invoice_number"].raw_score = 0.2
                extr["invoice_number"].method = "missing"

        # ---------- OPTIONAL LLM FALLBACK ----------
        llm_used = False
        llm_payload: dict[str, Any] | None = None

        # IMPORTANT: track "untrusted" per-field, not globally
        llm_untrusted_fields: set[str] = set()

        if llm is not None:
            # Only call LLM for fields still missing AFTER guardrails
            need_llm_fields = [f for f in ["company_name", "company_address", "bank_name"] if extr[f].value is None]

            if need_llm_fields:
                prompt = build_llm_prompt(text[:8000])  # keep prompt bounded
                llm_payload = llm.generate_json(prompt)
                llm_used = True

                if llm_payload is None:
                    # We tried and got nothing => mark requested fields as untrusted attempts
                    llm_untrusted_fields.update(need_llm_fields)
                else:
                    for f in need_llm_fields:
                        obj = llm_payload.get(f, None)
                        if not isinstance(obj, dict):
                            llm_untrusted_fields.add(f)
                            extr[f].reasons.append("llm_bad_payload_shape")
                            continue

                        val = obj.get("value")
                        ev = obj.get("evidence")

                        if val is None or ev is None:
                            # LLM explicitly says null -> keep missing (not "untrusted")
                            continue

                        if not isinstance(ev, str) or not ev.strip():
                            llm_untrusted_fields.add(f)
                            extr[f].reasons.append("llm_missing_evidence")
                            continue

                        if not _evidence_matches(text, ev):
                            llm_untrusted_fields.add(f)
                            extr[f].reasons.append("llm_evidence_mismatch")
                            continue

                        val_s = str(val).strip()

                        # Reject placeholders/blobs/too-long even from LLM
                        if _looks_like_placeholder(val_s):
                            llm_untrusted_fields.add(f)
                            extr[f].reasons.append("llm_rejected_placeholder")
                            continue
                        if _looks_like_ocr_blob(val_s):
                            llm_untrusted_fields.add(f)
                            extr[f].reasons.append("llm_rejected_ocr_blob")
                            continue
                        if _too_long_for_entity(val_s):
                            llm_untrusted_fields.add(f)
                            extr[f].reasons.append("llm_rejected_too_long")
                            continue

                        # accept LLM suggestion
                        extr[f].value = val_s
                        extr[f].method = "llm"
                        extr[f].evidence = ev.strip()
                        extr[f].raw_score = 0.60
                        extr[f].reasons.append("llm_fallback_used")

        # ---------- VALIDATION FLAGS ----------
        valid_map: dict[str, bool] = {}
        conflict_map: dict[str, bool] = {}

        # IBAN validity
        iban_val = extr["iban"].value
        valid_map["iban"] = bool(iban_val) and validate_iban(iban_val)
        conflict_map["iban"] = False

        # Total validity
        total_val = extr["total"].value
        if total_val and isinstance(total_val, str):
            amt = parse_eur_amount(total_val.split()[0])
            valid_map["total"] = (amt is not None) and (amt > 0)
        else:
            valid_map["total"] = False
        conflict_map["total"] = "multiple_total_candidates" in extr["total"].reasons

        # Dates validity
        valid_map["invoice_date"] = extr["invoice_date"].value is not None
        valid_map["due_date"] = extr["due_date"].value is not None
        conflict_map["invoice_date"] = False
        conflict_map["due_date"] = False

        # Invoice number validity
        valid_map["invoice_number"] = extr["invoice_number"].value is not None
        conflict_map["invoice_number"] = False

        # Entities + phone: weak validation
        for f in ["company_name", "company_address", "bank_name", "telephone"]:
            valid_map[f] = extr[f].value is not None
            conflict_map[f] = False

        # ---------- CONFIDENCE + ROUTING ----------
        field_outputs: dict[str, Any] = {}
        field_routes: dict[str, bool] = {}
        missing_count = 0

        for field in TARGET_FIELDS:
            e = extr[field]
            missing = e.value is None
            if missing:
                missing_count += 1

            ev_conf = _ocr_evidence_conf(ocr, e.evidence)

            conf_res = combine_confidence(
                settings=settings,
                raw_score=e.raw_score,
                ocr_avg_conf=ev_conf,
                valid=valid_map.get(field, False),
                method=e.method,
                conflict=conflict_map.get(field, False),
                # FIX: only mark the specific field untrusted (not global)
                untrusted_llm=(field in llm_untrusted_fields),
            )

            extra_reasons = list(e.reasons) + conf_res.reasons
            validation_failed = (
                not valid_map.get(field, False)
                and (field in ["iban", "total", "invoice_number", "invoice_date", "due_date"])
            )

            decision = route_field(
                settings=settings,
                field=field,
                conf=conf_res.conf,
                missing=missing,
                validation_failed=validation_failed,
                extra_reasons=extra_reasons,
            )

            field_outputs[field] = {
                "value": e.value,
                "confidence": round(conf_res.conf, 4),
                "method": e.method,
                "evidence": e.evidence,
                "reasons": decision.reasons,
            }
            field_routes[field] = decision.route_field

        invoice_route, invoice_reasons = route_invoice(settings, field_routes, ocr.avg_conf, missing_count)

        out = {
            "invoice_id": invoice_id,
            "avg_ocr_conf": round(ocr.avg_conf, 4),
            "fields": field_outputs,
            "route_to_human": field_routes,
            "invoice_route_to_human": invoice_route,
            "invoice_route_reasons": invoice_reasons,
            "llm_used": llm_used,
        }
        outputs.append(out)

        # ---------- EVALUATION (only if labels exist for those fields) ----------
        if labels:
            gt = {k: labels.get(k) for k in TARGET_FIELDS if k in labels}
            if gt:
                eval_rows = evaluate_one(out["fields"], gt)
                for r in eval_rows:
                    eval_rows_all.append({
                        "invoice_id": invoice_id,
                        "field": r.field,
                        "ok": r.ok,
                        "score": r.score,
                        "routed": bool(field_routes.get(r.field, False)),
                        "confidence": out["fields"][r.field]["confidence"],
                    })

    write_jsonl(settings.output_path, outputs)

    # Save evaluation summary
    if eval_rows_all:
        eval_path = os.path.join(os.path.dirname(settings.output_path) or ".", "eval_rows.jsonl")
        write_jsonl(eval_path, eval_rows_all)

        ok_count = sum(1 for r in eval_rows_all if r["ok"])
        total = len(eval_rows_all)
        routed = sum(1 for r in eval_rows_all if r["routed"])
        print(f"[EVAL] rows={total} ok={ok_count} acc={ok_count/total:.3f} routed={routed} routed_rate={routed/total:.3f}")


if __name__ == "__main__":
    main()
