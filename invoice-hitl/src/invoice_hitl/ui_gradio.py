from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Tuple, List

import gradio as gr
import pandas as pd
from PIL import Image

from .config import load_settings
from .preprocess import preprocess_pil
from .ocr import run_tesseract
from .llm import OllamaClient, build_llm_prompt
from .validate import validate_iban, parse_eur_amount
from .confidence import combine_confidence
from .hitl import route_field, route_invoice
from .utils import ensure_dir

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
# Guardrails / Sanitizers (same behavior as run.py)
# ----------------------------
import re
from rapidfuzz import fuzz

_PLACEHOLDER_PAT = re.compile(r"\[[^\]]+\]")  # e.g. [Adresse]
_BAD_TOKENS = ["pos", "gesamtbetrag", "rechnung", "ust", "umsatzsteuer", "summe", "kundennr", "rechnungsnr"]

def _looks_like_placeholder(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    if _PLACEHOLDER_PAT.search(s):
        return True
    low = s.lower()
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
    t = _normalize_spaces(text).lower()
    e = _normalize_spaces(evidence).lower()
    if not e:
        return False
    if e in t:
        return True
    return fuzz.partial_ratio(e, t) >= 92

def _reject_garbage_entity(extr_obj: Any, reason: str) -> None:
    extr_obj.reasons.append(reason)
    extr_obj.value = None
    extr_obj.evidence = None
    extr_obj.raw_score = min(getattr(extr_obj, "raw_score", 0.0), 0.2)
    extr_obj.method = "missing"

def _ocr_evidence_conf(ocr, evidence: str | None) -> float:
    if not evidence:
        return ocr.avg_conf
    ev_low = evidence.lower()
    matched = [t.conf for t in ocr.tokens if t.text.lower() in ev_low]
    if matched:
        return sum(matched) / len(matched)
    return ocr.avg_conf


def run_extraction_ui(img: Image.Image) -> Tuple[str, pd.DataFrame, str]:
    """
    Returns:
      - pretty JSON (string)
      - table dataframe
      - badge markdown (string)
    """
    settings = load_settings()
    ensure_dir(settings.debug_dir)

    # LLM
    llm = OllamaClient(settings.ollama_host, settings.ollama_model) if settings.use_llm else None

    # Preprocess + OCR
    pre = preprocess_pil(img, max_width=settings.max_width, do_threshold=settings.do_threshold)
    ocr = run_tesseract(pre, lang=settings.ocr_lang)
    text = ocr.full_text

    # --- Rule extraction ---
    extr: Dict[str, Any] = {}
    extr["iban"] = extract_iban(text)
    extr["invoice_date"] = extract_invoice_date(text)
    extr["due_date"] = extract_due_date(text)
    extr["total"] = extract_total(text)
    extr["invoice_number"] = extract_invoice_number(text)
    extr["telephone"] = extract_phone(text)

    extr["company_name"] = extract_company_name_heuristic(text)
    extr["company_address"] = extract_company_address_heuristic(text)
    extr["bank_name"] = extract_bank_name_heuristic(text)

    # --- Guardrails pre-LLM ---
    for f in ["company_name", "company_address", "bank_name"]:
        v = extr[f].value
        if isinstance(v, str):
            if _looks_like_placeholder(v):
                _reject_garbage_entity(extr[f], "rejected_placeholder")
            elif _looks_like_ocr_blob(v):
                _reject_garbage_entity(extr[f], "rejected_ocr_blob")
            elif _too_long_for_entity(v):
                _reject_garbage_entity(extr[f], "rejected_too_long")

    tv = extr["total"].value
    if isinstance(tv, str):
        if re.fullmatch(r"\d{1,2}\.\d{1,2}\s*EUR", tv.strip()):
            extr["total"].reasons.append("rejected_total_looks_like_date")
            extr["total"].value = None
            extr["total"].evidence = None
            extr["total"].raw_score = 0.2
            extr["total"].method = "missing"

    invv = extr["invoice_number"].value
    if isinstance(invv, str):
        if not any(ch.isdigit() for ch in invv):
            extr["invoice_number"].reasons.append("rejected_invoice_number_no_digit")
            extr["invoice_number"].value = None
            extr["invoice_number"].evidence = None
            extr["invoice_number"].raw_score = 0.2
            extr["invoice_number"].method = "missing"

    # --- LLM fallback (only for missing entity fields) ---
    llm_used = False
    llm_untrusted = False
    llm_payload: dict[str, Any] | None = None

    if llm is not None:
        need_llm_fields = [f for f in ["company_name", "company_address", "bank_name"] if extr[f].value is None]
        if need_llm_fields:
            prompt = build_llm_prompt(text[:8000])
            llm_payload = llm.generate_json(prompt)
            llm_used = True

            if llm_payload is None:
                llm_untrusted = True
            else:
                for f in need_llm_fields:
                    obj = llm_payload.get(f, None)
                    if not isinstance(obj, dict):
                        llm_untrusted = True
                        continue
                    val = obj.get("value")
                    ev = obj.get("evidence")

                    if val is None or ev is None:
                        continue
                    if not isinstance(ev, str) or not ev.strip():
                        llm_untrusted = True
                        continue
                    if not _evidence_matches(text, ev):
                        llm_untrusted = True
                        continue

                    val_s = str(val).strip()
                    if _looks_like_placeholder(val_s) or _looks_like_ocr_blob(val_s) or _too_long_for_entity(val_s):
                        llm_untrusted = True
                        continue

                    extr[f].value = val_s
                    extr[f].method = "llm"
                    extr[f].evidence = ev.strip()
                    extr[f].raw_score = 0.60
                    extr[f].reasons.append("llm_fallback_used")

    # --- Validation maps ---
    valid_map: Dict[str, bool] = {}
    conflict_map: Dict[str, bool] = {}

    iban_val = extr["iban"].value
    valid_map["iban"] = bool(iban_val) and validate_iban(iban_val)
    conflict_map["iban"] = False

    total_val = extr["total"].value
    if total_val and isinstance(total_val, str):
        amt = parse_eur_amount(total_val.split()[0])
        valid_map["total"] = (amt is not None) and (amt > 0)
    else:
        valid_map["total"] = False
    conflict_map["total"] = "multiple_total_candidates" in extr["total"].reasons

    valid_map["invoice_date"] = extr["invoice_date"].value is not None
    valid_map["due_date"] = extr["due_date"].value is not None
    conflict_map["invoice_date"] = False
    conflict_map["due_date"] = False

    valid_map["invoice_number"] = extr["invoice_number"].value is not None
    conflict_map["invoice_number"] = False

    for f in ["company_name", "company_address", "bank_name", "telephone"]:
        valid_map[f] = extr[f].value is not None
        conflict_map[f] = False

    # --- Confidence + routing ---
    field_outputs: Dict[str, Any] = {}
    field_routes: Dict[str, bool] = {}
    missing_count = 0

    from .config import TARGET_FIELDS  # keep in sync

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
            untrusted_llm=llm_untrusted and e.method == "llm",
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
        "invoice_id": "ui",
        "avg_ocr_conf": round(ocr.avg_conf, 4),
        "fields": field_outputs,
        "route_to_human": field_routes,
        "invoice_route_to_human": invoice_route,
        "invoice_route_reasons": invoice_reasons,
        "llm_used": llm_used,
    }

    # --- Table output ---
    rows: List[Dict[str, Any]] = []
    for f in TARGET_FIELDS:
        rows.append(
            {
                "field": f,
                "value": out["fields"][f]["value"],
                "confidence": out["fields"][f]["confidence"],
                "method": out["fields"][f]["method"],
                "route_to_human": bool(out["route_to_human"][f]),
            }
        )
    df = pd.DataFrame(rows, columns=["field", "value", "confidence", "method", "route_to_human"])

    # --- Badge markdown ---
    badge = f"""
### Invoice routing
**invoice_route_to_human:** `{invoice_route}`  
**reasons:** {", ".join(invoice_reasons) if invoice_reasons else "—"}  
**avg_ocr_conf:** `{out["avg_ocr_conf"]}`  
**llm_used:** `{out["llm_used"]}`
""".strip()

    pretty = json.dumps(out, indent=2, ensure_ascii=False)
    return pretty, df, badge


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Invoice HITL Extractor") as demo:
        gr.Markdown("# Invoice HITL Extractor")
        gr.Markdown("Upload an invoice image → run extraction → review JSON + per-field routing.")

        with gr.Row():
            inp = gr.Image(type="pil", label="Upload invoice image")

        run_btn = gr.Button("Run extraction", variant="primary")

        with gr.Row():
            json_out = gr.Code(label="Pretty JSON output", language="json")
        with gr.Row():
            table_out = gr.Dataframe(label="Fields table", interactive=False, wrap=True)
        with gr.Row():
            badge_out = gr.Markdown()

        run_btn.click(
            fn=run_extraction_ui,
            inputs=[inp],
            outputs=[json_out, table_out, badge_out],
        )

    return demo


def main() -> None:
    demo = build_app()
    # IMPORTANT for Docker: bind to 0.0.0.0
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    main()
