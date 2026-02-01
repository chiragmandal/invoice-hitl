from __future__ import annotations
from dataclasses import dataclass
from .config import Settings, CRITICAL_FIELDS

@dataclass
class RoutingDecision:
    route_field: bool
    reasons: list[str]

def field_threshold(settings: Settings, field: str) -> float:
    if field == "iban":
        return settings.thr_iban
    if field == "total":
        return settings.thr_total
    if field == "invoice_number":
        return settings.thr_invoice_number
    if field in ("invoice_date", "due_date"):
        return settings.thr_dates
    if field in ("company_name", "company_address", "bank_name"):
        return settings.thr_entity
    if field == "telephone":
        return settings.thr_phone
    return 0.65

def route_field(settings: Settings, field: str, conf: float, missing: bool, validation_failed: bool, extra_reasons: list[str]) -> RoutingDecision:
    thr = field_threshold(settings, field)
    reasons: list[str] = []

    if missing:
        reasons.append("missing_value")
    if validation_failed:
        reasons.append("validation_failed")
    reasons.extend(extra_reasons)

    should_route = missing or validation_failed or (conf < thr)
    if conf < thr:
        reasons.append(f"below_threshold_{thr:.2f}")
    return RoutingDecision(route_field=should_route, reasons=reasons)

def route_invoice(settings: Settings, field_routes: dict[str, bool], avg_ocr_conf: float, missing_count: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if avg_ocr_conf < settings.invoice_ocr_quality_thr:
        reasons.append("low_ocr_quality")

    # critical field gating
    for f in CRITICAL_FIELDS:
        if field_routes.get(f, False):
            reasons.append(f"critical_field_routed:{f}")

    if missing_count > settings.max_missing_fields_for_auto:
        reasons.append("too_many_missing_fields")

    return (len(reasons) > 0), reasons
