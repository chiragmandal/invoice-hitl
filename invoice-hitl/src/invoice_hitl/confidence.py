from __future__ import annotations

from dataclasses import dataclass
from .config import Settings
from .utils import clamp01


@dataclass
class ConfidenceResult:
    conf: float
    reasons: list[str]


def combine_confidence(
    settings: Settings,
    raw_score: float,
    ocr_avg_conf: float,
    valid: bool,
    method: str,
    conflict: bool = False,
    untrusted_llm: bool = False,
) -> ConfidenceResult:
    """
    Combines multiple weak signals into a single [0,1] confidence.

    Key changes vs your previous version (as discussed):
    - Don't let 'valid=False' nuke confidence for non-critical fields (we only
      add a *small* penalty, instead of hard 0/1 dominance).
    - Make OCR confidence less dominant (still important, but not everything).
    - LLM untrusted becomes a *penalty* rather than overwriting the prior.
    - Conflict is a penalty (as before), but kept modest.
    """

    reasons: list[str] = []

    # --- Base prior by extraction method ---
    if method == "rule":
        prior = settings.prior_rule_only
        reasons.append("prior_rule_only")
    elif method == "llm":
        prior = settings.prior_llm_evidence
        reasons.append("prior_llm_evidence")
    else:
        prior = 0.50
        reasons.append("prior_default")

    # --- Validation is a soft signal (not binary dominance) ---
    # Instead of valid_score=1/0, use a small bonus/penalty.
    if valid:
        valid_bonus = 0.10
        reasons.append("validation_passed")
    else:
        valid_bonus = -0.10
        reasons.append("validation_failed")

    # --- Conflict penalty (e.g. multiple candidates for totals) ---
    conflict_penalty = 0.15 if conflict else 0.0
    if conflict:
        reasons.append("conflict_penalty")

    # --- Untrusted LLM penalty (per-field) ---
    # IMPORTANT: do NOT overwrite prior. Penalize on top.
    llm_untrusted_penalty = 0.0
    if untrusted_llm:
        llm_untrusted_penalty = (settings.prior_llm_evidence - settings.prior_llm_untrusted)
        # ensure it's non-negative (in case settings are odd)
        if llm_untrusted_penalty < 0:
            llm_untrusted_penalty = 0.20
        reasons.append("llm_untrusted_output")

    # --- Combine ---
    # raw_score already encodes extractor confidence; keep it small.
    conf = (
        0.45 * ocr_avg_conf +
        0.30 * prior +
        0.10 * raw_score +
        valid_bonus -
        conflict_penalty -
        llm_untrusted_penalty
    )

    conf = clamp01(conf)
    return ConfidenceResult(conf=conf, reasons=reasons)
