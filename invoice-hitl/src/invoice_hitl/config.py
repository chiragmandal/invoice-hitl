from __future__ import annotations
from dataclasses import dataclass
import os

TARGET_FIELDS = [
    "bank_name",
    "company_name",
    "company_address",
    "due_date",
    "iban",
    "invoice_date",
    "invoice_number",
    "total",
    "telephone",
]

CRITICAL_FIELDS = ["iban", "total", "invoice_number"]

@dataclass(frozen=True)
class Settings:
    hf_dataset_id: str
    hf_split: str
    ollama_host: str
    ollama_model: str
    use_llm: bool
    output_path: str
    debug_dir: str
    limit: int

    # OCR / preprocess
    ocr_lang: str = "deu"
    max_width: int = 1600
    do_threshold: bool = False

    # Confidence priors
    prior_rule_validated: float = 0.90
    prior_rule_only: float = 0.70
    prior_llm_evidence: float = 0.60
    prior_llm_untrusted: float = 0.20

    # Routing thresholds
    thr_iban: float = 0.85
    thr_total: float = 0.80
    thr_invoice_number: float = 0.75
    thr_dates: float = 0.70
    thr_entity: float = 0.65  # company/bank/address
    thr_phone: float = 0.60

    invoice_ocr_quality_thr: float = 0.55
    max_missing_fields_for_auto: int = 2

def load_settings() -> Settings:
    hf_dataset_id = os.getenv("HF_DATASET_ID", "").strip()
    if not hf_dataset_id:
        raise ValueError("HF_DATASET_ID env var is required (set to the HuggingFace dataset id).")

    return Settings(
        hf_dataset_id=hf_dataset_id,
        hf_split=os.getenv("HF_DATASET_SPLIT", "train").strip(),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct").strip(),
        use_llm=os.getenv("USE_LLM", "1").strip() == "1",
        output_path=os.getenv("OUTPUT_PATH", "outputs/predictions.jsonl").strip(),
        debug_dir=os.getenv("DEBUG_DIR", "outputs/debug").strip(),
        limit=int(os.getenv("LIMIT", "25")),
    )
