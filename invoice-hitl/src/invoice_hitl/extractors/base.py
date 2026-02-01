from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Method = Literal["rule", "llm", "hybrid", "missing"]

@dataclass
class Extraction:
    field: str
    value: str | None
    method: Method
    evidence: str | None
    raw_score: float  # 0..1 (pre-aggregation)
    reasons: list[str]
