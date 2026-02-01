from __future__ import annotations
import json
import os
import re
from typing import Any, Iterable

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def safe_lower(s: str) -> str:
    return s.lower() if isinstance(s, str) else ""

def extract_lines(text: str) -> list[str]:
    raw = text.replace("\r", "\n")
    parts = []
    for ln in raw.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        # OCR often uses pipes instead of newlines
        for sub in ln.split("|"):
            sub = sub.strip()
            if not sub:
                continue
            parts.append(sub)

    # also split very long “lines” into chunks (optional but very helpful)
    out = []
    for ln in parts:
        if len(ln) > 180:
            # chunk by double spaces as pseudo-breaks
            chunks = [c.strip() for c in ln.split("  ") if c.strip()]
            out.extend(chunks if chunks else [ln])
        else:
            out.append(ln)
    return out


def find_first_matching_line(lines: list[str], patterns: list[str]) -> tuple[int, str] | None:
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(p in low for p in patterns):
            return i, ln
    return None

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
