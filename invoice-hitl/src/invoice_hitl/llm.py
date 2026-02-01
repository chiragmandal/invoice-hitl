from __future__ import annotations

import json
import time
import requests
from typing import Any


class OllamaClient:
    """
    Robust Ollama client:
    - Uses format=json for strict JSON output
    - Retries with exponential backoff on transient failures/timeouts
    - keep_alive to reduce cold-start latency
    """
    def __init__(self, host: str, model: str, timeout_s: int = 600, max_retries: int = 3):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def generate_json(self, prompt: str) -> dict[str, Any] | None:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,

            # Critical: ask Ollama to return strict JSON in response text
            "format": "json",

            # Keep model warm (reduces random latency spikes)
            "keep_alive": "10m",

            # Reduce randomness
            "options": {
                "temperature": 0,
            },
        }

        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout_s)
                r.raise_for_status()
                data = r.json()

                text = (data.get("response") or "").strip()
                if not text:
                    return None

                # In format=json mode, the entire response should be JSON
                try:
                    return json.loads(text)
                except Exception:
                    # Fallback: extract first {...} in case model/server ignored format=json
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        candidate = text[start:end + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            return None
                    return None

            except Exception as e:
                last_err = e
                # backoff: 2,4,8 seconds (cap)
                time.sleep(min(2 ** attempt, 8))

        return None


def build_llm_prompt(ocr_text: str) -> str:
    return f"""
You are extracting fields from OCR text of a German invoice.
Return ONLY valid JSON. No markdown. No extra keys.

Fields:
- company_name: string|null
- company_address: string|null
- bank_name: string|null

Rules:
1) Only use information present in the OCR text.
2) For each field also return a short evidence quote (substring) from OCR text.
3) If uncertain or not present, output null and evidence null.
4) Do NOT return placeholders like "[Adresse]" or "[Unternehmensname]". If those appear, output null.

Output schema:
{{
  "company_name": {{"value": ..., "evidence": ...}},
  "company_address": {{"value": ..., "evidence": ...}},
  "bank_name": {{"value": ..., "evidence": ...}}
}}

OCR TEXT:
<<<{ocr_text}>>>
""".strip()
