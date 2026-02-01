from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pytesseract
from pytesseract import Output
from PIL import Image
from .utils import normalize_whitespace

@dataclass
class OCRToken:
    text: str
    conf: float  # 0..1
    bbox: tuple[int, int, int, int]  # x, y, w, h

@dataclass
class OCRResult:
    full_text: str
    tokens: list[OCRToken]
    avg_conf: float

def run_tesseract(img: Image.Image, lang: str = "deu") -> OCRResult:
    data: dict[str, Any] = pytesseract.image_to_data(img, lang=lang, output_type=Output.DICT)

    tokens: list[OCRToken] = []
    texts: list[str] = []
    confs: list[float] = []

    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue

        # tesseract conf is often string; may be -1 for non-words
        try:
            c = float(data["conf"][i])
        except Exception:
            c = -1.0
        if c < 0:
            continue
        c01 = max(0.0, min(1.0, c / 100.0))

        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        tokens.append(OCRToken(text=txt, conf=c01, bbox=(x, y, w, h)))
        texts.append(txt)
        confs.append(c01)

    full_text = normalize_whitespace(" ".join(texts))
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return OCRResult(full_text=full_text, tokens=tokens, avg_conf=avg_conf)
