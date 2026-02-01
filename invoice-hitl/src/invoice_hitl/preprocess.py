from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

def preprocess_pil(img: Image.Image, max_width: int = 1600, do_threshold: bool = False) -> Image.Image:
    # Convert PIL -> OpenCV
    arr = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Resize
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Mild denoise
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    if do_threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 11
        )

    out = Image.fromarray(gray)
    return out
