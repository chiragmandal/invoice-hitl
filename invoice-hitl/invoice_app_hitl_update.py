# Install:
#   pip install python-doctr
#
# Optional (better logo scaling):
#   pip install pillow
#
# Run GUI:
#   python invoice_app.py
#
# Optional CLI:
#   python invoice_app.py --input "/path/to/invoice.pdf"
#
# Logo:
#   Put your logo PNG next to this file and name it: cloudfactory_logo.png
#   (or change InvoiceGUI.LOGO_FILENAME)

import argparse
import json
import os
import re
import threading
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# --- docTR ---
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# --- Tkinter GUI ---
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

# --- Optional PIL for nicer logo scaling ---
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = None
    ImageTk = None


# =============================================================================
# Extraction code (your Code 2) - same logic
# =============================================================================

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = (
        s.replace("ä", "a").replace("ö", "o").replace("ü", "u")
         .replace("ß", "ss")
    )
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def extract_date(text: str) -> Optional[str]:
    m = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", text)
    return m.group(0) if m else None

def extract_iban(text: str) -> Optional[str]:
    compact = re.sub(r"\s+", "", text).upper()
    m = re.search(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", compact)
    return m.group(0) if m else None

def extract_phone(text: str) -> Optional[str]:
    m = re.search(r"(?:telefon|tel)\s*[:.]?\s*([+\d][\d\s()/.-]{6,})", normalize_text(text))
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    m2 = re.search(r"\b(\+?\d[\d\s()/.-]{7,}\d)\b", text)
    if m2:
        return re.sub(r"\s+", " ", m2.group(1)).strip()

    return None

def parse_german_amount(text: str) -> Optional[str]:
    m = re.search(r"(\d{1,3}(?:\.\d{3})*|\d+),(\d{2})\s*€?", text)
    if not m:
        return None
    euros = m.group(1).replace(".", "")
    cents = m.group(2)
    return f"{euros},{cents} €"

def line_bbox(line_words) -> Tuple[float, float, float, float]:
    x0 = min(w.geometry[0][0] for w in line_words)
    y0 = min(w.geometry[0][1] for w in line_words)
    x1 = max(w.geometry[1][0] for w in line_words)
    y1 = max(w.geometry[1][1] for w in line_words)
    return (x0, y0, x1, y1)

def extract_lines_with_geometry(document) -> List[Dict[str, Any]]:
    lines = []
    for page_i, page in enumerate(document.pages):
        for block in page.blocks:
            for line in block.lines:
                if not line.words:
                    continue
                text = " ".join(w.value for w in line.words).strip()
                conf = sum(w.confidence for w in line.words) / max(1, len(line.words))
                bbox = line_bbox(line.words)
                lines.append({
                    "page": page_i,
                    "text": text,
                    "confidence": float(round(conf, 4)),
                    "bbox": bbox,
                    "y_center": (bbox[1] + bbox[3]) / 2.0,
                    "x0": bbox[0],
                })
    lines.sort(key=lambda d: (d["page"], d["y_center"], d["x0"]))
    return lines

def value_after_colon(text: str) -> Optional[str]:
    if ":" in text:
        rhs = text.split(":", 1)[1].strip()
        return rhs if rhs else None
    return None

def pack_field(value: Optional[str], confidence: Optional[float]) -> Dict[str, Any]:
    return {"wert": value, "konfidenz": confidence}

def best_confidence_for_match(lines: List[Dict[str, Any]], matched_value: str) -> Optional[float]:
    if not matched_value:
        return None
    mv_compact = re.sub(r"\s+", "", matched_value).upper()
    best = None
    for ln in lines:
        t_compact = re.sub(r"\s+", "", ln["text"]).upper()
        if mv_compact and mv_compact in t_compact:
            c = ln["confidence"]
            best = c if best is None else max(best, c)
    return best

LABELS_DE = {
    "rechnungsnummer": ["rechnungsnummer", "rechnung nr", "rechnungs-nr", "rechnungs nr"],
    "rechnungsdatum": ["rechnungsdatum", "datum", "datum der rechnung"],
    "falligkeitsdatum": [
        "falligkeitsdatum", "fälligkeitsdatum", "faelligkeitsdatum",
        "zahlbar bis", "fällig bis", "faellig bis", "fallig bis"
    ],
    "telefonnummer": ["telefon", "telefonnummer", "tel", "rufnummer"],
    "summe": ["summe", "gesamt", "gesamtbetrag", "rechnungsbetrag", "gesamt brutto", "brutto gesamt"],
    "iban": ["iban"],
    "name_der_bank": ["bank", "kreditinstitut", "hausbank"],
    "name_der_firma": ["firma", "unternehmen", "anbieter", "verkäufer", "verkaeufer", "rechnung von"],
    "adresse_der_firma": ["adresse", "anschrift"],
}

def find_best_label_line(lines: List[Dict[str, Any]], variants: List[str], min_score: float = 0.78):
    best = None
    best_score = 0.0
    for i, ln in enumerate(lines):
        t = ln["text"]
        for v in variants:
            score = similarity(t, v)
            if normalize_text(v) in normalize_text(t):
                score = max(score, 0.90)
            if score > best_score:
                best_score = score
                best = (i, ln, best_score)
    return best if best and best_score >= min_score else None

def pick_nearby_value(
    lines: List[Dict[str, Any]],
    label_idx: int,
    want_de_key: str,
    max_next_lines: int = 4
) -> Tuple[Optional[str], Optional[float]]:
    label_line = lines[label_idx]
    label_text = label_line["text"]

    same = value_after_colon(label_text)
    if same:
        return same, label_line["confidence"]

    if want_de_key == "rechnungsnummer":
        m = re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}[-/]\d+\b", label_text)
        if m:
            return m.group(0), label_line["confidence"]
        m2 = re.search(r"\b\d{4}-\d{1,2}-\d{1,2}-\d+\b", label_text)
        if m2:
            return m2.group(0), label_line["confidence"]

    if want_de_key in ("rechnungsdatum", "falligkeitsdatum"):
        d = extract_date(label_text)
        if d:
            return d, label_line["confidence"]

    if want_de_key == "telefonnummer":
        ph = extract_phone(label_text)
        if ph:
            return ph, label_line["confidence"]

    if want_de_key == "iban":
        ib = extract_iban(label_text)
        if ib:
            return ib, label_line["confidence"]

    if want_de_key == "summe":
        amt = parse_german_amount(label_text)
        if amt:
            return amt, label_line["confidence"]

    for j in range(label_idx + 1, min(len(lines), label_idx + 1 + max_next_lines)):
        ln = lines[j]
        txt = ln["text"]

        if want_de_key == "rechnungsnummer":
            m = re.search(r"\b\d{4}-\d{1,2}-\d{1,2}-\d+\b", txt)
            if m:
                return m.group(0), ln["confidence"]

        elif want_de_key in ("rechnungsdatum", "falligkeitsdatum"):
            d = extract_date(txt)
            if d:
                return d, ln["confidence"]

        elif want_de_key == "telefonnummer":
            ph = extract_phone("Telefon: " + txt) or extract_phone(txt)
            if ph:
                return ph, ln["confidence"]

        elif want_de_key == "iban":
            ib = extract_iban(txt)
            if ib:
                return ib, ln["confidence"]

        elif want_de_key == "summe":
            amt = parse_german_amount(txt)
            if amt:
                return amt, ln["confidence"]

        else:
            if txt.strip():
                return txt.strip(), ln["confidence"]

    return None, None

def fallback_summe(lines: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
    best = None
    for ln in lines:
        a = parse_german_amount(ln["text"])
        if not a:
            continue
        val = float(a.replace(" €", "").replace(".", "").replace(",", "."))
        if best is None or val > best[0]:
            best = (val, a, ln["confidence"])
    return (best[1], best[2]) if best else (None, None)

def guess_company_name_and_address(
    lines: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    legal_markers = ["gmbh", "ag", "kg", "ug", "gbr", "e v"]
    zip_city = re.compile(r"\b\d{5}\b")
    streetish = re.compile(r"\b(str\.|strasse|straße|weg|platz|allee|postfach)\b", re.IGNORECASE)

    header = lines[:25]
    company_idx = None
    for i, ln in enumerate(header):
        nt = normalize_text(ln["text"])
        if any(m in nt for m in legal_markers) or ("db" in nt and "vertrieb" in nt):
            company_idx = i
            break

    company_name = header[company_idx]["text"] if company_idx is not None else None
    company_name_conf = header[company_idx]["confidence"] if company_idx is not None else None

    if company_idx is None:
        return company_name, company_name_conf, None, None

    addr_parts = []
    addr_confs = []
    for ln in header[company_idx + 1: company_idx + 12]:
        t = ln["text"].strip()
        if not t:
            continue
        if streetish.search(t) or zip_city.search(t):
            addr_parts.append(t)
            addr_confs.append(ln["confidence"])
        if re.search(r"\b(rechnung|rechnungsnummer|rechnungsdatum)\b", normalize_text(t)):
            break

    company_address = ", ".join(addr_parts) if addr_parts else None
    company_address_conf = (sum(addr_confs) / len(addr_confs)) if addr_confs else None

    return company_name, company_name_conf, company_address, company_address_conf

def extract_invoice_fields(doc) -> Dict[str, Any]:
    lines = extract_lines_with_geometry(doc)
    all_text = "\n".join(ln["text"] for ln in lines)

    firma, firma_conf, adresse, adresse_conf = guess_company_name_and_address(lines)

    iban_value = extract_iban(all_text)
    iban_conf = best_confidence_for_match(lines, iban_value) if iban_value else None

    result_de = {
        "name_der_bank": pack_field(None, None),
        "name_der_firma": pack_field(firma, firma_conf),
        "adresse_der_firma": pack_field(adresse, round(adresse_conf, 4) if adresse_conf is not None else None),
        "falligkeitsdatum": pack_field(None, None),
        "iban": pack_field(iban_value, iban_conf),
        "rechnungsdatum": pack_field(None, None),
        "rechnungsnummer": pack_field(None, None),
        "summe": pack_field(None, None),
        "telefonnummer": pack_field(None, None),
        "meta": {"pages": len(getattr(doc, "pages", [])), "lines": len(lines)},
    }

    hit = find_best_label_line(lines, LABELS_DE["telefonnummer"], min_score=0.75)
    if hit:
        idx, _, _ = hit
        val, conf = pick_nearby_value(lines, idx, "telefonnummer")
        result_de["telefonnummer"] = pack_field(val, conf)
    if result_de["telefonnummer"]["wert"] is None:
        for ln in lines:
            ph = extract_phone(ln["text"])
            if ph:
                result_de["telefonnummer"] = pack_field(ph, ln["confidence"])
                break

    hit = find_best_label_line(lines, LABELS_DE["rechnungsnummer"], min_score=0.75)
    if hit:
        idx, _, _ = hit
        val, conf = pick_nearby_value(lines, idx, "rechnungsnummer")
        result_de["rechnungsnummer"] = pack_field(val, conf)

    if result_de["rechnungsnummer"]["wert"] is None:
        m = re.search(r"\b\d{4}-\d{1,2}-\d{1,2}-\d+\b", all_text)
        if m:
            val = m.group(0)
            conf = best_confidence_for_match(lines, val)
            result_de["rechnungsnummer"] = pack_field(val, conf)

    hit = find_best_label_line(lines, LABELS_DE["rechnungsdatum"], min_score=0.75)
    if hit:
        idx, _, _ = hit
        val, conf = pick_nearby_value(lines, idx, "rechnungsdatum")
        result_de["rechnungsdatum"] = pack_field(val, conf)

    if result_de["rechnungsdatum"]["wert"] is None:
        for ln in lines:
            d = extract_date(ln["text"])
            if d:
                result_de["rechnungsdatum"] = pack_field(d, ln["confidence"])
                break

    hit = find_best_label_line(lines, LABELS_DE["falligkeitsdatum"], min_score=0.75)
    if hit:
        idx, _, _ = hit
        val, conf = pick_nearby_value(lines, idx, "falligkeitsdatum")
        result_de["falligkeitsdatum"] = pack_field(val, conf)

    hit = find_best_label_line(lines, LABELS_DE["summe"], min_score=0.75)
    if hit:
        idx, _, _ = hit
        val, conf = pick_nearby_value(lines, idx, "summe")
        result_de["summe"] = pack_field(val, conf)

    if result_de["summe"]["wert"] is None:
        val, conf = fallback_summe(lines)
        result_de["summe"] = pack_field(val, conf)

    if result_de["iban"]["wert"] is None:
        hit = find_best_label_line(lines, LABELS_DE["iban"], min_score=0.75)
        if hit:
            idx, _, _ = hit
            val, conf = pick_nearby_value(lines, idx, "iban")
            result_de["iban"] = pack_field(val, conf)

    bank_markers = ["bank", "sparkasse", "volksbank", "postbank", "deutsche bank", "commerzbank", "ing", "dkb"]
    for ln in lines:
        nt = normalize_text(ln["text"])
        if any(b in nt for b in bank_markers):
            if "www" not in nt and "@" not in ln["text"]:
                result_de["name_der_bank"] = pack_field(ln["text"], ln["confidence"])
                break

    return result_de


# =============================================================================
# OCR pipeline + averaging
# =============================================================================

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = ocr_predictor(pretrained=True)
    return _MODEL

def load_document(path: str) -> DocumentFile:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return DocumentFile.from_pdf(path)
    return DocumentFile.from_images(path)

def run_pipeline(path: str) -> Dict[str, Any]:
    doc_file = load_document(path)
    model = get_model()
    ocr_doc = model(doc_file)
    return extract_invoice_fields(ocr_doc)

def active_field_confidences(fields: Dict[str, Any]) -> List[float]:
    confs: List[float] = []
    for k, v in fields.items():
        if k == "meta":
            continue
        if not (isinstance(v, dict) and "wert" in v and "konfidenz" in v):
            continue

        wert = v.get("wert")
        konf = v.get("konfidenz")

        if konf is None:
            continue
        if wert is None:
            continue
        if isinstance(wert, str) and wert.strip() == "":
            continue

        try:
            confs.append(float(konf))
        except Exception:
            pass
    return confs

def avg_confidence_of_active_fields(fields: Dict[str, Any]) -> Optional[float]:
    confs = active_field_confidences(fields)
    if not confs:
        return None
    return sum(confs) / len(confs)

def count_missing_fields(fields: Dict[str, Any]) -> int:
    missing = 0
    for k, v in fields.items():
        if k == "meta":
            continue
        if not (isinstance(v, dict) and "wert" in v and "konfidenz" in v):
            continue
        val = v.get("wert")
        if val is None or (isinstance(val, str) and val.strip() == ""):
            missing += 1
    return missing


# =============================================================================
# Fancy-ish GUI
# =============================================================================

def _short_path(p: str, max_len: int = 70) -> str:
    if not p:
        return ""
    if len(p) <= max_len:
        return p
    head = p[: max_len // 2 - 2]
    tail = p[-max_len // 2 :]
    return f"{head}…{tail}"

class InvoiceGUI:
    THRESHOLD = 0.8
    LOGO_FILENAME = "cloudfactory_logo.png"  # put next to invoice_app.py

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Invoice OCR Extractor")
        self.root.minsize(980, 560)

        # Try a nicer theme if available
        style = ttk.Style(self.root)
        available = set(style.theme_names())
        for preferred in ("clam", "vista", "aqua"):
            if preferred in available:
                style.theme_use(preferred)
                break

        # Fonts
        default_font = tkfont.nametofont("TkDefaultFont")
        self.font_normal = default_font.copy()
        self.font_normal.configure(size=max(10, default_font.cget("size")))

        self.font_title = self.font_normal.copy()
        self.font_title.configure(size=self.font_normal.cget("size") + 6, weight="bold")

        self.font_subtitle = self.font_normal.copy()
        self.font_subtitle.configure(size=self.font_normal.cget("size") + 1)

        # ttk styling
        style.configure("Card.TFrame", padding=12, relief="groove")
        style.configure("Title.TLabel", font=self.font_title)
        style.configure("SubTitle.TLabel", font=self.font_subtitle)
        style.configure("Table.Treeview", rowheight=28)
        style.configure(
            "Table.Treeview.Heading",
            font=(self.font_normal.actual("family"), self.font_normal.actual("size"), "bold")
        )
        style.map("Accent.TButton", foreground=[("disabled", "#888")])
        try:
            style.configure("Accent.TButton", padding=(12, 8))
        except Exception:
            pass

        # State
        self.latest_fields: Optional[Dict[str, Any]] = None
        self.latest_path: Optional[str] = None

        self.edit_mode: bool = False
        self.edit_vars: Dict[str, tk.StringVar] = {}

        # Track unsaved manual edits
        self.dirty: bool = False

        # Inline editor state
        self._edit_entry: Optional[tk.Entry] = None
        self._edit_item_id: Optional[str] = None
        self._edit_field_key: Optional[str] = None

        # Logo (keep a reference!)
        self.logo_img = self._load_logo()

        # Layout: header / content / footer
        self._build_header()
        self._build_content()
        self._build_footer()
        self._build_context_menu()
        self._bind_shortcuts()

        self._set_status("Ready", kind="ok")
        self._update_avg_and_routing(None)

    def _load_logo(self):
        """
        Loads a logo image from the same folder as this script.
        Uses PIL if available for smooth scaling; otherwise uses tk.PhotoImage.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_dir, self.LOGO_FILENAME)
        if not os.path.exists(logo_path):
            return None

        max_h = 52  # target header logo height
        try:
            if Image is not None and ImageTk is not None:
                img = Image.open(logo_path)
                if img.height > max_h:
                    ratio = max_h / float(img.height)
                    img = img.resize((max(1, int(img.width * ratio)), max_h))
                return ImageTk.PhotoImage(img)
            else:
                img = tk.PhotoImage(file=logo_path)
                if img.height() > max_h:
                    factor = max(2, int(round(img.height() / max_h)))
                    img = img.subsample(factor, factor)
                return img
        except Exception:
            return None

    # ---------------- UI Builders ----------------

    def _build_header(self):
        header = ttk.Frame(self.root, padding=(16, 14))
        header.pack(fill="x")

        # Logo
        if self.logo_img is not None:
            ttk.Label(header, image=self.logo_img).pack(side="left", padx=(0, 12))

        left = ttk.Frame(header)
        left.pack(side="left", fill="x", expand=True)

        ttk.Label(left, text="Invoice OCR Extractor", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Load an invoice image/PDF → extract fields → edit values directly in table → save *_updated.json",
            style="SubTitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        # Status "pill" using tk.Label (easy colors)
        self.status_pill = tk.Label(
            header,
            text="",
            padx=12,
            pady=6,
            bd=0,
            font=self.font_normal,
        )
        self.status_pill.pack(side="right")

    def _build_content(self):
        content = ttk.Frame(self.root, padding=(16, 0, 16, 16))
        content.pack(fill="both", expand=True)

        # Top card: file + actions + gauge
        top_card = ttk.Frame(content, style="Card.TFrame")
        top_card.pack(fill="x")

        # File row
        file_row = ttk.Frame(top_card)
        file_row.pack(fill="x")

        self.file_label = ttk.Label(file_row, text="📄 No file selected")
        self.file_label.pack(side="left", anchor="w", fill="x", expand=True)

        btns = ttk.Frame(file_row)
        btns.pack(side="right")

        self.btn_open = ttk.Button(btns, text="Open…", command=self.pick_file)
        self.btn_open.pack(side="left")

        self.btn_clear = ttk.Button(btns, text="Clear", command=self.clear_ui)
        self.btn_clear.pack(side="left", padx=(8, 0))

        # Options row
        opt_row = ttk.Frame(top_card)
        opt_row.pack(fill="x", pady=(10, 0))

        self.show_empty_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opt_row,
            text="Show empty fields",
            variable=self.show_empty_var,
            command=self._rerender_table_if_possible,
        ).pack(side="left")

        # Avg gauge + buttons (right)
        right = ttk.Frame(opt_row)
        right.pack(side="right")

        self.avg_text = ttk.Label(right, text="Avg confidence: N/A")
        self.avg_text.pack(side="left", padx=(0, 10))

        self.avg_gauge = ttk.Progressbar(right, orient="horizontal", length=180, mode="determinate", maximum=100)
        self.avg_gauge.pack(side="left", padx=(0, 10))

        self.route_btn = ttk.Button(
            right,
            text="Edit missing fields",
            style="Accent.TButton",
            command=self.route_to_human,  # still available
            state="disabled",
        )
        self.route_btn.pack(side="left")

        self.save_btn = ttk.Button(
            right,
            text="Save updated JSON",
            command=self.save_updated_json,
            state="disabled",
        )
        self.save_btn.pack(side="left", padx=(8, 0))

        # Progress bar (hidden until running)
        self.run_bar = ttk.Progressbar(top_card, orient="horizontal", mode="indeterminate")
        self.run_bar.pack(fill="x", pady=(12, 0))
        self.run_bar.pack_forget()

        # Table card
        table_card = ttk.Frame(content, style="Card.TFrame")
        table_card.pack(fill="both", expand=True, pady=(14, 0))

        hdr = ttk.Frame(table_card)
        hdr.pack(fill="x")

        ttk.Label(hdr, text="Extracted fields (double-click Value to edit)", font=self.font_subtitle).pack(side="left")

        self.meta_label = ttk.Label(hdr, text="", foreground="#666")
        self.meta_label.pack(side="right")

        body = ttk.Frame(table_card)
        body.pack(fill="both", expand=True, pady=(10, 0))

        self.tree = ttk.Treeview(
            body,
            columns=("field", "value", "confidence"),
            show="headings",
            style="Table.Treeview",
            height=14,
        )
        self.tree.heading("field", text="Field")
        self.tree.heading("value", text="Value")
        self.tree.heading("confidence", text="Confidence")
        self.tree.column("field", width=240, anchor="w")
        self.tree.column("value", width=560, anchor="w")
        self.tree.column("confidence", width=130, anchor="center")

        yscroll = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)

        # Row tag styles (alternating + low confidence)
        self.tree.tag_configure("odd", background="#F7F7F7")
        self.tree.tag_configure("even", background="#FFFFFF")
        self.tree.tag_configure("lowconf", foreground="#B00020")   # red-ish
        self.tree.tag_configure("midconf", foreground="#8A6D00")   # amber-ish

        # Right-click
        self.tree.bind("<Button-3>", self._on_right_click)

        # Double-click to edit Value column
        self.tree.bind("<Double-1>", self._on_double_click)

        # Edit card (optional panel; still supported)
        self.edit_card = ttk.Frame(content, style="Card.TFrame")
        self.edit_card.pack(fill="x", pady=(14, 0))
        self.edit_card.pack_forget()

        edit_hdr = ttk.Frame(self.edit_card)
        edit_hdr.pack(fill="x")
        ttk.Label(edit_hdr, text="Fill missing fields", font=self.font_subtitle).pack(side="left")
        ttk.Label(
            edit_hdr,
            text="Only fields with empty values are shown here.",
            foreground="#666",
        ).pack(side="right")

        self.edit_form = ttk.Frame(self.edit_card)
        self.edit_form.pack(fill="x", pady=(10, 0))
        self.edit_form.columnconfigure(1, weight=1)

    def _build_footer(self):
        footer = ttk.Frame(self.root, padding=(16, 0, 16, 14))
        footer.pack(fill="x")

        help_txt = "Tip: Double-click a Value cell to edit. Right-click a row to copy. Shortcut: Ctrl+O to open."
        ttk.Label(footer, text=help_txt, foreground="#666").pack(anchor="w")

    def _build_context_menu(self):
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="Copy Field", command=lambda: self._copy_selected(col=0))
        self.menu.add_command(label="Copy Value", command=lambda: self._copy_selected(col=1))
        self.menu.add_command(label="Copy Confidence", command=lambda: self._copy_selected(col=2))
        self.menu.add_separator()
        self.menu.add_command(label="Copy Row (tab-separated)", command=self._copy_selected_row)

    def _bind_shortcuts(self):
        self.root.bind_all("<Control-o>", lambda e: self.pick_file())
        self.root.bind_all("<Control-O>", lambda e: self.pick_file())

    # ---------------- Status + Controls ----------------

    def _set_status(self, text: str, kind: str = "ok"):
        # kind: ok / run / err
        if kind == "ok":
            bg, fg = "#E7F6EC", "#1B5E20"
        elif kind == "run":
            bg, fg = "#E8F0FE", "#0D47A1"
        else:
            bg, fg = "#FCE8E6", "#B00020"

        self.status_pill.configure(text=text, bg=bg, fg=fg)

    def _set_running(self, running: bool):
        if running:
            self.run_bar.pack(fill="x", pady=(12, 0))
            self.run_bar.start(12)
            self.btn_open.configure(state="disabled")
            self.btn_clear.configure(state="disabled")
            self.route_btn.configure(state="disabled")
            self.save_btn.configure(state="disabled")
        else:
            self.run_bar.stop()
            self.run_bar.pack_forget()
            self.btn_open.configure(state="normal")
            self.btn_clear.configure(state="normal")

    def clear_ui(self):
        self._destroy_inline_editor(commit=False)
        self.latest_fields = None
        self.latest_path = None
        self.edit_mode = False
        self.dirty = False
        self._hide_edit_panel()
        self.file_label.configure(text="📄 No file selected")
        self.meta_label.configure(text="")
        self._clear_table()
        self._update_avg_and_routing(None)
        self._set_status("Ready", kind="ok")

    def _hide_edit_panel(self):
        for w in self.edit_form.winfo_children():
            w.destroy()
        self.edit_vars = {}
        try:
            self.edit_card.pack_forget()
        except Exception:
            pass
        self.save_btn.configure(state="disabled")

    # ---------------- Table rendering ----------------

    def _clear_table(self):
        self._destroy_inline_editor(commit=False)
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _rerender_table_if_possible(self):
        if self.latest_fields:
            self._render_table(self.latest_fields)

    def _render_table(self, fields: Dict[str, Any]):
        self._clear_table()

        show_empty = bool(self.show_empty_var.get())

        rows = []
        for k, v in fields.items():
            if k == "meta":
                continue
            if not (isinstance(v, dict) and "wert" in v and "konfidenz" in v):
                continue

            val = v.get("wert")
            konf = v.get("konfidenz")

            val_str = "" if val is None else str(val)
            conf_str = "" if konf is None else f"{float(konf):.4f}"

            if not show_empty:
                if val is None or (isinstance(val, str) and val.strip() == "") or konf is None:
                    continue

            rows.append((k, val_str, conf_str, konf))

        for idx, (k, val_str, conf_str, konf) in enumerate(rows):
            tags = ["odd" if idx % 2 else "even"]
            if konf is not None:
                try:
                    f = float(konf)
                    if f < 0.60:
                        tags.append("lowconf")
                    elif f < self.THRESHOLD:
                        tags.append("midconf")
                except Exception:
                    pass

            self.tree.insert("", "end", values=(k, val_str, conf_str), tags=tuple(tags))

        meta = fields.get("meta") or {}
        pages = meta.get("pages", "")
        lines = meta.get("lines", "")
        self.meta_label.configure(text=f"pages: {pages}   lines: {lines}")

    # ---------------- Average confidence + buttons ----------------

    def _update_avg_and_routing(self, fields: Optional[Dict[str, Any]]):
        if not fields:
            self.avg_text.configure(text="Avg confidence: N/A")
            self.avg_gauge["value"] = 0
            self.route_btn.configure(state="disabled")
            self.save_btn.configure(state="disabled")
            return

        avg = avg_confidence_of_active_fields(fields)
        confs = active_field_confidences(fields)
        missing = count_missing_fields(fields)

        if avg is None:
            self.avg_text.configure(text="Avg confidence: N/A")
            self.avg_gauge["value"] = 0
        else:
            self.avg_text.configure(text=f"Avg confidence: {avg:.4f}  ({len(confs)} active)  |  Missing: {missing}")
            self.avg_gauge["value"] = max(0, min(100, avg * 100))

        # Enable "Edit missing fields" if:
        # - there are missing fields, OR
        # - avg confidence is below threshold (review anyway)
        enable_edit = (missing > 0) or (avg is not None and avg < self.THRESHOLD)
        self.route_btn.configure(state=("normal" if enable_edit else "disabled"))

        # Enable Save if:
        # - inline edits happened (dirty), OR
        # - user is using the missing-fields panel (edit_mode)
        self.save_btn.configure(state=("normal" if (self.dirty or self.edit_mode) else "disabled"))

    # ---------------- File picking + OCR threading ----------------

    def pick_file(self):
        self._destroy_inline_editor(commit=False)

        path = filedialog.askopenfilename(
            title="Select invoice file",
            filetypes=[
                ("Images and PDFs", "*.png *.jpg *.jpeg *.pdf"),
                ("PNG", "*.png"),
                ("JPG", "*.jpg *.jpeg"),
                ("PDF", "*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        # Reset edit mode / dirty on new file
        self.edit_mode = False
        self.dirty = False
        self._hide_edit_panel()

        self.latest_path = path
        self.file_label.configure(text=f"📄 {_short_path(path)}")
        self._set_status("Running OCR… (first run may download weights)", kind="run")
        self._set_running(True)
        self._clear_table()
        self._update_avg_and_routing(None)

        t = threading.Thread(target=self._worker_run, args=(path,), daemon=True)
        t.start()

    def _worker_run(self, path: str):
        try:
            fields = run_pipeline(path)
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
            return
        self.root.after(0, lambda: self._render_result(fields))

    def _show_error(self, msg: str):
        self._set_running(False)
        self._set_status("Error", kind="err")
        self._update_avg_and_routing(None)
        messagebox.showerror("Error", msg)

    def _render_result(self, fields: Dict[str, Any]):
        self._destroy_inline_editor(commit=False)
        self.latest_fields = fields
        self.edit_mode = False
        self.dirty = False
        self._hide_edit_panel()
        self._render_table(fields)
        self._update_avg_and_routing(fields)
        self._set_running(False)
        self._set_status("Done", kind="ok")

    # ---------------- Right-click copy menu ----------------

    def _on_right_click(self, event):
        row_id = self.tree.identify_row(event.y)
        if row_id:
            self.tree.selection_set(row_id)
            self.menu.tk_popup(event.x_root, event.y_root)

    def _copy_selected(self, col: int):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        if not vals:
            return
        text = str(vals[col]) if col < len(vals) else ""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _copy_selected_row(self):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        text = "\t".join(map(str, vals))
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    # ---------------- Inline table editing (Value column) ----------------

    def _on_double_click(self, event):
        """Start inline editing when double-clicking the Value column."""
        if not self.latest_fields:
            return

        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        col = self.tree.identify_column(event.x)  # "#1" field, "#2" value, "#3" confidence
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return

        # Only allow editing the Value column
        if col != "#2":
            return

        values = self.tree.item(row_id, "values")
        if not values or len(values) < 3:
            return

        field_key = str(values[0])
        if field_key == "meta":
            return

        self._begin_edit_value(row_id=row_id, field_key=field_key)

    def _begin_edit_value(self, row_id: str, field_key: str):
        # If editor already open, close it
        self._destroy_inline_editor(commit=False)

        bbox = self.tree.bbox(row_id, column="#2")
        if not bbox:
            return
        x, y, w, h = bbox

        current_val = self.tree.item(row_id, "values")[1]

        self._edit_item_id = row_id
        self._edit_field_key = field_key

        self._edit_entry = tk.Entry(self.tree)
        self._edit_entry.insert(0, current_val)
        self._edit_entry.select_range(0, tk.END)
        self._edit_entry.focus_set()

        self._edit_entry.place(x=x, y=y, width=w, height=h)

        self._edit_entry.bind("<Return>", lambda e: self._destroy_inline_editor(commit=True))
        self._edit_entry.bind("<Escape>", lambda e: self._destroy_inline_editor(commit=False))
        self._edit_entry.bind("<FocusOut>", lambda e: self._destroy_inline_editor(commit=True))

    def _destroy_inline_editor(self, commit: bool):
        """Destroy the inline editor. If commit=True, save value into model + UI."""
        if self._edit_entry is None:
            return

        new_value = self._edit_entry.get().strip()
        row_id = self._edit_item_id
        field_key = self._edit_field_key

        try:
            self._edit_entry.place_forget()
            self._edit_entry.destroy()
        except Exception:
            pass

        self._edit_entry = None
        self._edit_item_id = None
        self._edit_field_key = None

        if not commit or not self.latest_fields or not row_id or not field_key:
            return

        # Update data structure
        if field_key in self.latest_fields and isinstance(self.latest_fields[field_key], dict):
            if new_value == "":
                self.latest_fields[field_key]["wert"] = None
                self.latest_fields[field_key]["konfidenz"] = None
            else:
                self.latest_fields[field_key]["wert"] = new_value
                self.latest_fields[field_key]["konfidenz"] = 1.0  # human-provided

        # Update row display (value + confidence text)
        conf_str = "1.0000" if new_value != "" else ""
        self.tree.item(row_id, values=(field_key, new_value, conf_str))

        # Update row tags
        self._reapply_row_tags()

        # Mark dirty and enable save
        self.dirty = True
        self._update_avg_and_routing(self.latest_fields)

    def _reapply_row_tags(self):
        """Re-apply alternating row colors + confidence-based color tags after edits."""
        rows = list(self.tree.get_children())
        for idx, item_id in enumerate(rows):
            vals = self.tree.item(item_id, "values")
            if not vals or len(vals) < 3:
                continue

            conf_str = vals[2]
            tags = ["odd" if idx % 2 else "even"]

            if conf_str:
                try:
                    f = float(conf_str)
                    if f < 0.60:
                        tags.append("lowconf")
                    elif f < self.THRESHOLD:
                        tags.append("midconf")
                except Exception:
                    pass

            self.tree.item(item_id, tags=tuple(tags))

    # ---------------- Edit missing + Save updated ----------------

    def route_to_human(self):
        """
        Optional panel-based edit mode (still supported).
        NOTE: With inline table editing, most users won't need this.
        """
        if not self.latest_fields:
            return

        # Build edit UI for missing fields
        self.edit_mode = True
        self._hide_edit_panel()
        self.edit_mode = True  # restore

        missing_keys: List[str] = []
        for k, v in self.latest_fields.items():
            if k == "meta":
                continue
            if not (isinstance(v, dict) and "wert" in v and "konfidenz" in v):
                continue
            val = v.get("wert")
            if val is None or (isinstance(val, str) and val.strip() == ""):
                missing_keys.append(k)

        if not missing_keys:
            messagebox.showinfo("Nothing to edit", "No missing fields found.")
            self.edit_mode = False
            self._update_avg_and_routing(self.latest_fields)
            return

        for r, key in enumerate(missing_keys):
            ttk.Label(self.edit_form, text=key).grid(row=r, column=0, sticky="w", padx=(0, 12), pady=4)
            var = tk.StringVar(value="")
            self.edit_vars[key] = var
            ttk.Entry(self.edit_form, textvariable=var).grid(row=r, column=1, sticky="ew", pady=4)

        self.edit_card.pack(fill="x", pady=(14, 0))
        self._update_avg_and_routing(self.latest_fields)

    def save_updated_json(self):
        """
        Saves current fields to:
          <original_filename>_updated.json
        If edit panel is open, applies entered values before saving.
        """
        if not self.latest_fields or not self.latest_path:
            return

        # If using the edit panel, apply edits before saving
        if self.edit_mode and self.edit_vars:
            for k, var in self.edit_vars.items():
                val = (var.get() or "").strip()
                if not val:
                    continue
                if k in self.latest_fields and isinstance(self.latest_fields[k], dict):
                    self.latest_fields[k]["wert"] = val
                    self.latest_fields[k]["konfidenz"] = 1.0  # human-provided
                    self.dirty = True  # unsaved changes now exist

            # Refresh UI after applying panel edits
            self._render_table(self.latest_fields)
            self._update_avg_and_routing(self.latest_fields)

        base = os.path.splitext(os.path.basename(self.latest_path))[0]
        initialdir = os.path.dirname(self.latest_path) or os.getcwd()
        initialfile = f"{base}_updated.json"

        out = filedialog.asksaveasfilename(
            title="Save updated JSON",
            initialdir=initialdir,
            initialfile=initialfile,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not out:
            return

        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.latest_fields, f, ensure_ascii=False, indent=2)

        # Saved: clear dirty flag (but keep edit_mode as-is)
        self.dirty = False
        self._update_avg_and_routing(self.latest_fields)

        messagebox.showinfo("Saved", f"Updated JSON saved:\n{out}")


# =============================================================================
# CLI + Entry
# =============================================================================

def cli_main(path: str) -> int:
    fields = run_pipeline(path)
    avg = avg_confidence_of_active_fields(fields)
    print(json.dumps(fields, ensure_ascii=False, indent=2))
    if avg is None:
        print("\nAvg confidence (active fields): N/A")
    else:
        print(f"\nAvg confidence (active fields): {avg:.4f}")
    print("Missing fields:", count_missing_fields(fields))
    return 0

def gui_main():
    root = tk.Tk()
    root.geometry("1020x610")
    InvoiceGUI(root)
    root.mainloop()

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Invoice OCR extractor (docTR) + Fancy Tkinter GUI")
    p.add_argument("--input", "-i", type=str, help="Path to invoice image/PDF for CLI mode")
    return p

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.input:
        raise SystemExit(cli_main(args.input))
    gui_main()
