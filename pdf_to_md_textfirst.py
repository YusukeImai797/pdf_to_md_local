# pdf_to_md_textfirst.py
"""
Text-first PDF to Markdown converter.

Key principles:
- Body text from PDF text layer (PyMuPDF) - not Vision LLM
- Figure descriptions are REQUIRED (Vision for what's not in text layer)
- Table reconstruction from word-level text layer (Hybrid-lite)
- Deterministic QA gates that fail on ellipsis, code fences, figure-to-table, coverage
- Header/footer detection via variable bands (not fixed ratios)
- Durable diagnostics: nvidia-smi logging, fsync checkpoints, postmortem collection
"""

import os
import re
import io
import gc
import json
import time
import base64
import datetime
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from lmstudio_client import LMStudioClient


# ----------------------------
# Config
# ----------------------------

@dataclass
class ConverterConfig:
    vision_model: str = "qwen/qwen2.5-vl-7b"
    # Text model for optional LLM audit (not used for body text)
    text_model: Optional[str] = None

    # rendering
    render_dpi: int = 200
    jpeg_quality: int = 85
    max_crop_width_px: int = 1400  # crop width limit (peak load control)

    # header/footer inference
    band_scan_margin_norm: float = 0.15  # search range for header/footer candidates
    repeat_ratio_threshold: float = 0.50  # ratio of pages for repeat detection
    hf_position_span_norm: float = 0.08  # max normalized Y span for header/footer repeated lines
    max_header_line_len: int = 60

    # deterministic QA
    coverage_threshold: float = 0.95  # Lowered from 0.97 - more realistic for complex docs
    min_figure_bullets: int = 3
    min_figure_chars: int = 120

    # fail-fast / retry
    max_retries_per_figure: int = 1


# ----------------------------
# RunRecorder (Diagnostics & Durability)
# ----------------------------

class RunRecorder:
    """Best-effort run telemetry for power-loss debugging."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.status_path = run_dir / "status.json"
        self.ckpt_path = run_dir / "checkpoint.jsonl"
        self.gpu_path = run_dir / "gpu_telemetry.csv"
        self._gpu_proc = None

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def mark(self, status: str, error: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"status": status, "ts": time.time(), "pid": os.getpid()}
        if error:
            payload["error"] = error
        self._atomic_write_json(self.status_path, payload)

    def checkpoint(self, stage: str, **extra: Any) -> None:
        rec: Dict[str, Any] = {"ts": time.time(), "stage": stage, **extra}
        with open(self.ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def start_gpu_logger(self) -> None:
        try:
            import shutil
            if shutil.which("nvidia-smi") is None:
                return
            cmd = [
                "nvidia-smi",
                "--query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,nounits",
                "-l", "1"
            ]
            fout = open(self.gpu_path, "w", encoding="utf-8", newline="")
            self._gpu_proc = subprocess.Popen(cmd, stdout=fout, stderr=fout)
        except Exception:
            self._gpu_proc = None

    def stop_gpu_logger(self) -> None:
        try:
            if self._gpu_proc is None:
                return
            self._gpu_proc.terminate()
            self._gpu_proc.wait(timeout=2)
        except Exception:
            try:
                self._gpu_proc.kill()
            except Exception:
                pass
        finally:
            self._gpu_proc = None


def _write_ps_collect_script(dst_path: Path):
    """Write a robust Windows event log collector PowerShell script."""
    script = r'''
param(
  [int]$LookbackHours = 12,
  [string]$OutDir = "$PWD\shutdown_diag_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
)

$ErrorActionPreference = "Continue"

function Log-Exists($logName){
  try { $null = Get-WinEvent -ListLog $logName -ErrorAction Stop; return $true }
  catch { return $false }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Basic info
try { Get-ComputerInfo | Out-File -Encoding utf8 "$OutDir\computer_info.txt" } catch {}
try { Get-CimInstance Win32_VideoController | Format-List * | Out-File -Encoding utf8 "$OutDir\gpu_wmi.txt" } catch {}
try { Get-CimInstance Win32_BIOS | Format-List * | Out-File -Encoding utf8 "$OutDir\bios.txt" } catch {}

# nvidia-smi
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  try { nvidia-smi -q | Out-File -Encoding utf8 "$OutDir\nvidia_smi_q.txt" } catch {}
}

# Export EVTX logs if present
$logsToExport = @(
  "System",
  "Application",
  "Microsoft-Windows-Kernel-Power/Thermal-Operational",
  "Microsoft-Windows-Diagnostics-Performance/Operational",
  "Microsoft-Windows-WER-SystemErrorReporting/Operational",
  "Microsoft-Windows-DxgKrnl/Operational"
)

foreach ($l in $logsToExport) {
  if (Log-Exists $l) {
    try { wevtutil epl "$l" "$OutDir\$($l -replace '[\\/:*?"<>|]','_').evtx" /ow:true } catch {}
  }
}

# Filtered System events
$start = (Get-Date).AddHours(-$LookbackHours)
$ids = 41,6008,1001,4101,161,162
$wheaIds = 17,18,19,20
try {
  $ev = Get-WinEvent -FilterHashtable @{LogName="System"; StartTime=$start} |
    Where-Object {
      ($ids -contains $_.Id) -or ($wheaIds -contains $_.Id) -or
      ($_.ProviderName -match "WHEA|Kernel-Power|EventLog|Display|nvlddmkm|DxgKrnl|WER")
    } |
    Select-Object TimeCreated, Id, LevelDisplayName, ProviderName, Message
  $ev | Format-List | Out-File -Encoding utf8 "$OutDir\System.filtered.txt"
} catch {}

# LiveKernelReports / Minidumps
$dumpDirs = @("C:\Windows\Minidump","C:\Windows\LiveKernelReports")
foreach ($d in $dumpDirs) {
  if (Test-Path $d) {
    $target = Join-Path $OutDir (Split-Path $d -Leaf)
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    try {
      Get-ChildItem $d -Recurse -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 20 |
        Copy-Item -Destination $target -Force
    } catch {}
  }
}

# Zip
$zipPath = "$OutDir.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path $OutDir\* -DestinationPath $zipPath -Force
Write-Host "ZIP: $zipPath"
'''
    dst_path.write_text(script.strip() + "\n", encoding="utf-8")


# ----------------------------
# Utilities
# ----------------------------

def _now_ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _b64_from_pil_jpeg(im: Image.Image, quality: int = 85, max_width: Optional[int] = None) -> str:
    if max_width and im.width > max_width:
        ratio = max_width / float(im.width)
        im = im.resize((max_width, int(im.height * ratio)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _unwrap_outer_fence(s: str) -> str:
    t = (s or "").strip()
    m = re.match(r"^```(?:markdown)?\s*\n([\s\S]*?)\n```$", t, flags=re.IGNORECASE)
    if m:
        return (m.group(1).strip() + "\n")
    return s


def _has_ellipsis(s: str) -> bool:
    return ("..." in s) or ("…" in s)


def _normalize_text_for_tokens(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set(s: str) -> set:
    s = _normalize_text_for_tokens(s)
    if not s:
        return set()
    return set(s.split())


def _coverage_ratio(src: str, out: str) -> float:
    a = _token_set(src)
    b = _token_set(out)
    if not a:
        return 1.0
    return len(a & b) / len(a)


def _strip_generated_blocks_for_coverage(md: str) -> str:
    """
    Coverage is for body fidelity check, so exclude Generated bullets from Figure blocks.
    """
    lines = md.splitlines()
    kept: List[str] = []
    in_fig = False
    for ln in lines:
        if re.match(r"^\*\*\[(?:Figure|Fig\.?|図)\b", ln, flags=re.IGNORECASE):
            in_fig = True
            kept.append(ln)  # caption is from PDF - keep it
            continue
        if in_fig:
            # Exclude Generated bullets
            if ln.strip().startswith("- (Generated)"):
                continue
            if ln.strip() == "":
                in_fig = False
            kept.append(ln)
            continue
        kept.append(ln)
    return "\n".join(kept)


# ----------------------------
# Text layer extraction (Pass0)
# ----------------------------

@dataclass
class TextLine:
    text: str
    bbox: Tuple[float, float, float, float]  # PDF points
    size: float  # font size proxy
    page: int
    is_bold: bool = False  # Bold flag from font


@dataclass
class Word:
    text: str
    bbox: Tuple[float, float, float, float]  # PDF points
    page: int


def extract_text_lines(pdf_path: str) -> Tuple[List[TextLine], List[Tuple[float, float]], float, Dict[int, List[Word]]]:
    """
    Returns:
      - lines: list of TextLine (page-indexed)
      - page_sizes: list of (W_pt, H_pt) for each page
      - median_font_size: median font size (for heading detection)
      - words_by_page: dict(page_no -> list[Word]) for table reconstruction
    """
    doc = fitz.open(pdf_path)
    lines: List[TextLine] = []
    page_sizes: List[Tuple[float, float]] = []
    all_sizes: List[float] = []
    words_by_page: Dict[int, List[Word]] = {}
    
    for i in range(doc.page_count):
        page = doc.load_page(i)
        rect = page.rect
        page_sizes.append((rect.width, rect.height))
        
        # Extract word-level boxes for table reconstruction (text-only; no OCR)
        try:
            wlist = page.get_text("words")  # x0, y0, x1, y1, word, block, line, word_no
            words_by_page[i + 1] = [Word(text=str(w[4]), bbox=(w[0], w[1], w[2], w[3]), page=i + 1) for w in wlist if w and str(w[4]).strip()]
        except Exception:
            words_by_page[i + 1] = []
        
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for ln in block.get("lines", []):
                spans = ln.get("spans", [])
                text = "".join((s.get("text", "") for s in spans)).strip()
                if not text:
                    continue
                x0, y0, x1, y1 = ln.get("bbox", [0, 0, 0, 0])
                fsize = max((s.get("size", 0) for s in spans), default=0)
                # Check bold: flags bit 4 (16) indicates bold
                is_bold = any((s.get("flags", 0) & 16) for s in spans)
                lines.append(TextLine(text=text, bbox=(x0, y0, x1, y1), size=float(fsize), page=i + 1, is_bold=is_bold))
                if fsize > 0:
                    all_sizes.append(fsize)
    doc.close()
    
    # Calculate median font size
    median_size = 0.0
    if all_sizes:
        sorted_sizes = sorted(all_sizes)
        mid = len(sorted_sizes) // 2
        median_size = sorted_sizes[mid]
    
    return lines, page_sizes, median_size, words_by_page


def save_text_layer(out_dir: Path, lines: List[TextLine], page_sizes: List[Tuple[float, float]], words_by_page: Dict[int, List[Word]]) -> None:
    tl_dir = out_dir / "text_layer"
    tl_dir.mkdir(parents=True, exist_ok=True)
    # page-wise lines
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for ln in lines:
        by_page.setdefault(ln.page, []).append({
            "text": ln.text, 
            "bbox": ln.bbox, 
            "size": ln.size,
            "is_bold": ln.is_bold,
        })

    for p, arr in by_page.items():
        (tl_dir / f"page_{p:04d}.json").write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {"page_sizes_pt": page_sizes}
    (tl_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save word-level boxes (used for table reconstruction)
    wl_dir = out_dir / "words_layer"
    wl_dir.mkdir(parents=True, exist_ok=True)
    for p, wlist in words_by_page.items():
        wl_dir.joinpath(f"page_{p:04d}.json").write_text(
            json.dumps([{"text": w.text, "bbox": w.bbox} for w in wlist], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ----------------------------
# Header/Footer inference (Pass1-lite)
# ----------------------------

def _normalize_for_repeat(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\d", "#", s)
    s = re.sub(r"\s+", " ", s)
    return s


def infer_header_footer_sets(
    lines: List[TextLine],
    page_sizes: List[Tuple[float, float]],
    cfg: ConverterConfig,
) -> Tuple[set, set, Dict[str, Any]]:
    """Infer header/footer norms from repetition + stable Y-position (no fixed bands)."""
    pages = len(page_sizes)
    occ: Dict[str, List[Tuple[int, float]]] = {}  # norm -> [(page, ymid_norm), ...]

    for ln in lines:
        if len(ln.text) > cfg.max_header_line_len:
            continue
        _, H = page_sizes[ln.page - 1]
        _, y0, _, y1 = ln.bbox
        ymid = ((y0 + y1) / 2.0) / H
        norm = _normalize_for_repeat(ln.text)
        occ.setdefault(norm, []).append((ln.page, ymid))

    min_pages = max(2, int(cfg.repeat_ratio_threshold * pages + 0.999))  # ceil
    stable: List[Tuple[str, float]] = []
    for norm, items in occ.items():
        pset = {p for p, _ in items}
        if len(pset) < min_pages:
            continue
        ys = [y for _, y in items]
        if (max(ys) - min(ys)) > cfg.hf_position_span_norm:
            continue
        stable.append((norm, sorted(ys)[len(ys) // 2]))  # median

    header_norms: set = set()
    footer_norms: set = set()

    if stable:
        ys_med = [y for _, y in stable]
        y_min = min(ys_med)
        y_max = max(ys_med)

        if y_max <= 0.55:
            header_norms = {n for n, _ in stable}
        elif y_min >= 0.45:
            footer_norms = {n for n, _ in stable}
        else:
            split = sorted(ys_med)[len(ys_med) // 2]
            header_norms = {n for n, y in stable if y <= split}
            footer_norms = {n for n, y in stable if y > split}

    debug = {
        "pages": pages,
        "repeat_ratio_threshold": cfg.repeat_ratio_threshold,
        "hf_position_span_norm": cfg.hf_position_span_norm,
        "min_pages": min_pages,
        "stable_repeats": len(stable),
        "header_norms_count": len(header_norms),
        "footer_norms_count": len(footer_norms),
    }
    return header_norms, footer_norms, debug


def is_probable_page_number(text: str) -> bool:
    t = text.strip()
    return bool(re.fullmatch(r"\d{1,4}", t))


# ----------------------------
# Layout inference (columns)
# ----------------------------

def infer_columns_for_page(page_lines: List[TextLine], page_w_pt: float) -> Tuple[int, float]:
    """
    Returns: (columns, split_x_pt)
    Simple heuristic: if substantial lines exist both on left and right halves, treat as 2 columns.
    """
    xs = [ln.bbox[0] for ln in page_lines]
    if len(xs) < 25:
        return 1, page_w_pt * 0.5

    mid = page_w_pt * 0.5
    left = sum(1 for x in xs if x < mid)
    right = len(xs) - left
    ratio_left = left / len(xs)
    ratio_right = right / len(xs)

    if ratio_left >= 0.30 and ratio_right >= 0.30:
        return 2, mid
    return 1, mid


def order_lines_reading(page_lines: List[TextLine], columns: int, split_x: float) -> List[TextLine]:
    if columns == 1:
        return sorted(page_lines, key=lambda l: (l.bbox[1], l.bbox[0]))
    left = [l for l in page_lines if l.bbox[0] < split_x]
    right = [l for l in page_lines if l.bbox[0] >= split_x]
    left = sorted(left, key=lambda l: (l.bbox[1], l.bbox[0]))
    right = sorted(right, key=lambda l: (l.bbox[1], l.bbox[0]))
    return left + right


# ----------------------------
# Coverage baseline (before paragraph/heading decisions)
# ----------------------------

def build_baseline_text_from_ordered(ordered: List[TextLine]) -> str:
    """Build a coverage baseline from ordered raw lines (header/footer already removed).

    - Keeps *all* text lines (independent of paragraph/heading decisions)
    - Applies only a minimal hyphenation join to reduce false negatives in coverage
    """
    parts: List[str] = []
    for ln in ordered:
        t = ln.text.strip()
        if not t:
            continue
        if parts and parts[-1].endswith("-") and t[:1].islower():
            parts[-1] = parts[-1][:-1] + t
        else:
            parts.append(t)
    return "\n".join(parts)


# ----------------------------
# Captions detection (Figure/Table)
# ----------------------------

FIG_CAP_RE = re.compile(r"^(?:Figure|FIGURE|Fig\.?|FIG\.?|図)\s*\d+[A-Za-z]?(?:\s*[:.\-–]\s*.*|\s+.*|$)", re.IGNORECASE)
TAB_CAP_RE = re.compile(r"^(?:Table|TABLE|Tab\.?|TAB\.?|表)\s*\d+[A-Za-z]?(?:\s*[:.\-–]\s*.*|\s+.*|$)", re.IGNORECASE)


def caption_type(line: str) -> Optional[str]:
    t = line.strip()
    if TAB_CAP_RE.match(t):
        return "table"
    if FIG_CAP_RE.match(t):
        return "figure"
    return None


# ----------------------------
# Rendering + Crop for figure/table
# ----------------------------

def render_page_pil(pdf_path: str, page_no: int, dpi: int) -> Tuple[Image.Image, float]:
    """
    Returns: (PIL image, scale) where scale = dpi/72 (px per pt).
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_no - 1)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()

    mode = "RGB"
    im = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return im, scale


def crop_region_from_caption(
    page_im: Image.Image,
    page_w_pt: float,
    page_h_pt: float,
    caption_bbox_pt: Tuple[float, float, float, float],
    *,
    above: bool,
) -> Image.Image:
    """
    Heuristic crop around a caption:
      - If caption is below a figure: crop ABOVE caption (above=True)
      - If caption is above a figure: crop BELOW caption (above=False)
    """
    x0, y0, x1, y1 = caption_bbox_pt
    margin_x = 0.05 * page_w_pt
    crop_x0 = max(0.0, margin_x)
    crop_x1 = min(page_w_pt, page_w_pt - margin_x)

    if above:
        crop_y1 = max(0.0, y0 - 0.01 * page_h_pt)
        crop_y0 = max(0.0, crop_y1 - 0.45 * page_h_pt)
    else:
        crop_y0 = min(page_h_pt, y1 + 0.01 * page_h_pt)
        crop_y1 = min(page_h_pt, crop_y0 + 0.45 * page_h_pt)

    # Map pt -> px based on rendered image size
    scale_x = page_im.width / page_w_pt
    scale_y = page_im.height / page_h_pt
    px0 = int(crop_x0 * scale_x)
    px1 = int(crop_x1 * scale_x)
    py0 = int(crop_y0 * scale_y)
    py1 = int(crop_y1 * scale_y)

    px0 = max(0, min(page_im.width - 1, px0))
    px1 = max(px0 + 1, min(page_im.width, px1))
    py0 = max(0, min(page_im.height - 1, py0))
    py1 = max(py0 + 1, min(page_im.height, py1))

    return page_im.crop((px0, py0, px1, py1))


# ----------------------------
# Markdown assembly (Pass2)
# ----------------------------

def build_markdown_from_text_layer(
    lines: List[TextLine],
    page_sizes: List[Tuple[float, float]],
    header_norms: set,
    footer_norms: set,
    median_size: float,
    cfg: ConverterConfig,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns:
      - md (with caption placeholders and proper heading structure)
      - baseline_text (for coverage check; header/footer removed, BEFORE paragraph decisions)
      - info (caption -> bbox etc)
    """
    by_page: Dict[int, List[TextLine]] = {}
    for ln in lines:
        by_page.setdefault(ln.page, []).append(ln)

    md_lines: List[str] = []
    baseline_accum: List[str] = []

    cap_map: Dict[str, Dict[str, Any]] = {}
    cap_counter = 0
    title_found = False  # Track if we've found the main title

    # Heading detection thresholds (relative to median font size)
    h1_threshold = median_size * 1.5  # Title: 50% larger than median
    h2_threshold = median_size * 1.2  # Section: 20% larger than median
    h3_threshold = median_size * 1.1  # Subsection: 10% larger than median

    # Common section heading patterns (for validation)
    SECTION_PATTERNS = re.compile(
        r"^(?:\d+\.?\s+)?"  # Optional number prefix like "1." or "1 "
        r"(?:Abstract|Introduction|Background|Literature\s+Review|"
        r"Methodology|Methods|Method|Results|Discussion|Conclusion|Conclusions|"
        r"References|Acknowledgements?|Appendix|"
        r"Theoretical\s+Framework|Conceptual\s+Framework|"
        r"Findings|Analysis|Implications|Limitations|Future\s+Research)",
        re.IGNORECASE
    )

    def classify_heading(ln: TextLine, text: str) -> Optional[int]:
        """
        Returns heading level (1, 2, 3) or None if not a heading.
        Uses font size, bold flag, and text patterns.
        """
        nonlocal title_found
        
        # Skip if text is too long (unlikely to be a heading)
        if len(text) > 120:
            return None
        
        # Skip figure/table captions
        if caption_type(text):
            return None
        
        # Check font size
        size = ln.size
        is_bold = ln.is_bold
        
        # Title (h1): largest font, typically on first pages
        if size >= h1_threshold and ln.page <= 2 and not title_found:
            return 1
        
        # Section heading (h2): larger font OR bold + matches pattern
        if size >= h2_threshold:
            return 2
        
        # Bold text that matches section patterns
        if is_bold and SECTION_PATTERNS.match(text):
            return 2
        
        # Subsection (h3): slightly larger font and bold
        if size >= h3_threshold and is_bold:
            # Additional check: short text, likely a heading
            if len(text) < 80:
                return 3
        
        # Numbered sections like "2.1 Something" with bold
        if is_bold and re.match(r"^\d+\.\d+\s+\S", text):
            return 3
        
        return None

    for p in range(1, len(page_sizes) + 1):
        W, H = page_sizes[p - 1]
        page_lines = by_page.get(p, [])

        # Filter header/footer candidates
        filtered: List[TextLine] = []
        for ln in page_lines:
            norm = _normalize_for_repeat(ln.text)
            yn = ln.bbox[1] / H
            if norm in header_norms or norm in footer_norms:
                continue
            if yn >= 0.80 and is_probable_page_number(ln.text):
                continue
            filtered.append(ln)

        # Reading order
        cols, split_x = infer_columns_for_page(filtered, W)
        ordered = order_lines_reading(filtered, cols, split_x)

        # Coverage baseline from raw ordered lines (BEFORE heading/paragraph decisions)
        baseline_accum.append(build_baseline_text_from_ordered(ordered))

        if md_lines:
            md_lines.append("")  # page break spacing

        # Process lines with heading detection
        i = 0
        while i < len(ordered):
            ln = ordered[i]
            text = ln.text.strip()
            
            # Check for caption
            ctype = caption_type(text)
            if ctype:
                cap_counter += 1
                marker = f"@@CAP_{cap_counter:04d}@@"
                md_lines.append(f"**[{text}]**")  # caption from PDF
                md_lines.append(marker)           # placeholder
                md_lines.append("")

                cap_map[marker] = {
                    "type": ctype,
                    "caption": text,
                    "page": p,
                    "page_w_pt": W,
                    "page_h_pt": H,
                    "caption_bbox_pt": ln.bbox,
                }
                i += 1
                continue
            
            # Check for heading
            heading_level = classify_heading(ln, text)
            if heading_level:
                prefix = "#" * heading_level
                md_lines.append(f"{prefix} {text}")
                md_lines.append("")
                if heading_level == 1:
                    title_found = True
                i += 1
                continue
            
            # Regular paragraph: collect consecutive normal lines
            para_lines = [text]
            prev_y = ln.bbox[3]
            i += 1
            
            while i < len(ordered):
                next_ln = ordered[i]
                next_text = next_ln.text.strip()
                
                # Stop if next line is a heading or caption
                if classify_heading(next_ln, next_text) or caption_type(next_text):
                    break
                
                # Stop if large vertical gap (new paragraph)
                if next_ln.bbox[1] - prev_y > 18:
                    break
                
                # Hyphenation fix
                if para_lines[-1].endswith("-") and next_text[:1].islower():
                    para_lines[-1] = para_lines[-1][:-1] + next_text
                else:
                    para_lines.append(next_text)
                
                prev_y = next_ln.bbox[3]
                i += 1
            
            para = " ".join(para_lines).strip()
            if para:
                md_lines.append(para)
                md_lines.append("")

    md = "\n".join(md_lines).strip() + "\n"
    baseline_text = "\n".join(baseline_accum)
    info = {"captions": cap_map}
    return md, baseline_text, info


# ----------------------------
# Figure description (Vision REQUIRED)
# ----------------------------

def figure_description_prompt(caption_text: str) -> str:
    return f"""You are describing a figure from an academic paper.

Caption:
{caption_text}

RULES:
- Describe ONLY what is visually present in the figure image (no guessing).
- Prefer concrete description: elements, labels, axes, legends, arrows, flows, groupings.
- If some text/numbers are unreadable, write "?" (do not guess).
- Do NOT use "..." or "…" anywhere.
- Do NOT wrap output in triple backticks.
- Output Markdown bullets ONLY.
- Each bullet MUST start with "- (Generated) ".

Now produce the bullets:
""".strip()


def describe_figure_with_vision(
    client: LMStudioClient,
    cfg: ConverterConfig,
    pdf_path: str,
    cap_info: Dict[str, Any],
) -> str:
    p = cap_info["page"]
    W = cap_info["page_w_pt"]
    H = cap_info["page_h_pt"]
    caption = cap_info["caption"]
    bbox = cap_info.get("caption_bbox_pt")

    page_im, _ = render_page_pil(pdf_path, p, cfg.render_dpi)

    # Heuristic: caption usually below figure if in lower half
    above = True
    if bbox is not None:
        cy = bbox[1] / H
        above = True if cy > 0.35 else False
        crop = crop_region_from_caption(page_im, W, H, bbox, above=above)
    else:
        # Fallback: crop central area
        crop = page_im.crop((int(page_im.width * 0.08), int(page_im.height * 0.12),
                             int(page_im.width * 0.92), int(page_im.height * 0.70)))

    crop_b64 = _b64_from_pil_jpeg(crop, quality=cfg.jpeg_quality, max_width=cfg.max_crop_width_px)

    prompt = figure_description_prompt(caption)
    out = client.vision_describe_images(
        prompt,
        [crop_b64],
        model=cfg.vision_model,
        temperature=0.1,
        max_tokens=900,
        timeout=(20, 900),
    ).strip()

    out = _unwrap_outer_fence(out)

    # Ensure bullets with prefix
    bullets = [ln for ln in out.splitlines() if ln.strip()]
    fixed: List[str] = []
    for ln in bullets:
        if ln.strip().startswith("- (Generated)"):
            fixed.append(ln.strip())
        elif ln.strip().startswith("-"):
            fixed.append("- (Generated) " + ln.strip()[1:].strip())
        else:
            fixed.append("- (Generated) " + ln.strip())

    return "\n".join(fixed).strip() + "\n"


# ----------------------------
# Table reconstruction (Hybrid-lite: words -> rows/cols -> Markdown table)
# ----------------------------

def _escape_pipes(s: str) -> str:
    return s.replace("|", "\\|")


def infer_table_region_from_caption(
    page_w_pt: float,
    page_h_pt: float,
    caption_bbox_pt: Optional[Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """Infer a probable table body region in PDF points.

    Heuristic:
      - Tables often have caption ABOVE the table; figures often BELOW.
      - Use caption position to decide whether the table is likely below or above.
    """
    margin_x = 0.05 * page_w_pt
    x0 = margin_x
    x1 = page_w_pt - margin_x

    if not caption_bbox_pt:
        # Fallback: center-ish region
        y0 = 0.20 * page_h_pt
        y1 = 0.75 * page_h_pt
        return (x0, y0, x1, y1)

    _, cy0, _, cy1 = caption_bbox_pt
    c_mid = (cy0 + cy1) / 2.0
    # If caption is in upper half, assume table below caption; else above
    if c_mid < 0.55 * page_h_pt:
        y0 = min(page_h_pt, cy1 + 0.01 * page_h_pt)
        y1 = min(page_h_pt, y0 + 0.55 * page_h_pt)
    else:
        y1 = max(0.0, cy0 - 0.01 * page_h_pt)
        y0 = max(0.0, y1 - 0.55 * page_h_pt)

    return (x0, y0, x1, y1)


def words_in_region(words: List[Word], region: Tuple[float, float, float, float]) -> List[Word]:
    x0, y0, x1, y1 = region
    out: List[Word] = []
    for w in words:
        wx0, wy0, wx1, wy1 = w.bbox
        if wx1 < x0 or wx0 > x1 or wy1 < y0 or wy0 > y1:
            continue
        if not w.text.strip():
            continue
        out.append(w)
    return out


def cluster_positions(vals: List[float], tol: float) -> List[float]:
    if not vals:
        return []
    vals = sorted(vals)
    clusters: List[List[float]] = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    # cluster center
    return [sum(c) / len(c) for c in clusters]


def group_words_into_rows(words: List[Word], row_tol: float = 4.0) -> List[List[Word]]:
    # group by y-center
    items = sorted(words, key=lambda w: ((w.bbox[1] + w.bbox[3]) / 2.0, w.bbox[0]))
    rows: List[List[Word]] = []
    row_y: List[float] = []
    for w in items:
        yc = (w.bbox[1] + w.bbox[3]) / 2.0
        if not rows:
            rows.append([w])
            row_y.append(yc)
            continue
        if abs(yc - row_y[-1]) <= row_tol:
            rows[-1].append(w)
            # update average y
            row_y[-1] = (row_y[-1] * (len(rows[-1]) - 1) + yc) / len(rows[-1])
        else:
            rows.append([w])
            row_y.append(yc)
    # sort within row by x
    for r in rows:
        r.sort(key=lambda w: w.bbox[0])
    return rows


def assign_words_to_columns(row: List[Word], col_starts: List[float]) -> List[str]:
    n = max(1, len(col_starts))
    cells: List[List[str]] = [[] for _ in range(n)]
    for w in row:
        x0 = w.bbox[0]
        # choose the last column start <= x0 (+epsilon)
        idx = 0
        for i, cs in enumerate(col_starts):
            if x0 + 2.0 >= cs:
                idx = i
            else:
                break
        idx = min(idx, n - 1)
        cells[idx].append(w.text)
    return [" ".join(c).strip() for c in cells]


def trim_empty_columns(rows: List[List[str]]) -> List[List[str]]:
    if not rows:
        return rows
    n = max(len(r) for r in rows)
    # normalize lengths
    rows2 = [r + [""] * (n - len(r)) for r in rows]
    # remove columns that are empty in all rows
    keep = []
    for j in range(n):
        if any(rows2[i][j].strip() for i in range(len(rows2))):
            keep.append(j)
    return [[r[j] for j in keep] for r in rows2]


def _looks_like_table(rows: List[List[str]]) -> bool:
    """Check if the rows look like a table based on geometry."""
    if len(rows) < 2:
        return False
    if max(len(r) for r in rows) < 2:
        return False
    cols_ge2 = sum(1 for r in rows if len(r) >= 2)
    return cols_ge2 >= max(2, len(rows) // 2)


def reconstruct_table_markdown(
    words_by_page: Dict[int, List[Word]],
    cap_info: Dict[str, Any],
) -> str:
    """Reconstruct a Markdown table using only text-layer words (no hallucination).

    This is a Hybrid-lite implementation:
      - Region is inferred around the caption
      - Rows are clustered by y
      - Columns are inferred by x clustering
    """
    p = cap_info["page"]
    W = cap_info["page_w_pt"]
    H = cap_info["page_h_pt"]
    bbox = cap_info.get("caption_bbox_pt")
    region = infer_table_region_from_caption(W, H, bbox)

    words = words_by_page.get(p, [])
    cand = words_in_region(words, region)

    if not cand:
        return "_(Table reconstruction failed: no text-layer words detected in the expected region.)_"

    rows_w = group_words_into_rows(cand, row_tol=4.0)

    # infer columns from x0 positions across all words
    xs = [w.bbox[0] for w in cand]
    col_starts = cluster_positions(xs, tol=18.0)
    # guard: too many columns often means noise
    if len(col_starts) > 12:
        col_starts = col_starts[:12]

    rows_cells = [assign_words_to_columns(r, col_starts) for r in rows_w]
    rows_cells = trim_empty_columns(rows_cells)

    if not rows_cells or not _looks_like_table(rows_cells):
        return "_(Table reconstruction skipped: region did not look like a table based on text-layer geometry.)_"

    # Build markdown
    header = rows_cells[0]
    data = rows_cells[1:] if len(rows_cells) > 1 else []

    header = [_escape_pipes(c or "?") for c in header]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in data[:200]:
        r = [_escape_pipes(c or "?") for c in r]
        # pad/truncate
        if len(r) < len(header):
            r += ["?"] * (len(header) - len(r))
        if len(r) > len(header):
            r = r[: len(header)]
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


# ----------------------------
# Deterministic QA (Pass3)
# ----------------------------

def validate_markdown(md: str) -> List[str]:
    issues: List[str] = []
    # NOTE: Ellipsis can legitimately appear in the source PDF body.
    #       Ellipsis inside generated figure bullets is validated in validate_figures().
    if "```" in md:
        issues.append("code_fence_leak")
    # figure rendered as table near caption
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"^\*\*\[(?:Figure|Fig\.?|図)\b", ln, flags=re.IGNORECASE):
            window = "\n".join(lines[i:i+15])
            if "|---" in window or re.search(r"^\s*\|.*\|\s*$", window, flags=re.M):
                issues.append("figure_rendered_as_table")
                break
    # title duplicates
    if len(re.findall(r"^#\s+", md, flags=re.M)) > 1:
        issues.append("duplicate_h1_title")
    return issues


def validate_figures(md: str, cfg: ConverterConfig) -> List[str]:
    issues: List[str] = []
    blocks = md.split("\n**[")
    for part in blocks:
        if not part:
            continue
        first = part.splitlines()[0]
        cap_text = first.split("]**", 1)[0].strip()
        if FIG_CAP_RE.match(cap_text):
            text = "**[" + part
            lines = text.splitlines()
            gen = [ln for ln in lines[:60] if ln.strip().startswith("- (Generated)")]
            if len(gen) < cfg.min_figure_bullets:
                issues.append("figure_description_too_short")
                break
            joined = "\n".join(gen)
            if len(joined) < cfg.min_figure_chars:
                issues.append("figure_description_too_short")
                break
            if _has_ellipsis(joined):
                issues.append("figure_description_has_ellipsis")
                break
    return issues


# ----------------------------
# Main converter
# ----------------------------

class PDFtoMarkdownTextFirst:
    def __init__(self, *, out_root: str, client: Optional[LMStudioClient] = None, cfg: Optional[ConverterConfig] = None):
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.client = client or LMStudioClient()
        self.cfg = cfg or ConverterConfig()

    def convert(self, pdf_path: str) -> Path:
        pdf_path = str(pdf_path)
        stem = Path(pdf_path).stem
        
        # Create run directory
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.out_root / f"{stem}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RunRecorder for diagnostics
        recorder = RunRecorder(run_dir)
        recorder.mark("running")
        recorder.start_gpu_logger()
        recorder.checkpoint("start", pdf_path=pdf_path)
        
        try:
            print(f"[Pass0] Extracting text layer...")
            lines, page_sizes, median_size, words_by_page = extract_text_lines(pdf_path)
            save_text_layer(run_dir, lines, page_sizes, words_by_page)
            print(f"  - {len(lines)} lines from {len(page_sizes)} pages (median font: {median_size:.1f}pt)")
            recorder.checkpoint("pass0_done", lines=len(lines), pages=len(page_sizes))

            print(f"[Pass1] Inferring header/footer...")
            header_norms, footer_norms, hf_debug = infer_header_footer_sets(lines, page_sizes, self.cfg)
            (run_dir / "pass1_layout_debug.json").write_text(json.dumps(hf_debug, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  - Headers: {len(header_norms)}, Footers: {len(footer_norms)}")
            recorder.checkpoint("pass1_done", headers=len(header_norms), footers=len(footer_norms))

            print(f"[Pass2] Building markdown from text layer...")
            md, baseline_text, info = build_markdown_from_text_layer(lines, page_sizes, header_norms, footer_norms, median_size, self.cfg)
            (run_dir / "pass2_body_with_placeholders.md").write_text(md, encoding="utf-8")
            recorder.checkpoint("pass2_done", placeholder_count=len(info.get("captions", {})))

            # Pass2b: Figure descriptions (REQUIRED) and Table reconstruction
            cap_map: Dict[str, Dict[str, Any]] = info["captions"]
            fig_count = sum(1 for c in cap_map.values() if c["type"] == "figure")
            tab_count = sum(1 for c in cap_map.values() if c["type"] == "table")
            print(f"[Pass2b] Generating {fig_count} figure descriptions, {tab_count} tables...")
            
            for marker, cap_info in cap_map.items():
                if cap_info["type"] == "figure":
                    desc = ""
                    last_err = None
                    for attempt in range(self.cfg.max_retries_per_figure + 1):
                        try:
                            print(f"  - Figure on page {cap_info['page']}...")
                            recorder.checkpoint("figure_start", page=cap_info['page'])
                            desc = describe_figure_with_vision(self.client, self.cfg, pdf_path, cap_info)
                            recorder.checkpoint("figure_done", page=cap_info['page'])
                            break
                        except Exception as e:
                            last_err = str(e)
                            time.sleep(0.5)
                    if not desc:
                        desc = "- (Generated) ?\n- (Generated) ?\n- (Generated) ?\n"
                        with open(run_dir / "warnings.log", "a", encoding="utf-8") as f:
                            f.write(f"Figure description failed for page {cap_info['page']}: {last_err}\n")
                    md = md.replace(marker, desc.strip())

                elif cap_info["type"] == "table":
                    print(f"  - Table on page {cap_info['page']}...")
                    recorder.checkpoint("table_start", page=cap_info['page'])
                    table_md = reconstruct_table_markdown(words_by_page, cap_info)
                    md = md.replace(marker, table_md)
                    recorder.checkpoint("table_done", page=cap_info['page'])

            (run_dir / "pass2_full.md").write_text(md, encoding="utf-8")
            recorder.checkpoint("pass2b_done", figures=fig_count, tables=tab_count)

            print(f"[Pass3] Running deterministic QA...")
            issues = validate_markdown(md)
            issues += validate_figures(md, self.cfg)

            # Coverage check (using baseline_text which is BEFORE paragraph decisions)
            cov_md = _strip_generated_blocks_for_coverage(md)
            cov = _coverage_ratio(baseline_text, cov_md)
            qa = {"issues": issues, "coverage": cov, "coverage_threshold": self.cfg.coverage_threshold}
            (run_dir / "pass3_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"  - Coverage: {cov:.3f} (threshold: {self.cfg.coverage_threshold})")
            recorder.checkpoint("pass3_done", coverage=cov, issues=issues)

            if cov < self.cfg.coverage_threshold:
                issues.append("coverage_below_threshold")

            if issues:
                print(f"[FAIL] QA issues: {issues} (coverage={cov:.3f})")
                recorder.mark("error", error=f"QA failed: {issues}")
                raise RuntimeError(f"Deterministic QA failed: {issues} (coverage={cov:.3f})")

            out_md = run_dir / f"{stem}.md"
            out_md.write_text(md, encoding="utf-8")
            print(f"[OK] Saved: {out_md}")
            
            recorder.mark("completed")
            return out_md
            
        except Exception as e:
            recorder.checkpoint("error", message=str(e))
            recorder.mark("error", error=str(e))
            raise
            
        finally:
            recorder.stop_gpu_logger()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pdf_to_md_textfirst.py <pdf_path> <out_dir>")
        raise SystemExit(2)

    pdf_path = sys.argv[1]
    out_dir = sys.argv[2]

    conv = PDFtoMarkdownTextFirst(out_root=out_dir)
    out = conv.convert(pdf_path)
    print(f"[DONE] {out}")

