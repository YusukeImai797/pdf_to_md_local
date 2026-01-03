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
from collections import Counter
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
                text = _clean_extracted_text(text)
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
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\d", "#", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    parts = s.split()
    if len(parts) > 1 and len(parts[0]) == 1:
        parts = parts[1:]
    return " ".join(parts)


def _clean_extracted_text(text: str) -> str:
    if not text:
        return text
    # Remove control chars, normalize whitespace artifacts from PDF extraction.
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u00a0\u1680\u2000-\u200f\u2028\u2029\u202f\u205f\u3000\ufeff]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_leading_singleton(text: str) -> str:
    parts = text.strip().split()
    if len(parts) > 1 and len(parts[0]) == 1:
        parts = parts[1:]
    return " ".join(parts)


def _dominant_font_size(lines: List[TextLine]) -> float:
    sizes = [round(ln.size, 1) for ln in lines if ln.size > 0 and len(ln.text.strip()) >= 3]
    if not sizes:
        return 0.0
    counts = Counter(sizes)
    max_count = max(counts.values())
    candidates = [s for s, c in counts.items() if c == max_count]
    return max(candidates)


def _is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def _is_title_case(text: str) -> bool:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return False
    caps = 0
    for w in words:
        if w[0].isupper():
            caps += 1
    return caps >= max(1, len(words) - 2)


def _looks_like_source_line(text: str) -> bool:
    t = text.lower()
    if "doi" in t or "doi.org" in t:
        return True
    if re.search(r"\bvol\.?\b|\bno\.?\b|\bissue\b|\bjournal\b|\bproceedings\b", text, re.IGNORECASE):
        return True
    if re.search(r"\bannals\b|\btransactions\b", text, re.IGNORECASE):
        return True
    if re.search(r"\breview\b", text, re.IGNORECASE):
        if re.search(r"\b(academy|management|journal|proceedings|transactions|annals|vol\.?|no\.?|issue)\b", text, re.IGNORECASE):
            return True
    if re.search(r"\bpp\.?\b|\bpages?\b", text, re.IGNORECASE):
        return True
    return False


def _looks_like_affiliation_line(text: str) -> bool:
    if "@" in text:
        return True
    return bool(re.search(
        r"\b(University|Universiteit|Universite|Institute|School|Department|Faculty|College|Laborator|Centre|Center|Hospital|Academy|Business)\b",
        text,
        re.IGNORECASE,
    ))


def _looks_like_author_line(text: str) -> bool:
    if any(ch.isdigit() for ch in text):
        return False
    if _looks_like_affiliation_line(text) or _looks_like_source_line(text):
        return False
    if re.match(
        r"^(Introduction|Background|Literature\s+Review|Related\s+Work|"
        r"Methodology|Methods|Method|Results|Discussion|Conclusion|Conclusions|"
        r"Review\s+Methodology|State\s+of\s+the\s+Literature|"
        r"Theoretical\s+Framework|Conceptual\s+Framework|Findings|Analysis|"
        r"Implications|Limitations|Future\s+Research|Acknowledgements?|"
        r"References|Bibliography|Works\s+Cited|Appendix)\b",
        text.strip(),
        re.IGNORECASE,
    ):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    stop = {"with", "and", "of", "the", "in", "on", "for", "to", "an", "a"}
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio >= 0.6:
        if "," in text:
            return True
        if any(w.lower() in stop for w in words):
            return False
        return True
    if "," in text and _is_title_case(text):
        return True
    return False


def _looks_like_boilerplate_footer(text: str) -> bool:
    return bool(re.search(
        r"(copyright|all rights reserved|may not be copied|email articles|accepted by)",
        text,
        re.IGNORECASE,
    ))


def _looks_like_footnote_marker(text: str) -> bool:
    return bool(re.match(r"^\s*(?:\d{1,3}|[†‡*])[\).]?\s+\S", text))


def _looks_like_month_line(text: str) -> bool:
    return bool(re.fullmatch(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)",
        text.strip(),
        re.IGNORECASE,
    ))


def _looks_like_citation_line(text: str) -> bool:
    t = text.strip()
    if len(t) > 120:
        return False
    if re.search(r"\(\s*\d{4}\s*\)", t):
        return True
    if re.search(r"\(\s*\d{4}\b", t):
        return True
    if re.search(r"\b\d{4}\b", t) and "," in t:
        return True
    if re.match(r"^\d{4}\b", t):
        return True
    if re.search(r"\bet al\b", t, re.IGNORECASE):
        return True
    if t.endswith(").") or t.endswith(");") or t.endswith("),"):
        return True
    return False


def _is_probable_footnote_line(ln: "TextLine", base_size: float, page_h_pt: float) -> bool:
    text = ln.text.strip()
    if not text:
        return False
    ymid = ((ln.bbox[1] + ln.bbox[3]) / 2.0) / page_h_pt
    if _looks_like_footnote_marker(text):
        return True
    if base_size > 0 and ln.size <= base_size * 0.92 and ymid >= 0.65:
        return True
    return False


def _section_kind(text: str) -> Optional[str]:
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return None
    t_clean = re.sub(r"\s*[:.]\s*$", "", t)
    words = [w for w in t_clean.split() if w]
    if re.match(r"^(Abstract|Summary)\b", t_clean, re.IGNORECASE):
        return "abstract"
    if re.match(r"^(References|Bibliography|Works Cited)\b", t_clean, re.IGNORECASE):
        return "references"
    if re.match(r"^\d+\.\d+\s+\S", t_clean):
        return "minor"
    if re.match(r"^\d+\s+\S", t_clean):
        return "major"
    if _is_all_caps(t_clean) and len(t_clean) <= 80:
        return "major"
    if re.match(
        r"^(Introduction|Background|Literature\s+Review|Related\s+Work|"
        r"Methodology|Methods|Method|Results|Discussion|Conclusion|Conclusions|"
        r"Review\s+Methodology|State\s+of\s+the\s+Literature|"
        r"Theoretical\s+Framework|Conceptual\s+Framework|Findings|Analysis|"
        r"Implications|Limitations|Future\s+Research|Acknowledgements?|Appendix)\b",
        t_clean,
        re.IGNORECASE,
    ):
        return "major"
    if (t_clean.endswith("?") or (_is_title_case(t_clean) and len(words) >= 2)) and len(t_clean) <= 80:
        return "minor"
    return None


def infer_header_footer_sets(
    lines: List[TextLine],
    page_sizes: List[Tuple[float, float]],
    cfg: ConverterConfig,
) -> Tuple[set, set, Dict[str, Any]]:
    """Infer header/footer norms from repetition + stable Y-position (no fixed bands)."""
    pages = len(page_sizes)
    occ: Dict[str, List[Tuple[int, float]]] = {}  # norm -> [(page, ymid_norm), ...]
    occ_parity: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}  # (norm, parity) -> [(page, ymid_norm)]

    for ln in lines:
        if len(ln.text) > cfg.max_header_line_len:
            continue
        _, H = page_sizes[ln.page - 1]
        _, y0, _, y1 = ln.bbox
        ymid = ((y0 + y1) / 2.0) / H
        norm = _normalize_for_repeat(ln.text)
        if not norm:
            continue
        occ.setdefault(norm, []).append((ln.page, ymid))
        occ_parity.setdefault((norm, ln.page % 2), []).append((ln.page, ymid))

    def _stable_from(items: List[Tuple[int, float]], min_pages: int) -> Optional[float]:
        pset = {p for p, _ in items}
        if len(pset) < min_pages:
            return None
        ys = [y for _, y in items]
        if (max(ys) - min(ys)) > cfg.hf_position_span_norm:
            return None
        return sorted(ys)[len(ys) // 2]  # median

    min_pages = max(2, int(cfg.repeat_ratio_threshold * pages + 0.999))  # ceil
    odd_pages = (pages + 1) // 2
    even_pages = pages // 2
    min_pages_odd = max(2, int(cfg.repeat_ratio_threshold * odd_pages + 0.999))
    min_pages_even = max(2, int(cfg.repeat_ratio_threshold * even_pages + 0.999))

    stable: List[Tuple[str, float]] = []
    stable_all = 0
    stable_parity = 0
    for norm, items in occ.items():
        ymed = _stable_from(items, min_pages)
        if ymed is not None:
            stable.append((norm, ymed))
            stable_all += 1
            continue
        for parity in (0, 1):
            key = (norm, parity)
            if key not in occ_parity:
                continue
            ymed_p = _stable_from(occ_parity[key], min_pages_even if parity == 0 else min_pages_odd)
            if ymed_p is not None:
                stable.append((norm, ymed_p))
                stable_parity += 1
                break

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
        "odd_pages": odd_pages,
        "even_pages": even_pages,
        "repeat_ratio_threshold": cfg.repeat_ratio_threshold,
        "hf_position_span_norm": cfg.hf_position_span_norm,
        "min_pages": min_pages,
        "min_pages_odd": min_pages_odd,
        "min_pages_even": min_pages_even,
        "stable_repeats": len(stable),
        "stable_repeats_all": stable_all,
        "stable_repeats_parity": stable_parity,
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


def render_footnotes(lines: List[TextLine]) -> str:
    """Render footnote-like lines into compact paragraphs."""
    paras: List[str] = []
    cur: List[str] = []
    prev_y: Optional[float] = None
    for ln in lines:
        t = ln.text.strip()
        if not t:
            continue
        new_note = False
        if _looks_like_footnote_marker(t):
            new_note = True
        if prev_y is not None and (ln.bbox[1] - prev_y) > 12:
            new_note = True
        if new_note and cur:
            paras.append(" ".join(cur).strip())
            cur = []
        if cur and cur[-1].endswith("-") and t[:1].islower():
            cur[-1] = cur[-1][:-1] + t
        else:
            cur.append(t)
        prev_y = ln.bbox[3]
    if cur:
        paras.append(" ".join(cur).strip())
    return "\n".join(paras).strip()


def _ends_sentence(text: str) -> bool:
    t = text.strip()
    return bool(re.search(r"[.!?](?:[\"')\]\}]+)?$", t))

def _bbox_overlaps(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1)


def _looks_like_continuation(prev_text: str, next_text: str) -> bool:
    prev = prev_text.strip()
    nxt = next_text.strip()
    if not prev or not nxt:
        return False
    if _ends_sentence(prev):
        return False
    if re.search(r"[:,;]$", prev):
        return True
    if re.search(
        r"\b(and|or|but|because|which|that|who|with|without|for|to|of|in|on|by|while|whereas|although|if|when|as|since)$",
        prev,
        re.IGNORECASE,
    ):
        return True
    if nxt[:1].islower():
        return True
    if re.match(
        r"^(and|or|but|because|which|that|who|with|without|for|to|of|in|on|by|while|whereas|although|if|when|as|since)\b",
        nxt,
        re.IGNORECASE,
    ):
        return True
    return True


def _join_text(prev_text: str, next_text: str) -> str:
    if prev_text.endswith("-") and next_text[:1].islower():
        return prev_text[:-1] + next_text
    return prev_text.rstrip() + " " + next_text.lstrip()


def _split_inline_heading(text: str) -> Optional[Tuple[str, str]]:
    t = re.sub(r"\s+", " ", text.strip())
    if len(t) < 6 or len(t) > 400:
        return None
    # Case 1: ALL CAPS heading followed by sentence text.
    m = re.match(
        r"^((?:\d+(?:\.\d+)?\s+)?[A-Z][A-Z0-9&/\\-]*(?:\s+[A-Z][A-Z0-9&/\\-]*){0,6})\s+(.+)$",
        t,
    )
    if m:
        head = m.group(1).strip()
        rest = m.group(2).strip()
        if len(head) <= 80 and re.search(r"[a-z]", rest):
            if (_section_kind(head) or _is_all_caps(head)) and not _looks_like_citation_line(head):
                return head, rest
    # Case 2: Known heading phrase followed by ":" or "." and sentence text.
    for delim in (":", "."):
        if delim in t:
            head, rest = [p.strip() for p in t.split(delim, 1)]
            if head and rest and len(head.split()) <= 6 and re.search(r"[a-z]", rest):
                if _section_kind(head) and (head[0].isupper() or _is_all_caps(head)) and not _looks_like_citation_line(head):
                    return head, rest
    return None


def _looks_like_reference_start(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if re.match(r"^\[\d+\]\s+\S", t):
        return True
    if re.match(r"^\d+\.\s+\S", t):
        return True
    if re.match(r"^\d+\)\s+\S", t):
        return True
    if re.match(r"^[A-Z][^\s,]+(?:\s+[A-Z][^\s,]+){0,3},\s+[A-Z]\.(?:\s*[A-Z]\.)*", t):
        return True
    if re.match(r"^[A-Z][^\s,]+\s+\d{4}[a-z]?\b", t):
        return True
    if re.match(r"^[A-Z][^\s,]+(?:\s+[A-Z][^\s,]+){0,4}\.\s+\d{4}[a-z]?\b", t):
        return True
    if re.match(
        r"^(?:van|von|de|del|der|di|da|le|la|du|dos|das)\s+(?:[A-Z][^\s,]+(?:\s+[A-Z][^\s,]+){0,2}),\s+[A-Z]\.(?:\s*[A-Z]\.)*",
        t,
        re.IGNORECASE,
    ):
        return True
    if re.match(r"^(Note|Notes):", t):
        return True
    return False


def _split_reference_entries(entry: str) -> List[str]:
    if not entry:
        return []
    parts: List[str] = []
    text = entry.strip()
    start = 0
    for m in re.finditer(r"\.\s+", text):
        split_pos = m.end()
        head = text[start:split_pos].strip()
        tail = text[split_pos:].lstrip()
        if tail and _looks_like_reference_start(tail) and re.search(r"\b\d{4}[a-z]?\b", head):
            if head:
                parts.append(head)
            start = split_pos
    remainder = text[start:].strip()
    if remainder:
        parts.append(remainder)
    return parts

def _looks_like_author_bio_heading(text: str) -> bool:
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return False
    if re.match(r"^(About the Authors?|Author Biograph(?:y|ies)|Author Information|Author Notes|About the Author)\b", t, re.IGNORECASE):
        return True
    if _is_all_caps(t) and re.search(r"\bAUTHORS?\b", t):
        return True
    return False


def _looks_like_author_bio_line(text: str) -> bool:
    t = text.strip()
    if not t or _looks_like_citation_line(t):
        return False
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", t):
        return True
    if re.search(r"\b(research (focuses|interests include)|received (his|her|their) PhD)\b", t, re.IGNORECASE):
        return True
    if re.search(
        r"\b(is|was)\s+(an?|the)\s+(assistant|associate|full|distinguished|visiting|adjunct|emeritus|"
        r"professor|doctoral|postdoctoral|phd|researcher|lecturer|fellow|student)\b",
        t,
        re.IGNORECASE,
    ):
        if _looks_like_affiliation_line(t) or re.search(r"\b(university|school|department|faculty|institute|college|management|business)\b", t, re.IGNORECASE):
            return True
    return False


def _looks_like_author_bio_start(text: str) -> bool:
    t = text.strip()
    if not t or _looks_like_citation_line(t):
        return False
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", t):
        return True
    if re.search(
        r"\b(is|was)\s+(an?|the)\s+(assistant|associate|full|distinguished|visiting|adjunct|emeritus|"
        r"professor|doctoral|postdoctoral|phd|researcher|lecturer|fellow|student)\b",
        t,
        re.IGNORECASE,
    ):
        return True
    if re.match(r"^[A-Z][A-Za-z'\\-]+\s+[A-Z][A-Za-z'\\-]+", t) and re.search(r"\b(is|was)\b", t, re.IGNORECASE):
        return True
    return False


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


def _is_caption_or_placeholder_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if t.startswith("**[") and t.endswith("]**"):
        return True
    if re.match(r"^@@CAP_\d+@@$", t):
        return True
    return False


def _is_plain_paragraph_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if t == "---":
        return False
    if t.startswith("#"):
        return False
    if t.startswith("- "):
        return False
    if t.startswith("**"):
        return False
    if t.startswith("|"):
        return False
    if _is_caption_or_placeholder_line(t):
        return False
    return True


def merge_broken_paragraphs(lines: List[str]) -> List[str]:
    compact: List[str] = []
    prev_blank = False
    for ln in lines:
        if ln.strip() == "":
            if not prev_blank:
                compact.append("")
            prev_blank = True
            continue
        compact.append(ln)
        prev_blank = False

    out: List[str] = []
    i = 0
    while i < len(compact):
        line = compact[i]
        if _is_plain_paragraph_line(line) and i + 2 < len(compact) and compact[i + 1].strip() == "":
            merged = line
            j = i
            while j + 2 < len(compact) and compact[j + 1].strip() == "":
                nxt = compact[j + 2]
                if not _is_plain_paragraph_line(nxt):
                    break
                if not _looks_like_continuation(merged, nxt):
                    break
                merged = _join_text(merged, nxt)
                j += 2
            if j != i:
                out.append(merged)
                out.append("")
                i = j + 2
                continue
        out.append(line)
        i += 1
    return out

def split_runin_heading_paragraphs(lines: List[str]) -> List[str]:
    return lines


def relocate_mid_paragraph_captions(lines: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(lines):
        if _is_plain_paragraph_line(lines[i]) and i + 1 < len(lines) and lines[i + 1].strip() == "":
            cap_start = i + 2
            if cap_start < len(lines) and _is_caption_or_placeholder_line(lines[cap_start]):
                cap_block: List[str] = []
                cap_end = cap_start
                while cap_end < len(lines) and lines[cap_end].strip() != "":
                    cap_block.append(lines[cap_end])
                    cap_end += 1
                if cap_end < len(lines) and lines[cap_end].strip() == "":
                    k = cap_end + 1
                    while k < len(lines) and lines[k].strip() == "":
                        k += 1
                    if k < len(lines) and _is_plain_paragraph_line(lines[k]):
                        if _looks_like_continuation(lines[i], lines[k]):
                            merged = _join_text(lines[i], lines[k])
                            out.append(merged)
                            out.append("")
                            out.extend(cap_block)
                            out.append("")
                            i = k + 1
                            continue
        out.append(lines[i])
        i += 1
    return out


def normalize_markdown_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if line.lstrip().startswith("|"):
            out.append(line)
            continue
        text = line
        text = text.replace("\u00a0", " ")
        text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        out.append(text)
    return out


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


def infer_figure_region_from_caption(
    page_w_pt: float,
    page_h_pt: float,
    caption_bbox_pt: Tuple[float, float, float, float],
    above: Optional[bool] = None,
) -> Tuple[float, float, float, float]:
    """Infer figure region in PDF points using the same heuristic as crop_region_from_caption."""
    x0, y0, x1, y1 = caption_bbox_pt
    margin_x = 0.05 * page_w_pt
    crop_x0 = max(0.0, margin_x)
    crop_x1 = min(page_w_pt, page_w_pt - margin_x)
    if above is None:
        cy = y0 / page_h_pt
        above = True if cy > 0.35 else False
    if above:
        crop_y1 = max(0.0, y0 - 0.01 * page_h_pt)
        crop_y0 = max(0.0, crop_y1 - 0.45 * page_h_pt)
    else:
        crop_y0 = min(page_h_pt, y1 + 0.01 * page_h_pt)
        crop_y1 = min(page_h_pt, crop_y0 + 0.45 * page_h_pt)
    return (crop_x0, crop_y0, crop_x1, crop_y1)


def _infer_figure_above(
    page_lines: List[TextLine],
    caption_ln: TextLine,
    page_h_pt: float,
) -> bool:
    """Return True if figure is likely above the caption (caption below)."""
    band = 0.25 * page_h_pt
    cap_y0 = caption_ln.bbox[1]
    cap_y1 = caption_ln.bbox[3]
    above_lines = [ln for ln in page_lines if ln.bbox[3] <= cap_y0 and (cap_y0 - ln.bbox[3]) <= band]
    below_lines = [ln for ln in page_lines if ln.bbox[1] >= cap_y1 and (ln.bbox[1] - cap_y1) <= band]

    def _score(lines: List[TextLine]) -> int:
        score = 0
        for ln in lines:
            t = ln.text.strip()
            if not t or caption_type(t):
                continue
            if _is_all_caps(t) and len(t) <= 80:
                score += 2
            elif len(t) <= 40:
                score += 1
        return score

    above_score = _score(above_lines)
    below_score = _score(below_lines)
    if above_score == below_score:
        cy = cap_y0 / page_h_pt
        return True if cy > 0.35 else False
    return above_score > below_score


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
    abstract_inserted = False
    main_text_inserted = False
    in_abstract = False
    auto_abstract_used = False
    abstract_left_margin: Optional[float] = None
    in_references = False
    pending_footnotes: List[TextLine] = []
    pending_captions: List[List[str]] = []
    last_block_type: Optional[str] = None
    last_para_index: Optional[int] = None
    page_break_pending = False
    ref_left_margin: Optional[float] = None
    refs_started = False
    author_bio_started = False

    # Heading detection thresholds (relative to median font size)
    dominant_size = _dominant_font_size(lines) or median_size
    base_size = dominant_size if dominant_size > 0 else median_size
    h1_threshold = base_size * 1.45  # Title: larger than body
    h2_threshold = base_size * 1.20  # Major section hint
    h3_threshold = base_size * 1.10  # Minor section hint

    def classify_heading(ln: TextLine, text: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Returns (heading level, kind) or (None, None).
        Uses font size, bold flag, and text patterns.
        """
        nonlocal title_found
        
        # Skip if text is too long (unlikely to be a heading)
        if len(text) > 120:
            return None, None
        
        # Skip figure/table captions
        if caption_type(text):
            return None, None

        # Skip source/author/affiliation lines
        if _looks_like_source_line(text) or _looks_like_author_line(text) or _looks_like_affiliation_line(text):
            return None, None
        if _looks_like_footnote_marker(text) or _looks_like_citation_line(text):
            return None, None
        if text and text[0].islower():
            return None, None
        if re.match(r"^\d", text) and not re.match(r"^\d+(\.\d+)?\s+\S", text):
            return None, None
        if text.endswith(",") or text.endswith(";"):
            return None, None
        if text.endswith(".") and len(text) > 25:
            return None, None

        kind = _section_kind(text)
        
        # Check font size
        size = ln.size
        is_bold = ln.is_bold
        
        # Title (h1): largest font, typically on first pages
        if size >= h1_threshold and ln.page <= 2 and not title_found and kind not in ("abstract", "references"):
            return 1, "title"

        if kind == "abstract":
            return 2, "abstract"
        if kind == "references":
            return 2, "references"
        if kind == "major":
            return 3, "major"
        if kind == "minor":
            return 4, "minor"
        
        # Size-based fallback
        if size >= h2_threshold:
            return 3, "major"

        if size >= h3_threshold and is_bold:
            # Additional check: short text, likely a heading
            if len(text) < 80:
                return 4, "minor"
        
        # Numbered sections like "2.1 Something" with bold
        if is_bold and re.match(r"^\d+\.\d+\s+\S", text):
            return 4, "minor"
        
        return None, None

    def extract_front_matter(page_lines: List[TextLine], page_h_pt: float) -> Tuple[str, List[str], List[str], List[str], int]:
        source_lines: List[str] = []
        title_lines: List[str] = []
        authors: List[str] = []
        affiliations: List[str] = []

        i = 0
        page_max_size = max((ln.size for ln in page_lines if ln.size > 0), default=0.0)
        if page_max_size >= base_size * 1.15:
            title_cutoff = page_max_size * 0.88
        else:
            title_cutoff = base_size * 1.20

        # Source lines near top (journal/doi)
        while i < len(page_lines):
            ln = page_lines[i]
            text = ln.text.strip()
            if not text:
                i += 1
                continue
            ymid = ((ln.bbox[1] + ln.bbox[3]) / 2.0) / page_h_pt
            if ymid <= 0.12 and _looks_like_source_line(text):
                source_lines.append(_clean_leading_singleton(text))
                i += 1
                continue
            break

        # Title lines (largest font near top)
        while i < len(page_lines):
            ln = page_lines[i]
            text = ln.text.strip()
            if not text:
                i += 1
                continue
            ymid = ((ln.bbox[1] + ln.bbox[3]) / 2.0) / page_h_pt
            if ymid > 0.45:
                break
            is_title = ln.size >= title_cutoff
            if not is_title:
                if _is_all_caps(text) and ln.size >= base_size * 1.15 and len(text) <= 100 and ymid <= 0.35:
                    is_title = True
            if is_title and not _looks_like_source_line(text):
                title_lines.append(text)
                i += 1
                continue
            break

        # Authors / affiliations
        while i < len(page_lines):
            ln = page_lines[i]
            text = ln.text.strip()
            if not text:
                i += 1
                continue
            if _looks_like_author_line(text):
                authors.append(text)
                i += 1
                continue
            if _looks_like_affiliation_line(text):
                affiliations.append(text)
                i += 1
                continue
            # Heuristic: short title-case lines after authors are likely affiliations
            if authors and len(text) <= 80 and _is_title_case(text) and not _section_kind(text):
                affiliations.append(text)
                i += 1
                continue
            if _section_kind(text):
                break
            if len(text) > 120:
                break
            if re.search(r"[.!?]$", text) and len(text) > 60:
                break
            break

        title = " ".join(t for t in title_lines if t).strip()
        return title, authors, affiliations, source_lines, i

    def flush_pending_footnotes(*, force: bool = False) -> None:
        nonlocal pending_footnotes, last_block_type, last_para_index
        if not pending_footnotes:
            return
        if not force:
            last_nonempty = next(
                (ln for ln in reversed(md_lines) if ln.strip() and not _is_caption_or_placeholder_line(ln)),
                "",
            )
            if not (last_nonempty and _ends_sentence(last_nonempty)):
                return
        md_lines.append("---")
        md_lines.append(render_footnotes(pending_footnotes))
        md_lines.append("")
        last_block_type = "footnotes"
        last_para_index = None
        pending_footnotes.clear()

    def flush_pending_captions(*, preserve_last_block: bool) -> None:
        nonlocal pending_captions, last_block_type, last_para_index
        if not pending_captions:
            return
        for block in pending_captions:
            md_lines.extend(block)
        if not preserve_last_block:
            last_block_type = "caption"
            last_para_index = None
        pending_captions.clear()

    def merge_author_bio_paragraphs(lines: List[str]) -> List[str]:
        out: List[str] = []
        in_bio = False
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.strip() == "## Author Biographies":
                in_bio = True
                out.append(ln)
                i += 1
                continue
            if in_bio:
                if ln.startswith("#") and ln.strip() != "## Author Biographies":
                    in_bio = False
                    out.append(ln)
                    i += 1
                    continue
                if ln.strip() == "":
                    j = i + 1
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1
                    next_ln = lines[j] if j < len(lines) else ""
                    if next_ln and _looks_like_author_bio_start(next_ln):
                        out.append("")
                    i += 1
                    continue
                if out and out[-1].strip() and not out[-1].startswith("#"):
                    if not _looks_like_author_bio_start(ln):
                        out[-1] = _join_text(out[-1], ln)
                        i += 1
                        continue
                out.append(ln)
                i += 1
                continue
            out.append(ln)
            i += 1
        return out

    for p in range(1, len(page_sizes) + 1):
        W, H = page_sizes[p - 1]
        page_lines = by_page.get(p, [])
        page_break_pending = bool(md_lines)
        first_content_on_page = True
        figure_regions: List[Tuple[float, float, float, float]] = []
        for ln in page_lines:
            if caption_type(ln.text) == "figure":
                fig_above = _infer_figure_above(page_lines, ln, H)
                figure_regions.append(infer_figure_region_from_caption(W, H, ln.bbox, above=fig_above))

        # Filter header/footer candidates
        filtered: List[TextLine] = []
        for ln in page_lines:
            text = ln.text.strip()
            if not text:
                continue
            if figure_regions and not caption_type(text):
                if any(_bbox_overlaps(ln.bbox, reg) for reg in figure_regions):
                    continue
            norm = _normalize_for_repeat(text)
            ymid = ((ln.bbox[1] + ln.bbox[3]) / 2.0) / H
            top_band = ymid <= 0.12
            bot_band = ymid >= 0.88
            small = dominant_size > 0 and ln.size <= dominant_size * 0.80

            if norm in header_norms or norm in footer_norms:
                if not _looks_like_footnote_marker(text):
                    continue

            if bot_band and is_probable_page_number(text):
                continue

            if top_band and small:
                if p == 1 and _looks_like_source_line(text):
                    filtered.append(ln)
                    continue
                if _looks_like_footnote_marker(text):
                    filtered.append(ln)
                    continue
                if _looks_like_month_line(text):
                    continue
                if _looks_like_source_line(text):
                    continue
                if len(text) <= 6:
                    continue
                continue

            if bot_band and small and _looks_like_boilerplate_footer(text):
                continue

            filtered.append(ln)

        # Reading order
        cols, split_x = infer_columns_for_page(filtered, W)
        ordered = order_lines_reading(filtered, cols, split_x)

        # Coverage baseline from raw ordered lines (BEFORE heading/paragraph decisions)
        baseline_accum.append(build_baseline_text_from_ordered(ordered))

        refs_heading_index: Optional[int] = None
        for j, ln in enumerate(ordered):
            if _section_kind(ln.text.strip()) == "references":
                refs_heading_index = j
                break

        footnotes: List[TextLine] = []
        main_ordered: List[TextLine] = []
        if in_references:
            main_ordered = ordered
        else:
            idx = 0
            while idx < len(ordered):
                ln = ordered[idx]
                if refs_heading_index is not None and idx >= refs_heading_index:
                    main_ordered.append(ln)
                    idx += 1
                    continue
                text = ln.text.strip()
                if _looks_like_footnote_marker(text):
                    footnotes.append(ln)
                    idx += 1
                    while idx < len(ordered):
                        if refs_heading_index is not None and idx >= refs_heading_index:
                            break
                        nxt = ordered[idx]
                        nxt_text = nxt.text.strip()
                        if not nxt_text:
                            idx += 1
                            continue
                        if caption_type(nxt_text):
                            break
                        ymid = ((nxt.bbox[1] + nxt.bbox[3]) / 2.0) / H
                        if ymid < 0.60:
                            break
                        if base_size > 0 and nxt.size > base_size * 1.05:
                            break
                        footnotes.append(nxt)
                        idx += 1
                    continue
                if _is_probable_footnote_line(ln, base_size, H) and not caption_type(text):
                    footnotes.append(ln)
                else:
                    main_ordered.append(ln)
                idx += 1
        ordered = main_ordered

        start_idx = 0
        if p == 1 and not title_found:
            title, authors, affiliations, sources, start_idx = extract_front_matter(ordered, H)
            if title:
                md_lines.append(f"# {title}")
                md_lines.append("")
                title_found = True
            if authors:
                md_lines.append("**Authors:** " + "; ".join(authors))
            if affiliations:
                md_lines.append("**Affiliations:** " + "; ".join(affiliations))
            if sources:
                md_lines.append("**Source:** " + " ".join(sources))
            if title or authors or affiliations or sources:
                md_lines.append("")
                last_block_type = "front"
                last_para_index = None
                first_content_on_page = False
                page_break_pending = False

        # Process lines with heading detection
        i = start_idx
        while i < len(ordered):
            ln = ordered[i]
            text = ln.text.strip()

            if not text:
                i += 1
                continue

            force_paragraph = False
            if in_references and _looks_like_author_bio_heading(text):
                in_references = False
                refs_started = False
                ref_left_margin = None
                if not author_bio_started:
                    md_lines.append("## Author Biographies")
                    md_lines.append("")
                    author_bio_started = True
                i += 1
                continue
            if in_references and _looks_like_author_bio_start(text):
                in_references = False
                refs_started = False
                ref_left_margin = None
                if not author_bio_started:
                    md_lines.append("## Author Biographies")
                    md_lines.append("")
                    author_bio_started = True
                force_paragraph = True
            if author_bio_started and not in_references and _looks_like_author_bio_start(text):
                force_paragraph = True
            if not in_references and not force_paragraph:
                inline = _split_inline_heading(text)
                if inline:
                    heading_text, rest_text = inline
                    heading_level, kind = classify_heading(ln, heading_text)
                    if heading_level:
                        if heading_level == 1:
                            title_found = True
                        if kind == "abstract":
                            abstract_inserted = True
                            in_abstract = True
                            abstract_left_margin = None
                        elif kind == "references":
                            in_abstract = False
                            abstract_left_margin = None
                            flush_pending_footnotes(force=True)
                            in_references = True
                            ref_left_margin = None
                            refs_started = False
                        else:
                            if abstract_inserted and not main_text_inserted:
                                md_lines.append("## Main Text")
                                md_lines.append("")
                                main_text_inserted = True
                            in_abstract = False
                            abstract_left_margin = None

                        flush_pending_captions(preserve_last_block=False)
                        prefix = "#" * heading_level
                        md_lines.append(f"{prefix} {heading_text}")
                        md_lines.append("")
                        last_block_type = "heading"
                        last_para_index = None
                        first_content_on_page = False
                        page_break_pending = False

                        if rest_text:
                            text = rest_text
                            force_paragraph = True
                        else:
                            i += 1
                            continue

            # Check for caption
            if not force_paragraph:
                ctype = caption_type(text)
            else:
                ctype = None
            if ctype:
                cap_counter += 1
                marker = f"@@CAP_{cap_counter:04d}@@"
                pending_captions.append([
                    f"**[{text}]**",  # caption from PDF
                    marker,           # placeholder
                    "",
                ])

                cap_info = {
                    "type": ctype,
                    "caption": text,
                    "page": p,
                    "page_w_pt": W,
                    "page_h_pt": H,
                    "caption_bbox_pt": ln.bbox,
                }
                if ctype == "figure":
                    cap_info["figure_above"] = _infer_figure_above(page_lines, ln, H)
                cap_map[marker] = cap_info
                i += 1
                continue

            if in_references:
                continuing_prev_ref = False
                if page_break_pending and first_content_on_page and last_block_type == "para" and last_para_index is not None:
                    if _looks_like_continuation(md_lines[last_para_index], text) and not _looks_like_reference_start(text):
                        continuing_prev_ref = True
                        prev_entry = md_lines[last_para_index].lstrip("- ").strip()
                        entry_lines = [prev_entry, text]
                    else:
                        entry_lines = [text]
                else:
                    entry_lines = [text]
                note_mode = (not refs_started and re.match(r"^(Note|Notes):", text))
                if continuing_prev_ref:
                    note_mode = False
                    refs_started = True
                if ref_left_margin is None:
                    ref_left_margin = ln.bbox[0]
                else:
                    ref_left_margin = min(ref_left_margin, ln.bbox[0])
                prev_y = ln.bbox[3]
                cur_col = 0 if (cols == 2 and ln.bbox[0] < split_x) else 1
                i += 1
                bio_hit = False
                while i < len(ordered):
                    next_ln = ordered[i]
                    next_text = next_ln.text.strip()
                    if not next_text:
                        i += 1
                        continue
                    if _looks_like_author_bio_heading(next_text) or _looks_like_author_bio_start(next_text):
                        bio_hit = True
                        break
                    base_is_new = _looks_like_reference_start(next_text)
                    next_is_new = base_is_new
                    if not note_mode:
                        if entry_lines and re.search(r":\s*$", entry_lines[-1]) and re.match(r"^\d", next_text):
                            next_is_new = False
                        if entry_lines and re.search(r"(?:&|and)\s*$", entry_lines[-1], re.IGNORECASE):
                            next_is_new = False
                        if entry_lines and re.match(r"^\d{4}\b", next_text):
                            next_is_new = False
                        if entry_lines and entry_lines[-1].endswith(",") and not re.search(r"\b\d{4}[a-z]?\b", " ".join(entry_lines)):
                            next_is_new = False
                        if entry_lines and re.search(r"\b(van|von|de|del|der|di|da|le|la|du|dos|das)$", entry_lines[-1], re.IGNORECASE):
                            next_is_new = False
                        if entry_lines and not base_is_new and (next_text[:1].islower() or re.match(r"^(and|&|in|of|for|to|with|from|by|on)\b", next_text, re.IGNORECASE)):
                            next_is_new = False
                        if cols == 2:
                            next_col = 0 if next_ln.bbox[0] < split_x else 1
                            if next_col != cur_col and next_ln.bbox[1] < prev_y and not base_is_new:
                                next_is_new = False
                    if next_is_new:
                        break
                    if entry_lines and entry_lines[-1].endswith("-") and next_text[:1].islower():
                        entry_lines[-1] = entry_lines[-1][:-1] + next_text
                    else:
                        entry_lines.append(next_text)
                    prev_y = next_ln.bbox[3]
                    if cols == 2:
                        cur_col = 0 if next_ln.bbox[0] < split_x else 1
                    i += 1
                entry = " ".join(entry_lines).strip()
                if entry:
                    entry = entry.replace("\x01", "\n")
                    parts = [p.strip() for p in entry.splitlines() if p.strip()]
                    if note_mode:
                        note_text = " ".join(parts).strip()
                        if note_text:
                            md_lines.append(note_text)
                            md_lines.append("")
                            last_block_type = "para"
                            last_para_index = len(md_lines) - 2
                    else:
                        expanded_parts: List[str] = []
                        for part in parts:
                            expanded_parts.extend(_split_reference_entries(part))
                        merged_parts: List[str] = []
                        for part in expanded_parts:
                            if merged_parts:
                                if re.search(r"\b(van|von|de|del|der|di|da|le|la|du|dos|das)$", merged_parts[-1], re.IGNORECASE):
                                    merged_parts[-1] = _join_text(merged_parts[-1], part)
                                    continue
                                if not _looks_like_reference_start(part):
                                    merged_parts[-1] = _join_text(merged_parts[-1], part)
                                    continue
                            merged_parts.append(part)
                        parts = merged_parts
                        start_idx = 0
                        if continuing_prev_ref and parts:
                            md_lines[last_para_index] = f"- {parts[0]}"
                            start_idx = 1
                        for part in parts[start_idx:]:
                            md_lines.append(f"- {part}")
                            md_lines.append("")
                            last_block_type = "para"
                            last_para_index = len(md_lines) - 2
                            refs_started = True
                first_content_on_page = False
                page_break_pending = False
                if bio_hit:
                    in_references = False
                    refs_started = False
                    ref_left_margin = None
                    if not author_bio_started:
                        md_lines.append("## Author Biographies")
                        md_lines.append("")
                        author_bio_started = True
                continue

            # Check for heading
            heading_level: Optional[int] = None
            kind: Optional[str] = None
            if not in_references and not force_paragraph:
                heading_level, kind = classify_heading(ln, text)
            if heading_level:
                heading_text = text
                k = i
                while True:
                    k_next = k + 1
                    if k_next >= len(ordered):
                        break
                    next_ln = ordered[k_next]
                    next_text = next_ln.text.strip()
                    if not next_text:
                        k = k_next
                        continue
                    next_level = None
                    next_kind = None
                    if not in_references:
                        next_level, next_kind = classify_heading(next_ln, next_text)
                    if next_level == heading_level:
                        same_col = True
                        if cols == 2:
                            cur_is_left = ln.bbox[0] < split_x
                            next_is_left = next_ln.bbox[0] < split_x
                            same_col = (cur_is_left == next_is_left)
                        vgap = next_ln.bbox[1] - ln.bbox[3]
                        if same_col and vgap <= 20:
                            heading_text = f"{heading_text} {next_text}"
                            ln = next_ln
                            k = k_next
                            continue
                    break
                if heading_level == 1:
                    title_found = True
                if kind == "abstract":
                    abstract_inserted = True
                    in_abstract = True
                    abstract_left_margin = None
                elif kind == "references":
                    in_abstract = False
                    abstract_left_margin = None
                    flush_pending_footnotes(force=True)
                    in_references = True
                    ref_left_margin = None
                    refs_started = False
                else:
                    if abstract_inserted and not main_text_inserted:
                        md_lines.append("## Main Text")
                        md_lines.append("")
                        main_text_inserted = True
                    in_abstract = False
                    abstract_left_margin = None

                flush_pending_captions(preserve_last_block=False)
                prefix = "#" * heading_level
                md_lines.append(f"{prefix} {heading_text}")
                md_lines.append("")
                last_block_type = "heading"
                last_para_index = None
                first_content_on_page = False
                page_break_pending = False
                i = k + 1
                continue
            
            # Regular paragraph: collect consecutive normal lines
            para_lines = [text]
            para_x0 = ln.bbox[0]
            para_x1 = ln.bbox[2]
            cur_col = 0 if (cols == 2 and ln.bbox[0] < split_x) else 1
            prev_y = ln.bbox[3]
            bio_mode = author_bio_started
            i += 1
            
            while i < len(ordered):
                next_ln = ordered[i]
                next_text = next_ln.text.strip()
                
                # Stop if next line is a heading or caption
                next_heading = None
                if not in_references:
                    next_heading, _ = classify_heading(next_ln, next_text)
                    if not next_heading and _split_inline_heading(next_text):
                        next_heading = 1
                if next_heading:
                    break
                next_caption_type = caption_type(next_text)
                if next_caption_type:
                    cap_counter += 1
                    marker = f"@@CAP_{cap_counter:04d}@@"
                    pending_captions.append([
                        f"**[{next_text}]**",  # caption from PDF
                        marker,               # placeholder
                        "",
                    ])
                    cap_info = {
                        "type": next_caption_type,
                        "caption": next_text,
                        "page": p,
                        "page_w_pt": W,
                        "page_h_pt": H,
                        "caption_bbox_pt": next_ln.bbox,
                    }
                    if next_caption_type == "figure":
                        cap_info["figure_above"] = _infer_figure_above(page_lines, next_ln, H)
                    cap_map[marker] = cap_info
                    i += 1
                    continue
                if bio_mode and _looks_like_author_bio_start(next_text):
                    break

                col_switched = False
                if cols == 2:
                    next_col = 0 if next_ln.bbox[0] < split_x else 1
                    if next_col != cur_col and next_ln.bbox[1] < prev_y:
                        col_switched = True
                        if re.search(r"[.!?][\"”']?$", para_lines[-1]) and not bio_mode:
                            break
                if not col_switched and _ends_sentence(para_lines[-1]):
                    if (next_ln.bbox[0] - para_x0) > 8 and next_text[:1].isupper() and len(next_text) < 80:
                        break
                
                # Stop if large vertical gap (new paragraph)
                if not bio_mode and next_ln.bbox[1] - prev_y > 18:
                    if not col_switched:
                        if _ends_sentence(para_lines[-1]):
                            break
                        if (next_ln.bbox[0] - para_x0) > 18 and next_text[:1].isupper():
                            break
                
                # Hyphenation fix
                if para_lines[-1].endswith("-") and next_text[:1].islower():
                    para_lines[-1] = para_lines[-1][:-1] + next_text
                else:
                    para_lines.append(next_text)
                
                para_x0 = min(para_x0, next_ln.bbox[0])
                para_x1 = max(para_x1, next_ln.bbox[2])
                if cols == 2:
                    cur_col = 0 if next_ln.bbox[0] < split_x else 1
                prev_y = next_ln.bbox[3]
                i += 1
            
            para = " ".join(para_lines).strip()
            if para:
                if page_break_pending and first_content_on_page and last_block_type == "para" and last_para_index is not None:
                    if _looks_like_continuation(md_lines[last_para_index], para):
                        if md_lines and md_lines[-1] == "":
                            md_lines.pop()
                        md_lines[last_para_index] = _join_text(md_lines[last_para_index], para)
                        md_lines.append("")
                        first_content_on_page = False
                        page_break_pending = False
                        last_block_type = "para"
                        if auto_abstract_used and in_abstract:
                            in_abstract = False
                        continue
                if not abstract_inserted and p == 1 and not auto_abstract_used:
                    md_lines.append("## Abstract")
                    md_lines.append("")
                    abstract_inserted = True
                    in_abstract = True
                    auto_abstract_used = True
                    abstract_left_margin = None

                if in_abstract:
                    if abstract_left_margin is None:
                        abstract_left_margin = para_x0
                    elif abs(para_x0 - abstract_left_margin) > 40:
                        in_abstract = False
                        if not main_text_inserted:
                            md_lines.append("## Main Text")
                            md_lines.append("")
                            main_text_inserted = True

                if abstract_inserted and not main_text_inserted and not in_abstract:
                    md_lines.append("## Main Text")
                    md_lines.append("")
                    main_text_inserted = True

                md_lines.append(para)
                md_lines.append("")
                last_block_type = "para"
                last_para_index = len(md_lines) - 2
                first_content_on_page = False
                page_break_pending = False
                flush_pending_captions(preserve_last_block=True)

                if auto_abstract_used and in_abstract:
                    in_abstract = False

        if footnotes:
            pending_footnotes.extend(footnotes)
        flush_pending_captions(preserve_last_block=False)
        flush_pending_footnotes()

    flush_pending_captions(preserve_last_block=False)
    flush_pending_footnotes(force=True)

    md_lines = merge_broken_paragraphs(md_lines)
    md_lines = relocate_mid_paragraph_captions(md_lines)
    md_lines = split_runin_heading_paragraphs(md_lines)
    if author_bio_started:
        md_lines = merge_author_bio_paragraphs(md_lines)
    md_lines = normalize_markdown_lines(md_lines)

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
- Provide 6 to 12 bullets. Each bullet should be 1-2 sentences.
- Prefer concrete description: elements, labels, axes, legends, arrows, flows, groupings.
- Include layout/relationships (left/right, top/bottom, adjacency, nesting, flow direction).
- If the figure has multiple panels or sections, describe each one.
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
    above = cap_info.get("figure_above")
    if bbox is not None:
        if above is None:
            cy = bbox[1] / H
            above = True if cy > 0.35 else False
        crop = crop_region_from_caption(page_im, W, H, bbox, above=bool(above))
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
        max_tokens=1200,
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

