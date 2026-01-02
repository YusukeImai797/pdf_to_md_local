import os
import io
import base64
import re
import gc
import json
import time
import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from tqdm import tqdm

from lmstudio_client import LMStudioClient


def _pil_to_b64_jpeg(pil_image, *, max_width: int, quality: int) -> str:
    """Resize (if needed) and encode PIL image as base64 JPEG.

    This is intentionally lossy: the goal is to reduce peak VRAM / request size
    while keeping enough fidelity for layout/structure extraction.
    """
    w, h = pil_image.size
    if w > max_width:
        new_h = max(1, int(h * (max_width / w)))
        pil_image = pil_image.resize((max_width, new_h))
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a string and parse it.
    
    Prioritizes:
    1. ```json ... ``` markdown blocks
    2. Balanced braces extraction (first { to matching })
    """
    if not text:
        return None
    
    # Remove <think>...</think> if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Priority 1: Look for ```json ... ``` markdown block
    json_block_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1))
        except Exception:
            pass  # Fall through to balanced braces approach
    
    # Priority 2: Balanced braces extraction
    # Find the first '{' and count braces to find the matching '}'
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    depth = 0
    end_idx = -1
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    
    if end_idx == -1:
        return None
    
    blob = text[start_idx:end_idx + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None


def _caption_type_from_text(s: str) -> str:
    """Return 'table', 'figure', or 'unknown' based on caption keywords (language-aware)."""
    if not s:
        return "unknown"
    t = s.lower()
    # Table keywords
    if re.search(r"\btable\b|\btab\.?\b|表", t):
        return "table"
    # Figure keywords
    if re.search(r"\bfigure\b|\bfig\.?\b|図", t):
        return "figure"
    return "unknown"


class RunRecorder:
    """Durable run telemetry + checkpoints.

    NOTE: When the OS power-cycles, the process cannot flush logs "at that moment".
    This class writes continuously during execution (fsync) and also supports
    collecting Windows postmortem logs on the next run.
    """

    def __init__(self, out_root: Path, run_name: str):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = out_root / "_diag" / f"{run_name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.run_dir / "run_meta.json"
        self.checkpoint_path = self.run_dir / "checkpoint.jsonl"
        self.status_path = self.run_dir / "status.json"
        self.gpu_csv_path = self.run_dir / "gpu_telemetry.csv"

        self._nvsmi_proc: Optional[subprocess.Popen] = None

        self._write_status({"state": "running", "updated_at": datetime.datetime.now().isoformat()})

    def _write_status(self, status: Dict[str, Any]):
        # Atomic write + fsync so that power loss still leaves a durable state
        tmp = self.status_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps(status, ensure_ascii=False, indent=2))
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        tmp.replace(self.status_path)

    def mark_completed(self):
        self._write_status({"state": "completed", "updated_at": datetime.datetime.now().isoformat()})

    def mark_failed(self, error: str):
        self._write_status({
            "state": "error",
            "updated_at": datetime.datetime.now().isoformat(),
            "error": error[:2000],
        })

    def write_meta(self, meta: Dict[str, Any]):
        meta = dict(meta)
        meta["run_dir"] = str(self.run_dir)
        meta["started_at"] = datetime.datetime.now().isoformat()
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def checkpoint(self, stage: str, payload: Dict[str, Any]):
        rec = {
            "ts": datetime.datetime.now().isoformat(),
            "stage": stage,
            "payload": payload,
        }
        with self.checkpoint_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def start_nvidia_smi_logger(self, interval_sec: int = 1):
        # Use nvidia-smi built-in file logger (-f) for durability.
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory",
            "--format=csv",
            "-l", str(interval_sec),
            "-f", str(self.gpu_csv_path),
        ]
        try:
            self._nvsmi_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.checkpoint("telemetry", {"status": "started", "path": str(self.gpu_csv_path)})
        except FileNotFoundError:
            self.checkpoint("telemetry", {"status": "nvidia-smi not found"})
            self._nvsmi_proc = None
        except Exception as e:
            self.checkpoint("telemetry", {"status": "failed", "error": str(e)})
            self._nvsmi_proc = None

    def stop_nvidia_smi_logger(self):
        if self._nvsmi_proc and self._nvsmi_proc.poll() is None:
            try:
                self._nvsmi_proc.terminate()
                try:
                    self._nvsmi_proc.wait(timeout=2)
                except Exception:
                    self._nvsmi_proc.kill()
            except Exception:
                pass


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


class LocalConverter:
    def __init__(self, input_folder: str, output_folder: str, translation_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.translation_folder = translation_folder

        # Ensure folders exist
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(translation_folder, exist_ok=True)

        self.client = LMStudioClient()

        # Model IDs
        self.vision_model = "qwen/qwen2.5-vl-7b"
        self.text_model = "qwen/qwen3-30b-a3b"

        # Runtime knobs (tune to reduce peak load while preserving quality)
        self.pass1_pages_per_request = 10
        self.pass2_pages_per_request = 6

        self.pass1_dpi = 90
        self.pass2_dpi = 140
        self.table_refine_dpi = 220  # used only for specific pages if needed

        self.pass1_max_width = 1024
        self.pass2_max_width = 1600

        self.pass1_jpeg_quality = 60
        self.pass2_jpeg_quality = 75

        # Token caps (prevents runaway generation)
        self.pass1_max_tokens = 1200
        self.pass2_max_tokens = 4096
        self.audit_max_tokens = 1200

        # Safety: translation is optional (reduces model swapping load)
        self.enable_translation = False  # Recommend enabling only after conversion stability is confirmed

        # Diagnostics / durability
        self.diag_root = Path(self.output_folder)  # keep alongside outputs
        self._ps_diag_script = self.diag_root / "_diag" / "collect_shutdown_diag.ps1"
        self._ps_diag_script.parent.mkdir(parents=True, exist_ok=True)
        if not self._ps_diag_script.exists():
            _write_ps_collect_script(self._ps_diag_script)

    def convert_all(self):
        pdf_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files.")
        
        if not self.client.check_connection():
            print("Error: Could not connect to LMStudio. Is local server running at http://localhost:1234?")
            return

        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            self.process_pdf(pdf_file)

    def process_pdf(self, filename: str):
        input_path = os.path.join(self.input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(self.output_folder, base_name + ".md")

        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists in output)")
            return

        recorder = RunRecorder(Path(self.output_folder), run_name=base_name)
        recorder.write_meta({
            "filename": filename,
            "vision_model": self.vision_model,
            "text_model": self.text_model,
            "pass1_pages_per_request": self.pass1_pages_per_request,
            "pass2_pages_per_request": self.pass2_pages_per_request,
            "pass1_dpi": self.pass1_dpi,
            "pass2_dpi": self.pass2_dpi,
        })

        # Start continuous GPU telemetry (best effort)
        recorder.start_nvidia_smi_logger(interval_sec=1)

        try:
            # If the previous run ended abruptly, collect Windows postmortem logs now.
            self._collect_postmortem_if_needed(recorder)

            recorder.checkpoint("start", {"pdf": input_path})

            # 1) Pass1: outline + figure/table inventory (JSON)
            print(f"[Pass 1] Building outline and inventory for: {filename}")
            index = self._pass1_build_index(input_path, recorder=recorder)
            index_path = recorder.run_dir / "pass1_index.json"
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
            recorder.checkpoint("pass1_done", {"index_path": str(index_path)})

            # 2) Pass2: chunked multi-page conversion to Markdown (durable parts)
            print(f"[Pass 2] Converting pages to Markdown...")
            parts_dir = recorder.run_dir / "parts"
            parts_dir.mkdir(parents=True, exist_ok=True)
            draft_md = self._pass2_convert_chunks(
                input_path,
                index=index,
                parts_dir=parts_dir,
                recorder=recorder,
            )
            draft_path = recorder.run_dir / "pass2_draft.md"
            draft_path.write_text(draft_md, encoding="utf-8")
            recorder.checkpoint("pass2_done", {"draft_path": str(draft_path)})

            # 3) Pass3: deterministic cleanup + audit (no rewriting)
            print(f"[Pass 3] Finalizing and auditing...")
            final_md, audit = self._pass3_finalize(draft_md, index=index, recorder=recorder)
            audit_path = recorder.run_dir / "pass3_audit.json"
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

            # Save final Markdown
            Path(output_path).write_text(final_md, encoding="utf-8")
            print(f"Saved Markdown: {output_path}")
            recorder.checkpoint("saved_output", {"output_path": output_path})

            # Optional translation
            if self.enable_translation:
                translation_filename = base_name + "_jp.md"
                translation_path = os.path.join(self.translation_folder, translation_filename)
                print(f"Starting Full Translation for: {filename}...")
                translated = self._translate_full_text(final_md)
                Path(translation_path).write_text(translated, encoding="utf-8")
                print(f"Saved Full Translation: {translation_path}")
                recorder.checkpoint("translation_done", {"translation_path": translation_path})

            recorder.mark_completed()
            print(f"[Done] {filename}")

        except Exception as e:
            recorder.checkpoint("error", {"error": str(e)})
            recorder.mark_failed(str(e))
            print(f"Failed to process {filename}: {e}")

        finally:
            recorder.stop_nvidia_smi_logger()

    # ----------------------------
    # Pass 1: outline + inventory
    # ----------------------------
    def _pass1_build_index(self, pdf_path: str, recorder: RunRecorder) -> Dict[str, Any]:
        doc = fitz.open(pdf_path)
        n_pages = doc.page_count
        doc.close()

        merged: Dict[str, Any] = {
            "title": "",
            "outline": [],
            "objects": [],
            "header_footer_candidates": [],
            "page_count": n_pages,
        }

        for start in range(1, n_pages + 1, self.pass1_pages_per_request):
            end = min(n_pages, start + self.pass1_pages_per_request - 1)
            recorder.checkpoint("pass1_chunk_start", {"start": start, "end": end})
            print(f"  Pass1: pages {start}-{end}...")

            images = convert_from_path(pdf_path, dpi=self.pass1_dpi, first_page=start, last_page=end)
            images_b64 = []
            for im in images:
                images_b64.append(_pil_to_b64_jpeg(im, max_width=self.pass1_max_width, quality=self.pass1_jpeg_quality))

            prompt = f"""You are analyzing page images of a PDF.

Output ONLY valid JSON (no markdown, no prose, no <think>).

Goal:
1) Build an outline of headings with page numbers.
2) Build an inventory of visual objects: figures and tables.

Rules:
- Do NOT transcribe the whole body text.
- For each visual object, decide type in {{"figure","table","unknown"}}.
- Evidence MUST be words visible on the page (e.g., "Figure", "Fig.", "図", "Table", "表", "TABLE").
- If you cannot confidently decide, use "unknown".
- Also list short repeated strings that look like headers/footers (exact text only).

JSON schema:
{{
  "title": "...",
  "outline": [{{"level": 1, "heading": "...", "page": {start}}}, ...],
  "objects": [
    {{"page": {start}, "type": "table", "id": "Table 2", "caption": "...", "evidence": ["Table 2", "表2"]}},
    ...
  ],
  "header_footer_candidates": ["...", ...]
}}

You are given pages {start}..{end}.
""".strip()

            raw = self.client.vision_request_multi(
                prompt,
                images_b64,
                model=self.vision_model,
                max_tokens=self.pass1_max_tokens,
                timeout=(20, 900),
            ) or ""

            data = _safe_json_extract(raw) or {}

            # Merge
            if not merged["title"] and isinstance(data.get("title"), str):
                merged["title"] = data.get("title", "")

            if isinstance(data.get("outline"), list):
                merged["outline"].extend(data["outline"])

            if isinstance(data.get("objects"), list):
                merged["objects"].extend(data["objects"])

            if isinstance(data.get("header_footer_candidates"), list):
                merged["header_footer_candidates"].extend(data["header_footer_candidates"])

            # memory cleanup
            del images, images_b64, raw, data
            gc.collect()

            recorder.checkpoint("pass1_chunk_done", {"start": start, "end": end})

        # Deduplicate header/footer candidates
        hf = []
        seen = set()
        for s in merged["header_footer_candidates"]:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            hf.append(t)
        merged["header_footer_candidates"] = hf

        # Normalize object types using caption keywords (code-enforced)
        norm_objects = []
        for obj in merged["objects"]:
            if not isinstance(obj, dict):
                continue
            caption = str(obj.get("caption", "") or "")
            obj_type = str(obj.get("type", "unknown") or "unknown").lower()
            forced = _caption_type_from_text((obj.get("id", "") or "") + " " + caption)
            if forced in ("figure", "table"):
                obj_type = forced
            if obj_type not in ("figure", "table", "unknown"):
                obj_type = "unknown"
            obj["type"] = obj_type
            norm_objects.append(obj)
        merged["objects"] = norm_objects

        return merged

    # ----------------------------
    # Pass 2: chunk conversion
    # ----------------------------
    def _pass2_convert_chunks(
        self,
        pdf_path: str,
        index: Dict[str, Any],
        parts_dir: Path,
        recorder: RunRecorder,
    ) -> str:
        n_pages = int(index.get("page_count") or 0)
        if not n_pages:
            doc = fitz.open(pdf_path)
            n_pages = doc.page_count
            doc.close()

        # Helper to get objects for chunk
        objects = index.get("objects") if isinstance(index.get("objects"), list) else []
        outline = index.get("outline") if isinstance(index.get("outline"), list) else []

        def objects_for_range(p1: int, p2: int) -> List[Dict[str, Any]]:
            out = []
            for o in objects:
                try:
                    pg = int(o.get("page"))
                except Exception:
                    continue
                if p1 <= pg <= p2:
                    out.append(o)
            return out

        def outline_compact(max_items: int = 120) -> List[Dict[str, Any]]:
            # Keep it bounded; enough for section continuity.
            out = []
            for item in outline[:max_items]:
                if isinstance(item, dict) and "heading" in item and "page" in item:
                    out.append({"level": item.get("level", 1), "heading": item.get("heading"), "page": item.get("page")})
            return out

        prev_tail = ""
        all_parts: List[str] = []

        for start in range(1, n_pages + 1, self.pass2_pages_per_request):
            end = min(n_pages, start + self.pass2_pages_per_request - 1)
            part_path = parts_dir / f"pages_{start:04d}_{end:04d}.md"

            # Resume support: if already exists, reuse
            if part_path.exists():
                text = part_path.read_text(encoding="utf-8", errors="ignore")
                all_parts.append(text)
                prev_tail = text[-600:] if text else prev_tail
                recorder.checkpoint("pass2_chunk_skip_existing", {"start": start, "end": end, "path": str(part_path)})
                print(f"  Pass2: pages {start}-{end} (cached)")
                continue

            recorder.checkpoint("pass2_chunk_start", {"start": start, "end": end})
            print(f"  Pass2: pages {start}-{end}...")

            images = convert_from_path(pdf_path, dpi=self.pass2_dpi, first_page=start, last_page=end)
            images_b64 = []
            for im in images:
                images_b64.append(_pil_to_b64_jpeg(im, max_width=self.pass2_max_width, quality=self.pass2_jpeg_quality))

            # Build chunk-specific index snippet (small JSON)
            chunk_index = {
                "page_range": [start, end],
                "outline": outline_compact(),
                "objects_in_range": objects_for_range(start, end),
            }

            # Main conversion prompt (fidelity-first; no rewriting)
            prompt = f"""You are converting specific pages of a PDF (images) into Markdown.

You will be given:
- Page images for pages {start}..{end}
- A compact JSON outline + figure/table inventory for these pages
- The tail text from previous chunk (for continuity)

CRITICAL RULES (fidelity):
- Do NOT add any information not visible in the images.
- Do NOT remove visible information.
- If a word/number is unreadable, output "?" (do not guess).
- Output ONLY Markdown. No explanations. No <think>.

Structure:
- Use `#` for title only if the title appears on these pages.
- Use `##` / `###` / `####` to preserve heading hierarchy.
- Keep heading text exactly as shown on the page (do not paraphrase).

Header/Footer:
- Ignore page numbers, running headers/footers, watermarks.

Figures vs Tables (must follow captions):
- If the caption/label contains "Table", "Tab.", "表" => treat as TABLE.
- If the caption/label contains "Figure", "Fig.", "図" => treat as FIGURE.
- Never output a FIGURE as a Markdown table.
- For TABLE: output a Markdown table. If some cells are unreadable, use "?".
- For FIGURE: output a short descriptive block (paragraph + bullets) describing ONLY what is visible.

You MUST use this JSON as a guide (do not output it):
{json.dumps(chunk_index, ensure_ascii=False)}

Previous chunk tail (for continuity, do not repeat it verbatim):
{prev_tail}
""".strip()

            md = self.client.vision_request_multi(
                prompt,
                images_b64,
                model=self.vision_model,
                max_tokens=self.pass2_max_tokens,
                timeout=(30, 1800),
            ) or ""

            md = self._strip_thought_process(md)

            # Persist chunk immediately (durability)
            part_path.write_text(md, encoding="utf-8")
            all_parts.append(md)
            prev_tail = md[-600:] if md else prev_tail

            # memory cleanup
            del images, images_b64, md
            gc.collect()

            recorder.checkpoint("pass2_chunk_done", {"start": start, "end": end, "path": str(part_path)})

        return "\n\n".join(all_parts).strip() + "\n"

    # ----------------------------
    # Pass 3: deterministic cleanup + audit
    # ----------------------------
    def _pass3_finalize(self, md_text: str, index: Dict[str, Any], recorder: RunRecorder) -> Tuple[str, Dict[str, Any]]:
        # 3-1) Deterministic header/footer cleanup.
        # We only remove lines that exactly match candidates extracted in Pass1 AND appear repeatedly.
        lines = md_text.splitlines()
        cand = index.get("header_footer_candidates") if isinstance(index.get("header_footer_candidates"), list) else []
        cand = [c.strip() for c in cand if isinstance(c, str) and c.strip() and len(c.strip()) <= 120]

        # Count frequency of exact lines (trimmed)
        freq: Dict[str, int] = {}
        for ln in lines:
            t = ln.strip()
            if not t:
                continue
            freq[t] = freq.get(t, 0) + 1

        remove_set = set()
        for c in cand:
            if freq.get(c, 0) >= 3:
                remove_set.add(c)

        cleaned_lines = []
        for ln in lines:
            t = ln.strip()
            if t in remove_set:
                continue
            cleaned_lines.append(ln)

        cleaned = "\n".join(cleaned_lines)

        # 3-2) Normalize excessive blank lines (purely formatting)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"

        # 3-3) Title deduplication: ensure only one `# ` heading exists
        # Keep the first `# ` heading, convert subsequent ones to `## `
        title_found = False
        deduped_lines = []
        for line in cleaned.splitlines():
            if line.startswith('# ') and not line.startswith('## '):
                if title_found:
                    # Convert duplicate title to h2
                    deduped_lines.append('#' + line)  # # Title -> ## Title
                else:
                    deduped_lines.append(line)
                    title_found = True
            else:
                deduped_lines.append(line)
        cleaned = "\n".join(deduped_lines)

        recorder.checkpoint("pass3_cleanup_done", {"removed_candidates": len(remove_set), "title_deduplicated": title_found})

        # 3-3) Audit (LLM outputs JSON only; no rewriting allowed)
        audit_prompt = """You are auditing a Markdown conversion of a PDF.

Output ONLY valid JSON. Do NOT output Markdown. Do NOT rewrite any content.

Task:
Identify potential issues without changing content:
- figure/table type mismatch: caption says Table/表 but output is not a Markdown table; or caption says Figure/図 but output contains a Markdown table.
- missing/duplicated headings (same heading repeated suspiciously).
- obvious truncation markers (e.g., many '?' in a row).
- reference list formatting anomalies.

Rules:
- Do NOT infer or guess missing content.
- Provide only locations/snippets (short, <= 120 chars) to help manual review.

JSON schema:
{"issues":[{"type":"...","location":"...","snippet":"...","reason":"..."}]}
""".strip()

        messages = [
            {"role": "system", "content": "You are a strict auditor. Output only JSON."},
            {"role": "user", "content": audit_prompt + "\n\n" + cleaned[:120000]},
        ]
        resp = self.client.chat_completion(
            messages,
            model=self.text_model,
            temperature=0.0,
            max_tokens=self.audit_max_tokens,
            timeout=(30, 1200),
        )
        audit_raw = ""
        if resp and "choices" in resp and resp["choices"]:
            audit_raw = resp["choices"][0]["message"]["content"] or ""
        audit = _safe_json_extract(audit_raw) or {"issues": []}

        recorder.checkpoint("pass3_audit_done", {"issue_count": len(audit.get("issues", []))})

        return cleaned, audit

    def _strip_thought_process(self, text: str) -> str:
        """Remove <think> blocks and common chat prefixes."""
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        prefixes = [
            "Here is the reconstructed text:",
            "Here is the fixed text:",
            "Sure, here is the translation:",
            "Here is the Japanese translation:",
        ]
        for p in prefixes:
            if text.startswith(p):
                text = text[len(p):].lstrip()
        return text.strip()

    # ----------------------------
    # Translation (optional)
    # ----------------------------
    def _translate_full_text(self, text: str) -> str:
        """Translates the full text to Japanese using chunking."""
        if not text:
            return ""
            
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < 8000:  # Larger chunks for better context
                current_chunk += para + "\n\n"
            else:
                chunks.append(current_chunk)
                current_chunk = para + "\n\n"
        if current_chunk:
            chunks.append(current_chunk)
            
        translated_parts = []
        total_chunks = len(chunks)
        
        print(f"Translating {total_chunks} chunks...")

        for i, chunk in enumerate(chunks):
            print(f"  Translating chunk {i+1}/{total_chunks}...")
            prompt = """You are a professional academic translator.
Translate the following English Markdown text into natural, fluent Japanese Markdown.

RULES:
1. PRESERVE MARKDOWN FORMATTING (headings, bold, links, etc.) EXACTLY.
2. Translate academic terms accurately.
3. Do not omit any information.
4. Output ONLY the Japanese translation. No explanations. No <think>.

TEXT TO TRANSLATE:
"""
            
            messages = [
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": f"{prompt}\n\n{chunk}"}
            ]
            
            response = self.client.chat_completion(
                messages, 
                model=self.text_model, 
                temperature=0.3,
                max_tokens=8000,
                timeout=(30, 1800),
            )
            
            if response and 'choices' in response:
                translated = response['choices'][0]['message']['content']
                translated = self._strip_thought_process(translated)
                translated_parts.append(translated)
            else:
                print(f"  Error translating chunk {i+1}")
                translated_parts.append(chunk)  # Fallback to original

        return "\n\n".join(translated_parts)

    # ----------------------------
    # Postmortem collection
    # ----------------------------
    def _collect_postmortem_if_needed(self, recorder: RunRecorder):
        """Collect Windows logs if the previous run likely ended abruptly."""
        try:
            diag_dir = self.diag_root / "_diag"
            if not diag_dir.exists():
                return

            candidates = sorted([p for p in diag_dir.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            for p in candidates[:5]:
                if p.resolve() == recorder.run_dir.resolve():
                    continue
                status_file = p / "status.json"
                if status_file.exists():
                    st = json.loads(status_file.read_text(encoding="utf-8"))
                    if st.get("state") == "running":
                        recorder.checkpoint("postmortem_detected", {"previous_run": str(p)})
                        self._run_powershell_diag(recorder.run_dir)
                        return
        except Exception:
            return

    def _run_powershell_diag(self, work_dir: Path):
        try:
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(self._ps_diag_script), "-LookbackHours", "12"],
                cwd=str(work_dir),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


if __name__ == "__main__":
    # Example usage:
    # LocalConverter("input", "output", "translation").convert_all()
    pass
