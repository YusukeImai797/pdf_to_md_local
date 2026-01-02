# Local PDF to Markdown Converter

**Text-first architecture**: Body text from PDF text layer, Vision for figures only.

## Quick Start

```powershell
# Activate conda environment
conda activate pdf_local

# Run batch processing
python main.py

# Process single file
python main.py --single "input/paper.pdf"
```

## Requirements

- Python 3.10+
- LM Studio running locally with `qwen/qwen2.5-vl-7b` loaded
- Poppler (for pdf2image, if needed)

### Install dependencies

```powershell
conda activate pdf_local
pip install -r requirements.txt
```

## Architecture

```
Pass0: Text Layer Extraction (PyMuPDF)
  ↓
Pass1: Header/Footer Inference (repetition-based)
  ↓
Pass2: Markdown Assembly (headings via font size + bold)
  ↓
Pass2b: Figure Descriptions (Vision) + Table Reconstruction (word bbox)
  ↓
Pass3: Deterministic QA (coverage, ellipsis, code fence, etc.)
  ↓
Final Markdown
```

## Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Body fidelity** | Text from PDF text layer (PyMuPDF), not Vision LLM |
| **Figure descriptions** | REQUIRED via Vision (cropped figure regions) |
| **Table reconstruction** | Hybrid-lite: word bbox clustering → Markdown table |
| **Header/footer** | Variable bands via repetition detection (not fixed ratios) |
| **QA gates** | Deterministic: ellipsis, code fences, coverage |

## Output Structure

```
output/
  <paper>_<timestamp>/
    text_layer/          # Pass0: line-level JSON per page
    words_layer/         # Pass0: word-level JSON for table reconstruction
    pass1_layout_debug.json
    pass2_body_with_placeholders.md
    pass2_full.md
    pass3_qa.json        # QA results (coverage, issues)
    <paper>.md           # Final output
```

## Configuration

Edit `pdf_to_md_textfirst.py` → `ConverterConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coverage_threshold` | 0.95 | Minimum body coverage ratio |
| `min_figure_bullets` | 3 | Minimum bullets per figure |
| `render_dpi` | 200 | DPI for figure rendering |

## Files

- `main.py` - Entry point
- `pdf_to_md_textfirst.py` - Text-first converter (main)
- `lmstudio_client.py` - LM Studio API client
- `legacy/converter.py` - Old Vision-first approach (deprecated)
