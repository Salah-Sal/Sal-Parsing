# Marker PDF Parsing Guide: From Fast to Highest Quality

A practical guide for parsing arXiv-style papers with [Marker](https://github.com/datalab-to/marker), covering every quality tier from the fastest low-quality parse to the highest-fidelity conversion.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Tier Overview](#tier-overview)
- [Tier 1: Fastest / Lowest Quality](#tier-1-fastest--lowest-quality)
- [Tier 2: Fast / Acceptable Quality](#tier-2-fast--acceptable-quality)
- [Tier 3: Balanced (Default)](#tier-3-balanced-default)
- [Tier 4: High Quality (Local Models)](#tier-4-high-quality-local-models)
- [Tier 5: Highest Quality (LLM-Enhanced)](#tier-5-highest-quality-llm-enhanced)
- [LLM Service Options](#llm-service-options)
- [Key Parameters Reference](#key-parameters-reference)
- [Output Formats](#output-formats)
- [Batch Processing](#batch-processing)
- [Memory & GPU Tuning](#memory--gpu-tuning)
- [Tips for arXiv Papers](#tips-for-arxiv-papers)

---

## Quick Start

```bash
# Install
pip install marker-pdf

# Basic conversion (Tier 3 - balanced defaults)
marker_single paper.pdf --output_dir ./output

# Highest quality (Tier 5 - requires API key)
export GOOGLE_API_KEY="your-gemini-key"
marker_single paper.pdf --output_dir ./output --use_llm --force_ocr --redo_inline_math
```

---

## Tier Overview

| Tier | Speed | Quality | OCR | LLM | Cost | Best For |
|------|-------|---------|-----|-----|------|----------|
| **1 - Fastest** | ~2-5s/page | Low | Off | No | Free | Bulk text extraction, quick scanning |
| **2 - Fast** | ~5-10s/page | Acceptable | Partial | No | Free | Clean digital PDFs |
| **3 - Balanced** | ~10-20s/page | Good | Auto | No | Free | General-purpose (default) |
| **4 - High** | ~15-30s/page | Very Good | Full | No | Free | Scanned/complex PDFs, local only |
| **5 - Highest** | ~20-60s/page | Excellent | Full | Yes | API costs | arXiv papers with math/tables |

> Speeds are approximate for a single GPU. CPU-only will be significantly slower.

---

## Tier 1: Fastest / Lowest Quality

**Goal:** Extract text as fast as possible, skip all heavy processing.

```bash
marker_single paper.pdf \
    --output_dir ./output \
    --disable_image_extraction \
    --config_json tier1.json
```

`tier1.json`:
```json
{
    "disable_ocr": true,
    "lowres_image_dpi": 72,
    "highres_image_dpi": 72,
    "disable_image_extraction": true,
    "detection_batch_size": 24,
    "layout_batch_size": 24
}
```

**What this does:**
- Skips OCR entirely — relies solely on embedded PDF text
- Uses minimum DPI for layout detection
- Skips image extraction
- Maximizes batch sizes for speed

**Limitations:**
- Equations will be garbled or missing
- Scanned PDFs produce no output
- Tables may be poorly formatted
- No images in output
- Figures/charts are lost

**When to use:** You only need the raw body text from clean, digitally-authored PDFs and don't care about math, tables, or figures.

---

## Tier 2: Fast / Acceptable Quality

**Goal:** Reasonably fast with basic structural preservation.

```bash
marker_single paper.pdf \
    --output_dir ./output \
    --config_json tier2.json
```

`tier2.json`:
```json
{
    "lowres_image_dpi": 96,
    "highres_image_dpi": 96,
    "detection_batch_size": 16,
    "recognition_batch_size": 64,
    "layout_batch_size": 16,
    "equation_batch_size": 8,
    "disable_image_extraction": true
}
```

**What this does:**
- Runs layout detection and OCR at reduced resolution
- Uses larger batch sizes to maximize throughput
- Skips image extraction to save time
- OCR triggers automatically only on pages with bad/missing text

**Limitations:**
- Lower OCR accuracy from reduced DPI
- Equations recognized but may have errors
- Tables recognized but cells may be misaligned
- No images in output

**When to use:** Quick processing of digital PDFs where you need structure (headings, lists) but can tolerate some errors in math and tables.

---

## Tier 3: Balanced (Default)

**Goal:** Good quality across all content types. This is what you get out of the box.

```bash
marker_single paper.pdf --output_dir ./output
```

Equivalent explicit config:
```json
{
    "lowres_image_dpi": 96,
    "highres_image_dpi": 192,
    "force_ocr": false,
    "extract_images": true,
    "disable_image_extraction": false
}
```

**What this does:**
- Layout detection at 96 DPI (fast, sufficient for structure)
- OCR at 192 DPI (good accuracy)
- Automatic OCR: only runs on pages where embedded text is bad/missing
- Equation recognition with Surya's texify model
- Table recognition with Surya's table model
- Header/footer detection and removal
- Image extraction
- List and section hierarchy detection

**Quality for arXiv papers:**
- Body text: Excellent
- Section headings: Excellent
- Inline math: Moderate (some errors in complex expressions)
- Display equations: Good (simple) to moderate (complex multi-line)
- Tables: Good (simple) to moderate (complex spanning cells)
- Figures: Extracted as images
- References: Good

**When to use:** General-purpose conversion. Start here and move up only if quality is insufficient.

---

## Tier 4: High Quality (Local Models)

**Goal:** Maximize quality using only local models — no API keys or costs.

```bash
marker_single paper.pdf \
    --output_dir ./output \
    --force_ocr \
    --config_json tier4.json
```

`tier4.json`:
```json
{
    "force_ocr": true,
    "highres_image_dpi": 300,
    "lowres_image_dpi": 150,
    "recognition_batch_size": 32,
    "detection_batch_size": 8,
    "layout_batch_size": 8,
    "equation_batch_size": 4,
    "table_rec_batch_size": 4,
    "extract_images": true,
    "paginate_output": true
}
```

**What this does:**
- Forces OCR on every page (ignores embedded text which can be noisy)
- Higher DPI for both layout detection and OCR
- Smaller batch sizes to ensure stable processing at higher resolution
- Full table and equation recognition
- Page numbers in output for reference

**Quality improvement over Tier 3:**
- Better OCR accuracy from higher DPI
- More reliable on scanned or poorly-digitized PDFs
- Better equation rendering
- More accurate table cell detection

**Limitations:**
- Complex LaTeX equations may still have errors
- Tables with spanning cells or nested headers may not be perfect
- Slower than Tier 3 (~2x)
- More GPU memory required

**When to use:** Scanned papers, PDFs with embedded text you don't trust, or when you need better math/table accuracy but want to stay local and free.

---

## Tier 5: Highest Quality (LLM-Enhanced)

**Goal:** Best possible quality. Uses a vision LLM to correct and refine all outputs.

### 5a: Standard LLM Enhancement

```bash
export GOOGLE_API_KEY="your-gemini-key"

marker_single paper.pdf \
    --output_dir ./output \
    --use_llm \
    --force_ocr \
    --config_json tier5a.json
```

`tier5a.json`:
```json
{
    "use_llm": true,
    "force_ocr": true,
    "highres_image_dpi": 300,
    "lowres_image_dpi": 150,
    "max_concurrency": 3,
    "extract_images": true,
    "paginate_output": true
}
```

### 5b: Maximum Quality (with inline math redo)

```bash
export GOOGLE_API_KEY="your-gemini-key"

marker_single paper.pdf \
    --output_dir ./output \
    --use_llm \
    --force_ocr \
    --redo_inline_math \
    --config_json tier5b.json
```

`tier5b.json`:
```json
{
    "use_llm": true,
    "force_ocr": true,
    "redo_inline_math": true,
    "highres_image_dpi": 300,
    "lowres_image_dpi": 150,
    "max_concurrency": 5,
    "extract_images": true,
    "paginate_output": true
}
```

**What `--use_llm` enables (automatically):**
- **LLM Table Processor** — Sends table images to the LLM for accurate HTML table reconstruction. Handles complex spanning cells, merged headers, and rotated tables.
- **LLM Table Merge Processor** — Detects and merges tables split across pages.
- **LLM Equation Processor** — Sends equation images to the LLM for accurate LaTeX conversion. Far better than local OCR for complex multi-line equations.
- **LLM Image Description Processor** — Generates alt-text descriptions for figures (when `extract_images=true`, images are still extracted; descriptions are added).
- **LLM Page Correction Processor** — Final pass that compares each page's rendered markdown against the original page image and fixes formatting errors.

**What `--redo_inline_math` adds:**
- **LLM Math Block Processor** — Re-processes all inline math blocks (`$...$`) through the LLM. This catches errors in inline expressions that Surya's local model may have gotten wrong.

**Quality for arXiv papers:**
- Body text: Excellent
- Section headings: Excellent
- Inline math: Excellent (with `--redo_inline_math`)
- Display equations: Excellent (complex multi-line, aligned environments)
- Tables: Excellent (spanning cells, multi-row headers)
- Figures: Extracted + alt-text descriptions
- References: Excellent

**Cost estimates (Gemini Flash):**
- ~$0.01-0.05 per page depending on complexity
- A 10-page arXiv paper: ~$0.10-0.50
- Majority of cost comes from table and equation processing

**When to use:** When accuracy matters more than cost/speed — final conversions, building training data, or archiving important papers.

---

## LLM Service Options

Marker supports multiple LLM backends. Choose based on cost, quality, and privacy needs.

### Cloud Services (Best Quality)

| Service | Flag | Model | Cost | Notes |
|---------|------|-------|------|-------|
| **Gemini** (default) | `--llm_service marker.services.gemini.GoogleGeminiService` | `gemini-2.0-flash` | Cheapest | Best cost/quality ratio |
| **Gemini via Vertex** | `--llm_service marker.services.vertex.GoogleVertexService` | `gemini-2.0-flash-001` | Similar | For GCP users, enterprise |
| **Claude** | `--llm_service marker.services.claude.ClaudeService` | `claude-3-7-sonnet` | Higher | Excellent quality |
| **OpenAI** | `--llm_service marker.services.openai.OpenAIService` | `gpt-4o-mini` | Moderate | Good quality |
| **Azure OpenAI** | `--llm_service marker.services.azure_openai.AzureOpenAIService` | Custom | Varies | Enterprise/compliance |

### Local LLM (Free, Private)

| Service | Flag | Model | Notes |
|---------|------|-------|-------|
| **Ollama** | `--llm_service marker.services.ollama.OllamaService` | `llama3.2-vision` | Free, runs locally, lower quality |

### Examples

**Gemini (recommended for cost):**
```bash
export GOOGLE_API_KEY="your-key"
marker_single paper.pdf --use_llm
```

**Claude (high quality):**
```bash
marker_single paper.pdf --use_llm \
    --llm_service marker.services.claude.ClaudeService \
    --claude_api_key "your-key"
```

**OpenAI:**
```bash
marker_single paper.pdf --use_llm \
    --llm_service marker.services.openai.OpenAIService \
    --openai_api_key "your-key"
```

**Ollama (free, local):**
```bash
# First: ollama pull llama3.2-vision
marker_single paper.pdf --use_llm \
    --llm_service marker.services.ollama.OllamaService \
    --ollama_model llama3.2-vision
```

---

## Key Parameters Reference

### Parameters That Most Affect Quality

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `use_llm` | `false` | bool | **Largest quality jump** — enables LLM refinement |
| `force_ocr` | `false` | bool | Re-OCRs everything; helps with bad embedded text |
| `redo_inline_math` | `false` | bool | LLM-corrects inline math (requires `use_llm`) |
| `highres_image_dpi` | `192` | 72-400 | Higher = better OCR accuracy, slower |
| `lowres_image_dpi` | `96` | 72-200 | Higher = better layout detection, slower |

### Parameters That Most Affect Speed

| Parameter | Default | Impact |
|-----------|---------|--------|
| `disable_ocr` | `false` | Skipping OCR is the biggest speed gain |
| `disable_image_extraction` | `false` | Saves I/O and processing time |
| `detection_batch_size` | Auto | Larger = faster (more VRAM) |
| `recognition_batch_size` | Auto | Larger = faster (more VRAM) |
| `layout_batch_size` | Auto | Larger = faster (more VRAM) |
| `equation_batch_size` | Auto | Larger = faster (more VRAM) |
| `max_concurrency` | 3 | More concurrent LLM calls (Tier 5) |
| `pdftext_workers` | 4 | More workers for text extraction |

### Parameters for Specific Content

| Content Type | Key Parameters |
|--------------|----------------|
| **Math (display)** | `equation_batch_size`, `model_max_length` (default 1024) |
| **Math (inline)** | `redo_inline_math`, `disable_ocr_math`, `inlinemath_min_ratio` |
| **Tables** | `table_rec_batch_size`, `row_split_threshold`, `max_table_rows` |
| **Headers/Footers** | `common_element_threshold`, `text_match_threshold` |
| **Lists** | `min_x_indent`, `list_gap_threshold` |
| **Columns** | `column_gap_ratio` |
| **Images** | `extract_images`, `image_expansion_ratio` |

---

## Output Formats

```bash
# Markdown (default) — best for reading and downstream processing
marker_single paper.pdf --output_format markdown

# JSON — structured block tree with coordinates, useful for programmatic access
marker_single paper.pdf --output_format json

# HTML — rendered HTML with images, equations, and tables
marker_single paper.pdf --output_format html

# Chunks — flat list of top-level blocks with HTML, ideal for RAG pipelines
marker_single paper.pdf --output_format chunks
```

| Format | Use Case |
|--------|----------|
| `markdown` | Reading, LLM context, documentation |
| `json` | Programmatic access, block coordinates, custom pipelines |
| `html` | Web display, rich rendering |
| `chunks` | RAG / vector database ingestion |

---

## Batch Processing

### Single Machine, Multiple Files

```bash
# Process a folder of PDFs
marker /path/to/pdfs/ --output_dir ./output --workers 4

# Skip already-converted files
marker /path/to/pdfs/ --output_dir ./output --skip_existing

# Limit to first 100 files
marker /path/to/pdfs/ --output_dir ./output --max_files 100
```

### Multi-GPU

```bash
# 4 GPUs, 15 total workers
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert /path/to/pdfs/ ./output
```

### Specific Page Ranges

```bash
# Only pages 1-5 and 10
marker_single paper.pdf --page_range "0,1-4,9"
```

---

## Memory & GPU Tuning

### Low VRAM (4-8 GB)

```json
{
    "detection_batch_size": 2,
    "recognition_batch_size": 8,
    "layout_batch_size": 2,
    "equation_batch_size": 2,
    "table_rec_batch_size": 2,
    "pdftext_workers": 1,
    "max_concurrency": 1
}
```

### Medium VRAM (8-16 GB)

Use defaults — Marker auto-detects appropriate batch sizes.

### High VRAM (24+ GB)

```json
{
    "detection_batch_size": 24,
    "recognition_batch_size": 128,
    "layout_batch_size": 24,
    "equation_batch_size": 16,
    "table_rec_batch_size": 8
}
```

### CPU Only

```bash
TORCH_DEVICE=cpu marker_single paper.pdf --disable_multiprocessing
```

> CPU processing is significantly slower (5-10x). Consider using `--page_range` to limit pages.

---

## Tips for arXiv Papers

1. **Start with Tier 3** (defaults). Most arXiv papers are clean digital PDFs, and the default settings handle body text, headings, and references well.

2. **If equations are wrong**, jump to Tier 5a with `--use_llm`. The LLM equation processor is dramatically better at multi-line aligned equations, matrices, and complex notation.

3. **If inline math like `$\alpha$` is wrong**, add `--redo_inline_math` (Tier 5b). This is the most common issue with arXiv papers.

4. **If tables are garbled**, `--use_llm` fixes most table issues. For very large tables (>175 rows), increase `--max_table_rows`.

5. **Two-column layouts** are handled automatically by the layout model. No special configuration needed.

6. **Appendices and supplementary material** may have different formatting. Use `--page_range` to process them separately if needed.

7. **For a quick check** before a full conversion, try converting just the first few pages:
   ```bash
   marker_single paper.pdf --page_range "0-2" --output_dir ./test_output
   ```

8. **Debug mode** saves layout detection images so you can see what Marker "sees":
   ```bash
   marker_single paper.pdf --debug --output_dir ./debug_output
   ```

---

## Quick Decision Tree

```
Is the PDF clean digital text (not scanned)?
├── Yes
│   ├── Do you need math/equations to be accurate?
│   │   ├── No  → Tier 2 or 3
│   │   └── Yes
│   │       ├── Can you use an API? → Tier 5a or 5b
│   │       └── Must stay local?   → Tier 4
│   └── Do you only need body text?
│       └── Yes → Tier 1
└── No (scanned)
    ├── Can you use an API? → Tier 5a with --force_ocr
    └── Must stay local?   → Tier 4 with --force_ocr
```
