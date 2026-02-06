# Dolphin Parsing Guide

> **Dolphin** (ByteDance) — A 3B-parameter VLM-based document parser built on Qwen2.5-VL.
> Accepted at ACL 2025. Handles digital-born and photographed documents with a two-stage
> analyze-then-parse architecture.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Variants](#model-variants)
3. [Installation](#installation)
4. [Quality Tiers](#quality-tiers)
   - [Tier 1 — Layout Detection Only](#tier-1--layout-detection-only)
   - [Tier 2 — Full Page Parsing (Default)](#tier-2--full-page-parsing-default)
   - [Tier 3 — High-Resolution Parsing](#tier-3--high-resolution-parsing)
   - [Tier 4 — Element-Level Targeted Parsing](#tier-4--element-level-targeted-parsing)
   - [Tier 5 — Accelerated Inference (vLLM / TensorRT-LLM)](#tier-5--accelerated-inference-vllm--tensorrt-llm)
5. [Output Formats](#output-formats)
6. [Hardcoded Parameters Reference](#hardcoded-parameters-reference)
7. [Element Types](#element-types)
8. [Markdown Conversion Details](#markdown-conversion-details)
9. [Hardware Requirements](#hardware-requirements)
10. [Known Limitations & Gotchas](#known-limitations--gotchas)

---

## Architecture Overview

Dolphin uses a **single VLM** (Qwen2.5-VL) for the entire pipeline — no separate OCR, layout,
or table models. Everything runs through one neural network.

**Two-stage pipeline:**

1. **Stage 1 — Layout + Reading Order**: The VLM receives a full page image and the prompt
   `"Parse the reading order of this document."`. It outputs a structured string with bounding
   boxes, element labels, tags, and reading order.

2. **Stage 2 — Element-wise Content**: Each detected element is cropped from the page and sent
   back to the VLM with an element-specific prompt (e.g., `"Parse the table in the image."`).
   Elements are batched by type for efficiency.

**Distortion detection**: If >25% of detected bounding boxes overlap (IoU > 0.1), the page is
classified as a photographed/distorted document and falls back to holistic full-page parsing.

---

## Model Variants

| Model | Size | Overall Score | Text (Edit Dist) | Formula (CDM) | Table (TEDS) | Read Order (Edit) | HuggingFace |
|-------|------|---------------|-------------------|---------------|--------------|-------------------|-------------|
| Dolphin v1.0 | 0.3B | 74.67 | 0.125 | 67.85 | 68.70 | 0.124 | branch `v1.0` |
| Dolphin v1.5 | 0.3B | 85.06 | 0.085 | 79.44 | 84.25 | 0.071 | branch `v1.5` |
| **Dolphin-v2** | **3B** | **89.78** | **0.054** | **87.63** | **87.02** | **0.054** | `ByteDance/Dolphin-v2` |

**Scores from OmniDocBench v1.5.** Lower Edit Distance = better. Higher CDM/TEDS = better.

- **v2 (master branch)** — Current recommended model. 3B params, 21 element types, best quality.
- **v1.5 (branch `v1.5`)** — Lightweight 0.3B model. Better for CPU/low-RAM setups.
- **v1.0 (branch `v1.0`)** — Original 0.3B release. Superseded by v1.5.

---

## Installation

### Dependencies

```
datasets==3.6.0
torch==2.6.0
torchvision==0.21.0
transformers==4.51.0
deepspeed==0.16.4      # CUDA only — skip on macOS
triton==3.2.0          # CUDA only — skip on macOS
accelerate==1.4.0
torchcodec==0.2
decord==0.6.0
Levenshtein==0.27.1
qwen_vl_utils
matplotlib
jieba
opencv-python
bs4
albumentations==1.4.0
pymupdf==1.26
```

### Setup (CUDA GPU)

```bash
cd /path/to/Dolphin
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download model (~6GB for v2)
pip install huggingface_hub
huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model
# OR
git lfs install && git clone https://huggingface.co/ByteDance/Dolphin-v2 ./hf_model
```

### Setup (macOS / CPU-only)

`deepspeed` and `triton` are CUDA-only and will fail on macOS. Install manually:

```bash
cd /path/to/Dolphin
python -m venv venv && source venv/bin/activate

# Install without GPU-only packages
pip install torch==2.6.0 torchvision==0.21.0 transformers==4.51.0 accelerate==1.4.0
pip install datasets==3.6.0 Levenshtein==0.27.1 albumentations==1.4.0 pymupdf==1.26
pip install qwen_vl_utils matplotlib jieba opencv-python bs4
pip install torchcodec==0.2 decord==0.6.0

# Download model
huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model
```

**WARNING (Apple Silicon / CPU):** Dolphin-v2 is a 3B-parameter model. In float32 (CPU mode),
the weights alone require ~12GB RAM. On a 16GB M2 Pro, the system may run out of memory or
swap heavily. Inference will take **minutes per page**. Consider:
- Using the 0.3B v1.5 model instead: `git checkout v1.5` and download `ByteDance/Dolphin`
- Running on a machine with an NVIDIA GPU

---

## Quality Tiers

### Tier 1 — Layout Detection Only

**What:** Detect element bounding boxes and reading order. No text extraction.
**Use case:** Visualize document structure, verify layout detection quality before full parsing.
**Speed:** Fastest — single VLM pass per page (no element-wise Stage 2).

**Script:** `demo_layout.py`

```bash
# Single image
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png

# PDF (processes all pages)
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path document.pdf

# Directory of images
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `./hf_model` | Path to model weights |
| `--input_path` | (required) | Image, PDF, or directory |
| `--save_dir` | input dir | Where to save results |

**Outputs:**
- `output_json/{name}.json` — Bounding boxes with labels, reading order, tags (text field empty)
- `layout_visualization/{name}_layout.png` — Annotated image with colored overlays
- `markdown/{name}.md` — Empty (no text extracted)

**How it works** (from `demo_layout.py:173`):
```python
layout_results = model.chat("Parse the reading order of this document.", pil_image)
```
Single prompt → parse layout string → map coordinates → visualize.

---

### Tier 2 — Full Page Parsing (Default)

**What:** Complete two-stage parsing: layout detection + element-wise content extraction.
**Use case:** Primary mode for parsing documents to markdown.
**Speed:** Slower — Stage 1 + N element crops through VLM in batches.

**Script:** `demo_page.py`

```bash
# Single image
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png

# PDF (all pages, combined output)
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path document.pdf

# Directory
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs

# Custom batch size (default 4)
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path document.pdf --max_batch_size 8
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `./hf_model` | Path to model weights |
| `--input_path` | `./demo` | Image, PDF, or directory |
| `--save_dir` | input dir | Where to save results |
| `--max_batch_size` | `4` | Max elements per VLM batch |

**Outputs:**
- `output_json/{name}.json` — Full structured results with text, bboxes, labels, reading order
- `recognition_json/{name}.json` — Combined multi-page results (PDF only)
- `markdown/{name}.md` — Formatted markdown document
- `markdown/figures/{name}_figure_NNN.png` — Extracted figure images
- `layout_visualization/{name}_layout.png` — Annotated layout

**Pipeline** (from `demo_page.py`):
1. PDF → images via PyMuPDF at `target_size=896` px
2. Stage 1: `"Parse the reading order of this document."` → layout string
3. Parse layout → check for distortion (bbox overlap)
4. Crop elements → group by type (table, equation, code, text, figure)
5. Stage 2: Batch-process each group with type-specific prompts
6. Sort results by reading order → convert to markdown

**Element-specific prompts** (hardcoded):

| Element Type | Prompt | Label |
|-------------|--------|-------|
| Table | `"Parse the table in the image."` | `tab` |
| Formula | `"Read formula in the image."` | `equ` |
| Code | `"Read code in the image."` | `code` |
| Text/Paragraph | `"Read text in the image."` | `para`, `sec_*`, `list`, etc. |
| Figure | (saved as image, not sent to VLM) | `fig` |

**`max_batch_size` tuning:**
- **Lower (1-2):** Slower, each element processed individually. May produce slightly
  more consistent results since elements don't compete in a batch.
- **Default (4):** Balanced speed and quality.
- **Higher (8+):** Faster on GPU, more VRAM usage. On CPU, minimal benefit.

---

### Tier 3 — High-Resolution Parsing

**What:** Same pipeline as Tier 2 but with higher image resolution for better detail.
**Use case:** Dense documents, small text, complex tables/formulas.
**Speed:** Slower than Tier 2 due to larger images.

**Requires code modification** — resolution parameters are hardcoded, not exposed via CLI.

**Two key parameters to modify:**

1. **PDF rendering resolution** — `utils/utils.py:54` `convert_pdf_to_images()`:
   ```python
   def convert_pdf_to_images(pdf_path, target_size=896):  # default 896
   ```
   Increase to 1600 or 2048 for higher-quality page images.

2. **Image input cap** — `utils/utils.py:432` `resize_img()`:
   ```python
   def resize_img(image, max_size=1600, min_size=28):  # caps at 1600
   ```
   This limits the largest dimension sent to the VLM. Can increase but watch for
   memory/speed impact.

3. **Qwen VL smart resize** — Applied after `resize_img`, constrains to a 28-pixel grid:
   ```python
   smart_resize(h, w, factor=28, min_pixels=784, max_pixels=2560000)
   ```
   `max_pixels=2560000` (~1600x1600) is the effective upper bound from the Qwen processor.

**Example modification for higher resolution:**

```python
# In utils/utils.py, change:
def convert_pdf_to_images(pdf_path, target_size=1600):  # was 896

def resize_img(image, max_size=2048, min_size=28):      # was 1600
```

**Trade-offs:**
- Higher resolution → better small-text/detail recognition
- Higher resolution → more VRAM/RAM usage, slower inference
- Beyond ~1600px, gains diminish due to the Qwen VL `max_pixels` constraint

---

### Tier 4 — Element-Level Targeted Parsing

**What:** Parse individual element images (pre-cropped tables, formulas, code, text).
**Use case:** When you already have element crops (from another layout detector or manual cropping)
and want Dolphin's VLM to recognize specific content.
**Speed:** One VLM call per element.

**Script:** `demo_element.py`

```bash
# Parse a table image
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path table_crop.png --element_type table

# Parse a formula image
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path equation.png --element_type formula

# Parse code
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path code_block.png --element_type code

# Parse text (default)
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path paragraph.png --element_type text

# Process a directory of element images
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/element_imgs --element_type table

# Print results to console
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path table.png --element_type table --print_results
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `./hf_model` | Path to model weights |
| `--input_path` | (required) | Image or directory of images |
| `--element_type` | `text` | One of: `text`, `table`, `formula`, `code` |
| `--save_dir` | input dir | Where to save results |
| `--print_results` | false | Print results to console |

**Element type → VLM prompt mapping:**

| `--element_type` | Prompt | Output Label |
|-----------------|--------|--------------|
| `text` | `"Read text in the image."` | `para` |
| `table` | `"Parse the table in the image."` | `tab` |
| `formula` | `"Read formula in the image."` | `equ` |
| `code` | `"Read code in the image."` | `code` |

**When to use this tier:**
- You want to parse specific elements from a document
- You have a better layout detector and want to use Dolphin only for content recognition
- You're evaluating Dolphin's recognition quality on specific element types

---

### Tier 5 — Accelerated Inference (vLLM / TensorRT-LLM)

**What:** Production deployment with optimized inference engines.
**Use case:** High-throughput batch processing, API serving.
**Speed:** Significantly faster than raw HuggingFace inference.

The README references deployment support for:
- **vLLM** — See `deployment/vllm/` (available on GitHub, not in this clone)
- **TensorRT-LLM** — See `deployment/tensorrt_llm/` (available on GitHub, not in this clone)

**Requirements:** NVIDIA GPU with CUDA. Not applicable for CPU/macOS.

To access deployment code, check the GitHub repository's `deployment/` directory or the
linked documentation in the README.

---

## Output Formats

### JSON Output (`output_json/{name}.json`)

Per-element structured data:

```json
[
  {
    "label": "sec_0",
    "text": "Document Title",
    "bbox": [210, 136, 910, 172],
    "reading_order": 0,
    "tags": []
  },
  {
    "label": "para",
    "text": "First paragraph text...",
    "bbox": [202, 217, 921, 325],
    "reading_order": 1,
    "tags": ["author"]
  },
  {
    "label": "tab",
    "text": "<table>...</table>",
    "bbox": [100, 500, 800, 700],
    "reading_order": 5,
    "tags": []
  }
]
```

### Combined PDF JSON (`recognition_json/{name}.json`)

```json
{
  "source_file": "document.pdf",
  "total_pages": 9,
  "pages": [
    {
      "page_number": 1,
      "elements": [ ... ]
    },
    ...
  ]
}
```

### Markdown Output (`markdown/{name}.md`)

Converted from JSON via `MarkdownConverter`. Elements sorted by reading order.
Pages separated by `---` for multi-page PDFs. Figures referenced as relative paths.

### Layout Visualization (`layout_visualization/{name}_layout.png`)

Annotated image with:
- Colored semi-transparent overlays per element (alpha=0.3)
- Labels: `"{reading_order}: {label} | {tags}"`
- 20-color palette (light pastels, cycled)

---

## Hardcoded Parameters Reference

These parameters are NOT exposed via CLI and require code edits to change:

| Parameter | Value | Location | Impact |
|-----------|-------|----------|--------|
| PDF render target size | `896` px | `utils.py:54` `convert_pdf_to_images()` | PDF page resolution |
| Image max size | `1600` px | `utils.py:432` `resize_img()` | Max dimension sent to VLM |
| Image min size | `28` px | `utils.py:432` `resize_img()` | Min dimension (upscaled if smaller) |
| Max new tokens | `4096` | `demo_page.py:105` `model.generate()` | Max output length per VLM call |
| do_sample | `False` | `demo_page.py:106` | Deterministic generation |
| temperature | `None` | `demo_page.py:107` | N/A when do_sample=False |
| repetition_penalty | `1.05` | `demo_page.py:108` (commented out) | Uncomment to penalize repetition |
| Figure PNG quality | `95` | `utils.py:43` `save_figure_to_local()` | Saved figure compression |
| Layout viz alpha | `0.3` | `utils.py:291` `visualize_layout()` | Overlay transparency |
| IoU threshold | `0.1` | `utils.py:487` `check_bbox_overlap()` | Overlap detection sensitivity |
| Overlap box ratio | `0.25` | `utils.py:487` `check_bbox_overlap()` | % boxes for distortion flag |
| Min element size | `3x3` px | `demo_page.py:230` | Skip tiny detected elements |
| Smart resize factor | `28` | `utils.py:208` via `qwen_vl_utils` | Qwen VL grid alignment |
| Smart resize max pixels | `2,560,000` | `utils.py:208` via `qwen_vl_utils` | ~1600x1600 effective cap |
| Margin crop threshold | `200` (grayscale) | `utils.py:270` `crop_margin()` | Content vs. background |

---

## Element Types

Dolphin-v2 detects 21 element types. The main ones relevant to parsing:

| Label | Description | Stage 2 Processing |
|-------|-------------|-------------------|
| `para` | Paragraph text | VLM: "Read text in the image." |
| `sec_0` | Section heading (H1) | VLM: "Read text in the image." → `#` |
| `sec_1` | Subsection heading (H2) | VLM: "Read text in the image." → `##` |
| `sec_2`–`sec_5` | Deeper headings (all → H3) | VLM: "Read text in the image." → `###` |
| `tab` | Table | VLM: "Parse the table in the image." → HTML |
| `equ` | Equation/formula | VLM: "Read formula in the image." → LaTeX |
| `code` | Code block | VLM: "Read code in the image." → ` ```bash ` |
| `fig` | Figure/image | Saved as PNG (not sent to VLM) |
| `list` | List item | VLM: "Read text in the image." → `- item` |
| `fnote` | Footnote | VLM: "Read text in the image." |
| `watermark` | Watermark | Detected but typically ignored |
| `distorted_page` | Fallback for distorted docs | Full-page VLM processing |

**Tags** (metadata attached to elements): `author`, `paper_abstract`, `meta_num`, etc.

---

## Markdown Conversion Details

The `MarkdownConverter` class (`utils/markdown_utils.py`) transforms JSON results to markdown:

**Heading mapping:**
- `sec_0` → `#` (H1)
- `sec_1` → `##` (H2)
- `sec_2` through `sec_5` → `###` (all flattened to H3)

**Content handling by type:**
- **Text/paragraphs**: Newline cleanup (hyphenated breaks removed, Chinese-aware joining)
- **Tables**: Extracted as HTML `<table>` tags (NOT markdown pipe tables)
- **Formulas**: Wrapped in `$$...$$` LaTeX blocks
- **Code**: Wrapped in ` ```bash ` fenced blocks
- **Figures**: `![Figure N](../figures/{filename})`
- **Lists**: `- {item text}`

**LaTeX normalization** (applied to formulas and inline math):
- `\bm` → `\mathbf`
- `\upmu` → `\mu`
- `\varmathbb` → `\mathbb`
- `\in fty` → `\infty`
- `\eqno` → `\quad`

**Page separators** (multi-page PDF): `\n\n---\n\n` between pages.

---

## Hardware Requirements

### GPU (Recommended)

| Model | VRAM (bfloat16) | Speed (approx.) |
|-------|-----------------|-----------------|
| Dolphin-v2 (3B) | ~8GB | Seconds per page |
| Dolphin-v1.5 (0.3B) | ~2GB | Sub-second per page |

### CPU / Apple Silicon

| Model | RAM (float32) | Expected Speed |
|-------|---------------|----------------|
| Dolphin-v2 (3B) | ~12GB weights + overhead | **Minutes per page** (very slow) |
| Dolphin-v1.5 (0.3B) | ~1.5GB weights + overhead | Seconds to tens of seconds per page |

**Apple M2 Pro (16GB) considerations:**
- Dolphin-v2 (3B) in float32 needs ~12GB for weights alone. With PyTorch overhead and
  image processing, this will likely cause heavy memory pressure and swapping. May OOM.
- Dolphin-v1.5 (0.3B) is far more practical for CPU testing.
- No Metal/MPS GPU acceleration — PyTorch's MPS backend is not used by the code
  (it only checks `torch.cuda.is_available()`).

### Disk Space

- Dolphin-v2 model: ~6GB
- Dolphin-v1.5 model: ~600MB
- Dependencies (venv): ~3-5GB (mostly PyTorch)

---

## Known Limitations & Gotchas

1. **No page range support**: `demo_page.py` processes ALL pages of a PDF. There is no
   `--page_range` argument. To process specific pages, you need to either:
   - Pre-split the PDF (e.g., with PyMuPDF)
   - Modify `convert_pdf_to_images()` in `utils/utils.py`

2. **GPU-only packages in requirements.txt**: `deepspeed==0.16.4` and `triton==3.2.0`
   will fail to install on macOS/CPU-only systems. Install other packages manually.

3. **No pip-installable package**: Dolphin is a collection of scripts, not a Python package.
   No `setup.py` or `pyproject.toml` with build config. No `__init__.py` files. Must run
   scripts directly from the repo directory.

4. **Tables output as HTML, not markdown**: The VLM produces HTML `<table>` tags. The
   markdown converter preserves them as HTML in the `.md` file. This is valid markdown but
   may not render in all viewers.

5. **All prompts are hardcoded**: You cannot customize VLM prompts via CLI. Changing prompts
   requires editing `demo_page.py` or `demo_element.py`.

6. **PDF resolution default is low (896px)**: The `target_size=896` for PDF conversion is
   optimized for speed. For dense documents, increase via code edit.

7. **DOLPHIN class is duplicated**: Each demo script (`demo_page.py`, `demo_element.py`,
   `demo_layout.py`) contains its own copy of the `DOLPHIN` class. They are nearly identical
   but have minor differences (e.g., `demo_element.py:100-103` does NOT set `do_sample=False`).

8. **Heading hierarchy is flat**: Sections `sec_2` through `sec_5` ALL map to `###` (H3).
   You lose heading depth below H2.

9. **Code blocks always tagged as bash**: All code output uses ` ```bash ` regardless of
   actual language.

10. **No progress bar**: Processing large PDFs gives only per-page `"Processing page X/Y"`
    output with no ETA or progress bar.

11. **License is non-commercial (README says MIT but model has Qwen Research License)**:
    The code is MIT-licensed, but the underlying Qwen2.5-VL model has its own license terms.
    Check the model card for commercial use restrictions.

12. **Distortion detection can misfire**: The overlap-based distortion detector
    (IoU > 0.1 for >25% of boxes) may trigger on legitimately overlapping elements
    (e.g., watermarks, marginalia), causing fallback to full-page mode which loses
    element-level structure.

---

## Quick Reference: Running Each Tier

```bash
# Tier 1: Layout only (fastest)
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path document.pdf

# Tier 2: Full parsing (default, recommended)
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path document.pdf

# Tier 2 with larger batch (faster on GPU)
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path document.pdf --max_batch_size 8

# Tier 4: Element-level (targeted)
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path cropped_table.png --element_type table --print_results
```

**Supported input formats:** `.jpg`, `.jpeg`, `.png`, `.pdf`

**Output directory structure:**
```
results/
├── markdown/
│   ├── document.md
│   └── figures/
│       ├── document_figure_000.png
│       └── document_figure_001.png
├── output_json/
│   └── document.json
├── recognition_json/          # PDF only
│   └── document.json
└── layout_visualization/
    └── document_layout.png
```
