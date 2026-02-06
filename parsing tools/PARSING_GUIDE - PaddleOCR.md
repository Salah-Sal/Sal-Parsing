# PaddleOCR Parsing Guide for arXiv Papers

**Version:** PaddleOCR 3.4.0 (January 29, 2026)
**Repo:** `./PaddleOCR`
**Use case:** Converting arXiv-style research papers (PDF) to structured markdown/JSON

---

## Architecture Overview

PaddleOCR 3.x is a modular OCR and document AI engine. It has three layers:

1. **Individual Models** — single-task: text detection, text recognition, layout detection, formula recognition, table structure, etc.
2. **Pipelines** — chain multiple models together: `PaddleOCR` (OCR), `PPStructureV3` (document parsing), `PaddleOCRVL` (VLM-based parsing)
3. **Applications** — high-level: `PPChatOCRv4Doc` (LLM Q&A over documents), `PPDocTranslation` (translate documents)

For arXiv paper parsing, the relevant pipelines are **PPStructureV3** (traditional, multi-model) and **PaddleOCRVL** (VLM-based, single model).

---

## Installation

```bash
# CPU-only (Apple Silicon / x86) — tested versions
pip install paddlepaddle==3.3.0
pip install paddleocr==3.4.0

# GPU (CUDA 11.x / 12.x)
pip install paddlepaddle-gpu==3.3.0
pip install paddleocr==3.4.0

# With document parsing extras (python-docx, etc.)
pip install paddleocr[doc-parser]

# All features (translation, information extraction, etc.)
pip install paddleocr[all]

# For PDF page-to-image conversion (required — see Gotchas)
pip install pypdfium2
```

**Python:** 3.8–3.12 (tested: 3.12.11)
**Framework:** PaddlePaddle 3.3.0, PaddleX 3.4.1 (auto-installed with paddleocr)

---

## Quality Tiers

### Tier 1 — Basic OCR (Text Only)

**What it does:** Detects text regions in each page, recognizes the text, returns bounding boxes + text + confidence scores. No layout analysis, no structure.

**Pipeline:** `PaddleOCR`

**IMPORTANT:** `PaddleOCR.predict()` does NOT support a `page_range` parameter. To process specific pages, convert them to images first using `pypdfium2`, then feed individual image paths to `predict()`.

```python
"""PaddleOCR Tier 1 — Basic OCR (tested, working code)"""
import time, os, tempfile, shutil
import pypdfium2 as pdfium
from paddleocr import PaddleOCR

paper = "paper.pdf"
NUM_PAGES = 9
DPI = 150  # 150 DPI is sufficient for OCR; 300 is overkill

# Step 1: Convert PDF pages to images (no page_range support on predict())
pdf = pdfium.PdfDocument(paper)
tmpdir = tempfile.mkdtemp()
page_images = []
for i in range(min(NUM_PAGES, len(pdf))):
    page = pdf[i]
    bitmap = page.render(scale=DPI / 72)
    img = bitmap.to_pil()
    img_path = os.path.join(tmpdir, f"page_{i+1:02d}.png")
    img.save(img_path)
    page_images.append(img_path)
pdf.close()

# Step 2: Initialize PaddleOCR — use mobile models + disable preprocessing for speed
ocr = PaddleOCR(
    lang='en',
    text_detection_model_name='PP-OCRv5_mobile_det',      # 4.7MB (vs 101MB server)
    text_recognition_model_name='en_PP-OCRv5_mobile_rec',  # lightweight
    use_doc_orientation_classify=False,  # skip orientation detection
    use_doc_unwarping=False,             # skip dewarping
    use_textline_orientation=False,      # skip textline angle
)

# Step 3: Run OCR per page
all_results = []
for img_path in page_images:
    results = ocr.predict(img_path)
    all_results.append(results)

# Step 4: Extract text — OCRResult is dict-like, NOT attribute-based
md_lines = []
for page_idx, page_results in enumerate(all_results):
    md_lines.append(f"\n## Page {page_idx + 1}\n")
    for result in page_results:
        texts = result.get('rec_texts', [])   # use dict access, NOT result.rec_texts
        for text in texts:
            md_lines.append(text)

shutil.rmtree(tmpdir)
```

**Models used:**

With default `lang='en'` (server models — heavy, slow):
| Component | Default Model | Size |
|-----------|--------------|------|
| Detection | PP-OCRv5_server_det | 101MB |
| Recognition | PP-OCRv5_server_rec | 81MB |
| Orientation | PP-LCNet_x1_0_doc_ori | 7MB |
| Unwarping | UVDoc | 30.3MB |
| Textline Ori | PP-LCNet_x1_0_textline_ori | 7MB |

With explicit mobile model names (lean — recommended for Tier 1):
| Component | Model | Size |
|-----------|-------|------|
| Detection | PP-OCRv5_mobile_det | 4.7MB |
| Recognition | en_PP-OCRv5_mobile_rec | ~16MB |

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lang` | `None` | Language: `'en'`, `'ch'`, `'fr'`, etc. (100+ languages). **Ignored when model names are specified.** |
| `ocr_version` | `None` | `'PP-OCRv3'`, `'PP-OCRv4'`, `'PP-OCRv5'` |
| `text_detection_model_name` | `None` | Override detection model (e.g., `'PP-OCRv5_mobile_det'`) |
| `text_recognition_model_name` | `None` | Override recognition model (e.g., `'en_PP-OCRv5_mobile_rec'`) |
| `use_doc_orientation_classify` | `False` | Auto-rotate pages |
| `use_doc_unwarping` | `False` | Dewarp curved pages |
| `use_textline_orientation` | `False` | Detect rotated text lines |
| `text_det_limit_side_len` | varies | Max image side length for detection |
| `text_rec_score_thresh` | varies | Min confidence for recognized text |
| `return_word_box` | `False` | Return character-level bounding boxes |

**Output:** Raw text lines with coordinates. No reading order, no structure. Two-column papers will have interleaved text (figure labels mixed with body text).

**Speed:** Fastest tier. Detection + recognition only.

**Benchmark (M2 Pro, 16GB, CPU):**
| Config | Time (9 pages) | Per page | RAM | Output |
|--------|---------------|----------|-----|--------|
| Default (`lang='en'`) | Very slow | ~90s+ | ~10GB | ✓ |
| Mobile models + no preprocessing | 277.9s | 30.9s | ~1.6GB | 44,211 chars |

**Use for:** Quick text extraction when you don't care about document structure.

---

### Tier 2 — OCR with Preprocessing

**What it does:** Same as Tier 1 but with document image preprocessing — orientation correction and dewarping. Important for scanned papers or photos of papers.

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=True,  # fix rotated pages
    use_doc_unwarping=True,             # fix curved/warped pages
)
results = ocr.predict('scanned_paper.pdf')
```

**Additional models:**
| Component | Model | Size |
|-----------|-------|------|
| Orientation | PP-LCNet_x1_0_doc_ori | 7MB |
| Unwarping | UVDoc | 30.3MB |

**Use for:** Scanned or photographed papers where pages may be rotated or warped.

---

### Tier 3 — Layout-Aware Document Parsing (PPStructureV3, Default)

**What it does:** Full document structure analysis. Detects layout regions (text, title, table, figure, formula, header, footer, reference), then runs OCR within each region. Produces structured output with reading order.

**Pipeline:** `PPStructureV3`

```python
from paddleocr import PPStructureV3

engine = PPStructureV3(lang='en')
results = engine.predict('paper.pdf')

for page in results:
    for block in page['blocks']:
        print(f"Type: {block['type']}, Text: {block.get('text', '')}")
```

**Models used:**
| Component | Default Model | Size | Purpose |
|-----------|--------------|------|---------|
| Layout | PP-DocLayout_plus-L | 126MB | Detect 20 region types |
| Text Detection | PP-OCRv5_server_det | 101MB | Find text within regions |
| Text Recognition | PP-OCRv5_server_rec | 81MB | Recognize text |
| Table Classification | PP-LCNet_x1_0_table_cls | 6.6MB | Wired vs wireless table |
| Table Structure | SLANeXt | 351MB | Predict table HTML |
| Table Cells | RT-DETR-L | 124MB | Detect table cells |

**Layout classes detected (PP-DocLayout_plus-L, 20 classes):**
- Document title, Paragraph title, Text, Page number
- Summary/Abstract, Table of contents, References, Footnote
- Header, Footer, Algorithm, Formula, Formula number
- Image, Table, Figure/table title, Seal
- Chart, Sidebar text, Reference content

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_table_recognition` | `True` | Enable table → HTML conversion |
| `use_formula_recognition` | `False` | Enable formula → LaTeX (off by default!) |
| `use_chart_recognition` | `False` | Enable chart → table conversion |
| `use_seal_recognition` | `False` | Enable seal/stamp text recognition |
| `use_region_detection` | `False` | Additional region detection |
| `layout_threshold` | `None` | Layout detection confidence threshold |
| `layout_nms` | `None` | Layout NMS threshold |
| `format_block_content` | `None` | Format output blocks |
| `markdown_ignore_labels` | `None` | Labels to skip in markdown output |

**Output:** Structured JSON with per-region type, bounding box, and content. Tables as HTML. Markdown output available.

**Use for:** Standard arXiv paper parsing with layout awareness, table extraction, and reading order.

---

### Tier 4 — Full Document Parsing with Formulas

**What it does:** Same as Tier 3 plus formula recognition. Mathematical equations are detected by the layout model and converted to LaTeX.

```python
from paddleocr import PPStructureV3

engine = PPStructureV3(
    lang='en',
    use_formula_recognition=True,  # enable LaTeX output for equations
)
results = engine.predict('paper.pdf')
```

**Additional model:**
| Component | Default Model | Size | BLEU (En) |
|-----------|--------------|------|-----------|
| Formula | PP-FormulaNet_plus-M | 592MB | 91.45% |

**Alternative formula models (swap via `formula_recognition_model_name`):**
| Model | Size | BLEU (En) | Speed | Notes |
|-------|------|-----------|-------|-------|
| PP-FormulaNet_plus-L | 698MB | 92.22% | 1476ms | Best accuracy |
| PP-FormulaNet_plus-M | 592MB | 91.45% | 1040ms | Default, balanced |
| PP-FormulaNet_plus-S | 248MB | 88.71% | 179ms | Fast |
| PP-FormulaNet-L | 695MB | 90.36% | 1482ms | English-focused |
| PP-FormulaNet-S | 224MB | 87.00% | 182ms | Lightweight |
| LaTeX_OCR_rec | 99MB | 74.55% | 1089ms | Smallest |

**Output:** Same as Tier 3, but equation regions now contain LaTeX strings (e.g., `$$E = mc^2$$`).

**Use for:** Papers with significant mathematical content where you need LaTeX output.

---

### Tier 5 — Full Document Parsing + Charts + All Features

**What it does:** Everything enabled: layout, OCR, tables, formulas, charts, seals, orientation correction.

```python
from paddleocr import PPStructureV3

engine = PPStructureV3(
    lang='en',
    use_formula_recognition=True,
    use_chart_recognition=True,
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
)
results = engine.predict('paper.pdf')
```

**Additional model:**
| Component | Default Model | Purpose |
|-----------|--------------|---------|
| Chart | PP-Chart2Table | Convert charts to tables |

**Use for:** Maximum extraction from papers with charts, formulas, complex tables, and scanned/warped pages.

---

### Tier 6 — VLM-Based Document Parsing (PaddleOCRVL)

**What it does:** Uses a 0.9B Vision-Language Model to parse documents end-to-end. A single multimodal model understands document structure directly from the image, rather than chaining separate detection → recognition models.

**Pipeline:** `PaddleOCRVL`

```python
from paddleocr import PaddleOCRVL

vlm = PaddleOCRVL(pipeline_version="v1.5")
results = vlm.predict('paper.pdf')
```

**Model:**
| Component | Model | Size | Accuracy |
|-----------|-------|------|----------|
| VLM | PaddleOCR-VL-1.5-0.9B | ~2GB | 94.5% on OmniDocBench v1.5 |

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline_version` | `"v1.5"` | `"v1"` (109 langs) or `"v1.5"` (111 langs, improved) |
| `vl_rec_backend` | `"native"` | `"native"`, `"vllm-server"`, `"sglang-server"`, `"mlx-vlm-server"` |
| `use_layout_detection` | `True` | Layout detection before VLM |
| `use_chart_recognition` | `False` | Enable chart parsing |
| `format_block_content` | `None` | Format output blocks |
| `merge_layout_blocks` | `None` | Merge adjacent blocks |
| `max_new_tokens` | varies | Max VLM output tokens |
| `temperature` | varies | VLM generation temperature |

**v1.5 improvements over v1:**
- Better real-world document handling (skew, warping, scanning, varied lighting)
- Cross-page table merging
- Cross-page heading identification
- Seal recognition
- Text spotting
- Tibetan and Bengali added (now 111 languages)

**Use for:** Highest quality parsing. Best for complex real-world documents. Handles text, tables, formulas, charts in a unified model. Requires more memory than traditional pipeline.

---

### Tier 7 — LLM-Powered Information Extraction (PPChatOCRv4Doc)

**What it does:** Combines PPStructureV3 document parsing with ERNIE 4.5 LLM for intelligent Q&A over documents. You can ask natural language questions about the parsed document.

```python
from paddleocr import PPChatOCRv4Doc

engine = PPChatOCRv4Doc()
# Parse the document
visual_results = engine.visual_predict('paper.pdf')
# Ask questions
results = engine.predict(
    visual_results,
    prompts=["What is the main contribution of this paper?"]
)
```

**Requires:** ERNIE 4.5 API access (Baidu AI cloud)

**Use for:** Document understanding, automated paper summarization, extracting specific data points from papers.

---

## Legacy PPStructure (v2.x API)

The repo also contains the older PPStructure system in `/ppstructure/`. This uses PaddleOCR 2.x APIs and is **incompatible with 3.x**.

**Legacy entry points:**
```bash
# CLI
python ppstructure/predict_system.py --image_dir=paper.pdf --recovery=True --recovery_to_markdown=True

# Python
from ppstructure.predict_system import StructureSystem
```

**Legacy models:**
| Component | Model | Classes |
|-----------|-------|---------|
| Layout (English) | PicoDet (PubLayNet) | 5: text, title, list, table, figure |
| Layout (Chinese) | PicoDet (CDLA) | 10: text, title, figure, figure_caption, table, table_caption, header, footer, reference, equation |
| Table | SLANet | HTML structure + cell matching |
| Formula | LaTeXOCR | LaTeX output |

**Legacy recovery to markdown:**
- Title → `# Heading`
- Text → Paragraphs (with smart line merging based on indentation analysis)
- Table → HTML `<table>` embedded in markdown
- Figure → `<div align="center"><img src="..."></div>`
- Equation → `$$LaTeX$$`
- Header/Footer → Skipped

**Legacy two-column detection:**
The `sorted_layout_boxes()` function detects two-column layouts by analyzing left/right x-coordinates relative to page midline. Regions are sorted top-to-bottom within each column.

**Important:** Use the 3.x API (`PPStructureV3`, `PaddleOCRVL`) for new projects. The legacy PPStructure is included for backward compatibility only.

---

## Model Catalog

### Text Detection

| Model | Type | Size | Accuracy | Speed (GPU) |
|-------|------|------|----------|-------------|
| **PP-OCRv5_server_det** | Server | 101MB | 83.8% | 70ms |
| PP-OCRv5_mobile_det | Mobile | 4.7MB | 79.0% | 6ms |
| PP-OCRv4_server_det | Server | 109MB | 82.6% | 99ms |
| PP-OCRv4_mobile_det | Mobile | 4.7MB | 63.8% | 4ms |
| PP-OCRv3_server_det | Server | 102MB | 80.1% | — |
| PP-OCRv3_mobile_det | Mobile | 2.1MB | 78.7% | — |

### Text Recognition

| Model | Type | Size | Accuracy | Languages |
|-------|------|------|----------|-----------|
| **PP-OCRv5_server_rec** | Server | 81MB | 86.4% | CN/EN/JP/Traditional/Pinyin |
| PP-OCRv5_mobile_rec | Mobile | 16MB | 81.3% | CN/EN/JP/Traditional/Pinyin |
| PP-OCRv4_server_rec_doc | Server | 182MB | 86.6% | CN (document-optimized, 15k+ chars) |
| en_PP-OCRv4_mobile_rec | Mobile | 7.5MB | 70.4% | English |
| latin_PP-OCRv3_mobile_rec | Mobile | 8.7MB | 76.9% | Latin scripts (37+ languages) |

### Layout Detection

| Model | Classes | Size | mAP | Speed (GPU) |
|-------|---------|------|-----|-------------|
| **PP-DocLayout_plus-L** | 20 | 126MB | 83.2% | 53ms |
| PP-DocLayout-L | 23 | 124MB | 90.4% | 34ms |
| PP-DocLayout-M | 23 | 22.6MB | 75.2% | 13ms |
| PP-DocLayout-S | 23 | 4.8MB | 70.9% | 12ms |
| PP-DocBlockLayout | 1 (block) | 124MB | 95.9% | 35ms |

### Formula Recognition

| Model | Size | BLEU (En) | BLEU (Zh) | Speed (GPU) |
|-------|------|-----------|-----------|-------------|
| **PP-FormulaNet_plus-L** | 698MB | 92.2% | 90.6% | 1476ms |
| PP-FormulaNet_plus-M | 592MB | 91.5% | 89.8% | 1040ms |
| PP-FormulaNet_plus-S | 248MB | 88.7% | 53.3% | 179ms |
| PP-FormulaNet-L | 695MB | 90.4% | 45.8% | 1482ms |
| PP-FormulaNet-S | 224MB | 87.0% | 45.7% | 182ms |
| LaTeX_OCR_rec | 99MB | 74.6% | 40.0% | 1089ms |

### Table Recognition

| Model | Size | Accuracy | Speed (GPU) |
|-------|------|----------|-------------|
| SLANeXt (wired) | 351MB | 69.7% | 86ms |
| SLANeXt (wireless) | 351MB | 69.7% | 86ms |
| SLANet_plus | 6.9MB | 63.7% | 23ms |
| SLANet | 6.9MB | 59.5% | 24ms |

### VLM Models

| Model | Size | Accuracy | Languages |
|-------|------|----------|-----------|
| **PaddleOCR-VL-1.5-0.9B** | ~2GB | 94.5% (OmniDocBench v1.5) | 111 |
| PaddleOCR-VL-0.9B | ~2GB | — | 109 |
| PP-DocBee2-3B | ~6GB | — | Document Q&A |

### Auxiliary Models

| Model | Size | Purpose |
|-------|------|---------|
| PP-LCNet_x1_0_doc_ori | 7MB | Page orientation (4 classes) |
| PP-LCNet_x0_25_textline_ori | 7MB | Text line orientation |
| UVDoc | 30.3MB | Document dewarping |
| PP-LCNet_x1_0_table_cls | 6.6MB | Table type classification |
| RT-DETR-L_wired_table_cell_det | 124MB | Table cell detection |
| PP-Chart2Table | — | Chart → table conversion |

---

## Recommended Configurations for arXiv Papers

### Budget / Fastest

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='en',
    text_detection_model_name='PP-OCRv5_mobile_det',
    text_recognition_model_name='en_PP-OCRv4_mobile_rec',
)
results = ocr.predict('paper.pdf')
```

**Total model size:** ~12MB
**Output:** Raw text lines, no structure

### Standard (Recommended Starting Point)

```python
from paddleocr import PPStructureV3

engine = PPStructureV3(lang='en')
results = engine.predict('paper.pdf')
```

**Total model size:** ~765MB (layout + det + rec + table models)
**Output:** Structured blocks with types, reading order, table HTML

### Research Paper Optimized

```python
from paddleocr import PPStructureV3

engine = PPStructureV3(
    lang='en',
    use_formula_recognition=True,               # LaTeX for equations
    formula_recognition_model_name='PP-FormulaNet_plus-M',  # best balance
    use_chart_recognition=True,                  # charts → tables
)
results = engine.predict('paper.pdf')
```

**Total model size:** ~1.4GB
**Output:** Structured blocks with LaTeX equations, table HTML, chart tables

### Maximum Quality (VLM)

```python
from paddleocr import PaddleOCRVL

vlm = PaddleOCRVL(pipeline_version="v1.5")
results = vlm.predict('paper.pdf')
```

**Total model size:** ~2GB
**Output:** End-to-end parsed markdown with all elements understood by a single VLM

---

## Hardware Guidance

### Apple M2 Pro, 16GB — Tested Results

| Tier | Pipeline | RAM (tested) | Time (9 pages) | Feasible? |
|------|----------|-------------|----------------|-----------|
| 1 (default server models) | PaddleOCR | **~10GB** | Very slow | Barely (swaps) |
| 1 (mobile models, lean) | PaddleOCR | **~1.6GB** | 277.9s (30.9s/pg) | Yes |
| 2 | PaddleOCR + preproc | ~2–3GB (est.) | — | Yes |
| 3 | PPStructureV3 (default) | ~3–4GB (est.) | — | Yes |
| 4 | PPStructureV3 + formulas | ~4–5GB (est.) | — | Yes |
| 5 | PPStructureV3 + all | ~5–6GB (est.) | — | Yes |
| 6 | PaddleOCRVL | ~4–6GB (est.) | — | Yes, but slow (no GPU) |
| 7 | PPChatOCRv4Doc | N/A | — | Requires ERNIE API |

All tiers run on CPU only (Apple Silicon via PaddlePaddle CPU backend — no Metal/MPS support). Performance is significantly slower than GPU. For Tier 1, the **default `lang='en'` config loads server models + all preprocessing, consuming ~10GB RAM**. Always specify mobile model names explicitly for lean operation.

**Important:** Models auto-download from HuggingFace on first use and are cached to `~/.paddlex/official_models/`. Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to skip connectivity checks on subsequent runs.

---

## Device Configuration

```python
# CPU (default on Mac)
engine = PPStructureV3(device='cpu')

# GPU
engine = PPStructureV3(device='gpu')

# Specific GPU
engine = PPStructureV3(device='gpu:0')

# Multi-GPU
engine = PPStructureV3(device='gpu:0,1')

# CPU optimizations
engine = PPStructureV3(
    device='cpu',
    enable_mkldnn=True,       # Intel MKL-DNN acceleration
    cpu_threads=10,           # thread pool
)

# GPU optimizations
engine = PPStructureV3(
    device='gpu',
    use_tensorrt=True,        # TensorRT acceleration
    precision='fp16',         # half precision
)
```

---

## CLI Usage

```bash
# Basic OCR
paddleocr predict --pipeline OCR --input paper.pdf --device cpu

# Document structure parsing
paddleocr predict --pipeline PP-StructureV3 --input paper.pdf --device cpu

# VLM parsing
paddleocr predict --pipeline PaddleOCR-VL --input paper.pdf --device cpu

# Legacy PPStructure (2.x API)
cd /path/to/PaddleOCR
python ppstructure/predict_system.py \
    --image_dir=paper.pdf \
    --recovery=True \
    --recovery_to_markdown=True \
    --output=./output/ \
    --layout_model_dir=path/to/layout_model \
    --formula=True
```

---

## MCP Server

PaddleOCR includes an MCP (Model Context Protocol) server for integration with LLM tools like Claude Desktop.

**Location:** `/mcp_server/paddleocr_mcp/`

```bash
# Install
pip install paddleocr-mcp

# Run
paddleocr-mcp
```

This allows Claude Desktop and other MCP-compatible agents to call PaddleOCR for document parsing directly.

---

## Output Format Reference

### PPStructureV3 Output (per page)

```python
{
    "blocks": [
        {
            "type": "title",           # text/title/table/figure/formula/header/footer/reference/...
            "bbox": [x1, y1, x2, y2],  # bounding box
            "text": "Section 3. Methods",
            "score": 0.95,
            "img_idx": 0,              # page index
        },
        {
            "type": "table",
            "bbox": [...],
            "html": "<html><body><table>...</table></body></html>",
        },
        {
            "type": "formula",
            "bbox": [...],
            "latex": "E = mc^2",
        },
    ]
}
```

### PaddleOCRVL Output

Returns structured markdown with elements identified by the VLM.

### Legacy PPStructure Markdown Recovery

When using `--recovery_to_markdown=True`:
- Title → `# Title text`
- Text → Paragraph (smart line merging)
- Table → HTML `<table>` embedded in markdown
- Figure → `<div align="center"><img src="path"></div>`
- Equation → `$$LaTeX$$`
- Header/Footer → Skipped

---

## Key Differences from Other Tools

| Feature | PaddleOCR | Marker | SciPDF | Docling |
|---------|-----------|--------|--------|---------|
| **Approach** | Modular ML pipeline or VLM | Surya-based deep learning | GROBID CRF + spaCy | DocLayNet + various |
| **OCR engine** | Built-in (PP-OCR) | Built-in (Surya) | None (text layer) | Tesseract/EasyOCR |
| **Formula → LaTeX** | Yes (PP-FormulaNet) | Higher tiers | No | No |
| **Table → HTML** | Yes (SLANet) | Markdown pipe tables | No (blob) | Yes |
| **VLM option** | Yes (0.9B model) | No | No | SmolDocling VLM |
| **Layout classes** | 20–23 | ~10 | 5 (PubLayNet) | 11 (DocLayNet) |
| **Language support** | 100–111 | ~50 | ~1 (GROBID) | ~50 |
| **Framework** | PaddlePaddle | PyTorch | Docker/CRF | PyTorch |
| **PDF text layer** | Optional (prefers OCR) | Primary | Primary | Primary |

---

## Known Limitations & Gotchas

1. **PaddlePaddle dependency:** PaddleOCR requires PaddlePaddle framework, not PyTorch. This means a separate framework install.
2. **Model download:** Models auto-download from HuggingFace on first use (~100MB–700MB per model). First run is slow. Cached to `~/.paddlex/official_models/`.
3. **OCR-first approach:** Unlike Marker/SciPDF/Docling which primarily extract the PDF text layer, PaddleOCR performs OCR on rendered page images. This is more robust for scanned papers but slower for born-digital PDFs.
4. **Two-column papers (Tier 1):** Basic OCR has NO reading order. In two-column papers, text from figure labels, chart axes, and body text will be interleaved randomly. Need PPStructureV3 (Tier 3+) for layout-aware reading order.
5. **No hyperlinks:** PaddleOCR works from images, not the PDF object tree. It cannot extract hyperlinks, cross-references, or PDF metadata.
6. **macOS GPU:** No GPU acceleration on macOS. CPU inference only (PaddlePaddle doesn't support Metal/MPS).
7. **No `page_range` on `predict()`:** PaddleOCR 3.x `predict()` does NOT accept a `page_range` parameter. You must convert specific pages to images first (e.g., with `pypdfium2`) and feed image paths individually.
8. **`lang` parameter ignored with model names:** When you specify `text_detection_model_name` or `text_recognition_model_name`, the `lang` parameter is silently ignored (with a warning).
9. **OCRResult is dict-like:** In PaddleOCR 3.x, results from `predict()` are dict-like objects. Use `result.get('rec_texts', [])` or `result['rec_texts']`, NOT `result.rec_texts`.
10. **Default `lang='en'` loads heavy models:** Specifying just `lang='en'` loads server_det (101MB), server_rec (81MB), orientation, unwarping, and textline models — consuming ~10GB RAM on macOS. For lean operation, explicitly set mobile model names and disable preprocessing flags.
11. **stdout buffering:** When running PaddleOCR scripts in background/piped mode, Python stdout is buffered. Add `flush=True` to `print()` or use `PYTHONUNBUFFERED=1` for real-time output.

---

## File Locations

| Path | Description |
|------|-------------|
| `paddleocr/__init__.py` | Public API exports |
| `paddleocr/_pipelines/ocr.py` | PaddleOCR class |
| `paddleocr/_pipelines/pp_structurev3.py` | PPStructureV3 class |
| `paddleocr/_pipelines/paddleocr_vl.py` | PaddleOCRVL class |
| `paddleocr/_pipelines/pp_chatocrv4_doc.py` | PPChatOCRv4Doc class |
| `paddleocr/_models/` | Individual model wrappers |
| `ppstructure/predict_system.py` | Legacy PPStructure system |
| `ppstructure/layout/predict_layout.py` | Legacy layout detection |
| `ppstructure/table/predict_table.py` | Legacy table recognition |
| `ppstructure/recovery/recovery_to_markdown.py` | Legacy markdown recovery |
| `ppocr/utils/dict/layout_dict/` | Layout class dictionaries |
| `ppocr/utils/dict/table_structure_dict.txt` | Table structure tokens |
| `configs/` | Model training configs (det/, rec/, cls/, table/, kie/) |
| `mcp_server/` | MCP server for Claude Desktop integration |