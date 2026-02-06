# Docling: arXiv Paper Parsing Guide

**Tool:** [Docling](https://github.com/docling-project/docling) by IBM Research Zurich
**Version:** 2.72.0
**License:** MIT
**Python:** 3.10+

Docling is a modular document processing framework that converts documents to Markdown (and other formats) using a multi-stage pipeline of AI models. It supports layout analysis, OCR, table extraction, formula recognition, image description, and full Vision-Language Model (VLM) processing. It sits between MarkItDown (lightweight extraction) and Marker (deep-learning OCR) in terms of flexibility, but offers the most pipeline customization of the three.

---

## Quick Reference

| Tier | Method | Speed | Quality | Cost | Best For |
|------|--------|-------|---------|------|----------|
| 1 | Backend text only (no models) | ~seconds | Basic | Free | Quick text dump from clean PDFs |
| 2 | Basic layout (Heron, no OCR) | ~seconds/page | Good | Free | Digital PDFs, basic structure |
| 3 | Standard pipeline (default) | ~seconds/page | Better | Free | General documents, tables + OCR |
| 4 | Enhanced pipeline (Egret + accurate tables + formulas) | ~10-30s/page | High | Free | Complex scientific papers |
| 5 | VLM pipeline (Granite Docling) | ~minutes/page | Highest | Free (local) / API cost | Best fidelity, complex layouts |

---

## Installation

```bash
# Minimal (core only)
pip install docling

# With OCR support (recommended for arXiv papers)
pip install 'docling[ocr]'

# With VLM support (for Tier 5)
pip install 'docling[vlm]'

# With audio transcription
pip install 'docling[asr]'

# Everything
pip install 'docling[all]'
```

### Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `ocr` | `rapidocr`, `easyocr`, `tesserocr` | OCR engines |
| `vlm` | `transformers`, `torch`, `mlx` | Vision-Language Models |
| `asr` | `whisper`, `pydub` | Audio transcription |
| `all` | All of the above | Everything |

### System Dependencies

Some OCR engines need system packages:

```bash
# macOS (for Tesseract)
brew install tesseract

# macOS Vision (built-in, no install needed — OcrMacOptions)

# Ubuntu/Debian
apt-get install tesseract-ocr libtesseract-dev
```

---

## Tier 1: Backend Text Only (Fastest, Basic)

Extracts embedded text from the PDF using the Docling backend parser without running any AI models. Similar to what MarkItDown does.

### CLI

```bash
# There's no direct CLI flag for "no models" mode.
# The closest is using the standard pipeline with everything disabled.
# For truly model-free extraction, use the Python API.
```

### Python API

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=False,
    do_code_enrichment=False,
    do_formula_enrichment=False,
    do_picture_classification=False,
    do_picture_description=False,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert("/path/to/paper.pdf", page_range=(1, 9))
md = result.document.export_to_markdown()

with open("output.md", "w") as f:
    f.write(md)
```

### What You Get

- Embedded PDF text extracted via DoclingParse backend
- Basic reading order from layout clustering
- Paragraph/section structure from text positioning
- Fast processing (seconds for a typical paper)

### What You Don't Get

- No OCR (scanned content is skipped)
- No table structure recognition
- No math/formula extraction
- No image extraction or description
- Basic heading detection only

---

## Tier 2: Basic Layout (Good Structure, No OCR)

Runs the Heron layout model to detect page elements (headings, paragraphs, tables, figures, etc.) but skips OCR and enrichment. Good for clean digital PDFs where embedded text is reliable.

### CLI

```bash
docling paper.pdf \
    --to md \
    --no-ocr \
    --no-tables \
    --output ./output
```

### Python API

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, LayoutOptions, DOCLING_LAYOUT_HERON
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=False,
    do_code_enrichment=False,
    do_formula_enrichment=False,
    do_picture_classification=False,
    do_picture_description=False,
    layout_options=LayoutOptions(
        model_spec=DOCLING_LAYOUT_HERON,
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert("/path/to/paper.pdf", page_range=(1, 9))
md = result.document.export_to_markdown()

with open("output.md", "w") as f:
    f.write(md)
```

> **Note:** Layout model configs are module-level constants (e.g. `DOCLING_LAYOUT_HERON`), not enum members. Import them directly from `docling.datamodel.pipeline_options`.

### What This Adds Over Tier 1

- AI-based layout analysis (headings, sections, lists, figures, tables detected)
- Better reading order from layout model
- Element-type classification (distinguishes headings from body text)
- Page element bounding boxes

---

## Tier 3: Standard Pipeline — Default (Balanced)

The default Docling configuration. Runs Heron layout model + auto-selected OCR + fast table structure recognition. This is what you get out of the box.

### CLI

```bash
# Default — all standard features enabled
docling paper.pdf --to md --output ./output

# With page range (1-indexed)
# (No direct page-range CLI flag — use Python API for page selection)
```

### Python API

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()  # All defaults

result = converter.convert("/path/to/paper.pdf", page_range=(1, 9))
md = result.document.export_to_markdown()

with open("output.md", "w") as f:
    f.write(md)
```

### Default Settings

| Setting | Default Value |
|---------|---------------|
| Layout model | Heron (fast, balanced) |
| OCR | Auto (selects best available engine) |
| Table structure | Enabled, ACCURATE mode |
| Code enrichment | Disabled |
| Formula enrichment | Disabled |
| Picture description | Disabled |
| Image export | Placeholder |

### What This Adds Over Tier 2

- OCR for scanned/image-based content
- Table structure recognition (cell detection, row/column indexing)
- Proper markdown table output
- Better handling of mixed text + image PDFs

### What You Don't Get

- No math/LaTeX formula extraction
- No code block detection
- No image descriptions
- No figure content extraction

### Expected Performance

- **Speed:** ~2-10 seconds per page (CPU), faster with GPU
- **Output:** Structured markdown with headings, tables, lists
- **Accuracy:** Good for clean digital PDFs with standard layouts

---

## Tier 4: Enhanced Pipeline (High Quality)

Upgrades to a larger layout model (Egret), enables formula/code enrichment, and uses accurate table extraction. Best for complex academic papers.

### CLI

```bash
docling paper.pdf \
    --to md \
    --table-mode accurate \
    --enrich-code \
    --enrich-formula \
    --output ./output
```

### Python API

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    LayoutOptions,
    DOCLING_LAYOUT_EGRET_LARGE,
    TableStructureOptions,
    TableFormerMode,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions(
    # Layout: use larger, more accurate model
    layout_options=LayoutOptions(
        model_spec=DOCLING_LAYOUT_EGRET_LARGE,
    ),
    # OCR: enabled with auto engine
    do_ocr=True,
    # Tables: accurate mode
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.ACCURATE,
    ),
    # Enrichment: code + formulas
    do_code_enrichment=True,
    do_formula_enrichment=True,
    # Images
    generate_picture_images=True,
    # Acceleration
    accelerator_options=AcceleratorOptions(
        num_threads=4,
        device="auto",  # Uses MPS on Apple Silicon, CUDA on NVIDIA
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert("/path/to/paper.pdf", page_range=(1, 9))
md = result.document.export_to_markdown()

with open("output.md", "w") as f:
    f.write(md)
```

### Layout Model Options

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| `DOCLING_LAYOUT_HERON` (default) | Fast | Good | General use |
| `DOCLING_LAYOUT_HERON_101` | Fast | Good | Updated Heron |
| `DOCLING_LAYOUT_EGRET_MEDIUM` | Medium | Better | Production |
| `DOCLING_LAYOUT_EGRET_LARGE` | Slow | High | Complex docs |
| `DOCLING_LAYOUT_EGRET_XLARGE` | Very slow | Highest | Maximum accuracy |

### OCR Engine Options

| Engine | Speed | Languages | GPU | Best For |
|--------|-------|-----------|-----|----------|
| `OcrAutoOptions` (default) | Auto | Auto | Auto | Let Docling decide |
| `RapidOcrOptions` | Fast | EN, CN | ONNX | Lightweight, cross-platform |
| `EasyOcrOptions` | Medium | 80+ | CUDA | Multi-language |
| `TesseractOcrOptions` | Medium | 100+ | No | Well-established |
| `OcrMacOptions` | Fast | System | Apple Silicon | macOS native |

### Table Extraction Modes

| Mode | Speed | Accuracy | Best For |
|------|-------|----------|----------|
| `FAST` | Quick | Good | Volume processing |
| `ACCURATE` | Slower | Better | Complex tables, merged cells |

### What This Adds Over Tier 3

- More accurate layout detection (Egret vs Heron)
- Mathematical formula extraction (LaTeX)
- Code block recognition
- Higher-fidelity table extraction
- Image extraction with bounding boxes

### Expected Performance

- **Speed:** ~10-30 seconds per page (CPU), ~5-15 with GPU
- **Output:** Structured markdown with formulas, code blocks, tables
- **Accuracy:** High for complex scientific papers

---

## Tier 5: VLM Pipeline (Highest Quality)

Uses a Vision-Language Model to process entire pages at once, understanding the full visual layout rather than relying on individual model stages. This is the highest quality option but slowest.

### CLI

```bash
docling paper.pdf \
    --to md \
    --pipeline vlm \
    --vlm-model granite_docling \
    --output ./output
```

### Python API — Transformers Backend (CPU/CUDA)

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    AcceleratorOptions,
    smoldocling_vlm_conversion_options,
)
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline

pipeline_options = VlmPipelineOptions(
    vlm_options=smoldocling_vlm_conversion_options,  # SmolDocling 256M
    generate_page_images=True,
    accelerator_options=AcceleratorOptions(
        num_threads=4,
        device="cpu",  # or "cuda" for NVIDIA GPU
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert("/path/to/paper.pdf", page_range=(1, 9))
md = result.document.export_to_markdown()

with open("output.md", "w") as f:
    f.write(md)
```

### Python API — MLX Backend (Apple Silicon)

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    AcceleratorOptions,
    smoldocling_vlm_mlx_conversion_options,
)
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline

# Requires: pip install mlx mlx-lm mlx-vlm
pipeline_options = VlmPipelineOptions(
    vlm_options=smoldocling_vlm_mlx_conversion_options,  # SmolDocling 256M MLX
    generate_page_images=True,
    accelerator_options=AcceleratorOptions(
        num_threads=4,
        device="mps",
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert("/path/to/paper.pdf", page_range=(1, 9))
md = result.document.export_to_markdown()

with open("output.md", "w") as f:
    f.write(md)
```

> **Note:** The MLX backend requires compatible versions of `mlx-vlm`. If you get shape mismatch errors during model loading, try pinning `mlx-vlm` to an earlier version or use the transformers backend on CPU instead.

### Available VLM Models

Pre-built options importable from `docling.datamodel.pipeline_options`:

| Import Name | Model | Size | Backend | Device |
|-------------|-------|------|---------|--------|
| `smoldocling_vlm_conversion_options` | SmolDocling 256M | 256M | Transformers | CPU/CUDA |
| `smoldocling_vlm_mlx_conversion_options` | SmolDocling 256M MLX | 256M | MLX | MPS |
| `granite_vision_vlm_conversion_options` | Granite Vision | Large | Transformers | CPU/CUDA |

Additional models from `docling.datamodel.vlm_model_specs`:

| Constant | Model | Backend |
|----------|-------|---------|
| `GRANITEDOCLING_TRANSFORMERS` | Granite Docling 258M | Transformers |
| `GRANITEDOCLING_MLX` | Granite Docling 258M MLX | MLX |
| `GRANITEDOCLING_VLLM` | Granite Docling 258M | vLLM |

### What This Adds Over Tier 4

- Full page understanding via vision model
- Better handling of complex/irregular layouts
- Understands figures, charts, and diagrams contextually
- Can generate descriptions of visual content
- Handles edge cases that pipeline models miss

### Limitations

- Significantly slower (minutes per page on CPU, even with small 256M models)
- MLX backend may have version compatibility issues with `mlx-vlm`
- Quality depends on VLM model choice
- May hallucinate content not in the document
- Model downloads required on first run

### Expected Performance

- **Speed:** ~3-5 minutes per page (CPU), ~30s-1min with GPU/MLX
- **Output:** High-fidelity markdown with visual understanding
- **Model size:** ~256M-500MB for SmolDocling/Granite Docling

---

## Tier 4+: Enhanced with Image Descriptions

A variant of Tier 4 that adds VLM-based picture descriptions and classification. Useful when figure content matters.

### Python API

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    LayoutOptions,
    DOCLING_LAYOUT_EGRET_LARGE,
    TableStructureOptions,
    TableFormerMode,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions(
    layout_options=LayoutOptions(
        model_spec=DOCLING_LAYOUT_EGRET_LARGE,
    ),
    do_ocr=True,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.ACCURATE,
    ),
    do_code_enrichment=True,
    do_formula_enrichment=True,
    # Image features
    generate_picture_images=True,
    do_picture_classification=True,
    do_picture_description=True,
    # Acceleration
    accelerator_options=AcceleratorOptions(
        num_threads=4,
        device="auto",
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert("/path/to/paper.pdf", page_range=(1, 9))

# Export with referenced images
result.document.save_as_markdown(
    "output.md",
    image_mode="referenced",  # Saves images as separate PNGs
)
```

### Image Export Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `placeholder` | `<!-- image -->` markers only | Text-only output |
| `embedded` | Base64-encoded inline images | Single-file output |
| `referenced` | Separate PNG files with `![](path)` links | Full fidelity |

---

## CLI Complete Reference

```bash
docling [OPTIONS] SOURCES...

# Input/Output
--from {pdf,docx,html,md,...}          Input format filter
--to {md,json,html,yaml,text,doctags}  Output format(s) (repeatable)
--output <directory>                    Output directory (default: .)

# Pipeline Selection
--pipeline {standard,vlm,asr}          Processing pipeline
--vlm-model <preset>                   VLM model preset name

# OCR Control
--ocr / --no-ocr                       Enable/disable OCR (default: on)
--force-ocr                            Force full-page OCR (replace text)
--ocr-engine {auto,rapidocr,easyocr,   OCR engine selection
              tesseract,tesserocr,
              ocrmac}
--ocr-lang "en;fr;de"                  OCR languages (semicolon-separated)
--psm <0-13>                           Tesseract Page Segmentation Mode

# Table Processing
--tables / --no-tables                 Enable/disable table extraction
--table-mode {fast,accurate}           Table extraction accuracy

# Enrichment
--enrich-code                          Enable code block detection
--enrich-formula                       Enable math formula extraction
--enrich-picture-classes               Enable image classification
--enrich-picture-description           Enable image descriptions
--enrich-chart-extraction              Enable chart data extraction

# Hardware
--device {auto,cpu,cuda,mps,xpu}       Accelerator device
--num-threads <N>                       CPU thread count

# PDF Backend
--pdf-backend {dlparse_v4,dlparse_v2,  PDF parsing backend
               dlparse_v1,pypdfium}

# Image Output
--image-export-mode {placeholder,       Image handling in export
                     embedded,referenced}

# Processing Control
--document-timeout <seconds>            Timeout per document
--abort-on-error                        Stop on first error

# Debugging
--verbose / -v / -vv                    Logging verbosity
--profiling                             Show timing statistics
--save-profiling                        Save timings to JSON
--show-layout                           Show bounding boxes in HTML output

# Plugins
--enable-remote-services                Allow remote API calls
--allow-external-plugins                Enable third-party plugins
--show-external-plugins                 List available plugins

# Info
--version                               Show version
```

---

## Output Formats

Docling supports multiple export formats:

| Format | Flag | Description |
|--------|------|-------------|
| Markdown | `--to md` | Standard markdown with tables, headings, lists |
| JSON | `--to json` | Complete document structure with all metadata |
| YAML | `--to yaml` | Human-readable structured format |
| HTML | `--to html` | Web-ready HTML output |
| Text | `--to text` | Plain text, no formatting |
| DocTags | `--to doctags` | IBM's semantic annotation format |

### Python Export API

```python
result = converter.convert("paper.pdf")
doc = result.document

# Markdown
md_text = doc.export_to_markdown()

# Save with image handling
doc.save_as_markdown("output.md", image_mode="referenced")
doc.save_as_json("output.json", image_mode="embedded")
doc.save_as_html("output.html", image_mode="referenced", split_page_view=True)
doc.save_as_yaml("output.yaml")
```

---

## GPU / Acceleration

### Device Selection

| Device | Flag | Notes |
|--------|------|-------|
| Auto | `--device auto` | Best available (CUDA > MPS > CPU) |
| CPU | `--device cpu` | Always available, slowest |
| CUDA | `--device cuda` | NVIDIA GPUs |
| MPS | `--device mps` | Apple Silicon (M1/M2/M3/M4) |
| XPU | `--device xpu` | Intel Arc GPUs |

### Environment Variables

```bash
export DOCLING_DEVICE=mps           # Force device
export DOCLING_NUM_THREADS=8        # CPU threads
export OMP_NUM_THREADS=8            # Alternative thread control
export DOCLING_CUDA_USE_FLASH_ATTENTION2=1  # NVIDIA optimization
```

### Model-Device Compatibility

| Model | CPU | CUDA | MPS | XPU |
|-------|-----|------|-----|-----|
| Heron Layout | Yes | Yes | Yes | Yes |
| Egret Layout | Yes | Yes | Yes | Yes |
| TableFormer | Yes | Yes | No | Yes |
| RapidOCR | Yes | Yes | No | No |
| EasyOCR | Yes | Yes | Yes | No |
| VLMs (Transformers) | Yes | Yes | Yes | Yes |
| VLMs (MLX) | No | No | Yes | No |

---

## Batch Processing

### CLI

```bash
# Multiple files
docling paper1.pdf paper2.pdf paper3.pdf --to md --output ./output

# Directory of PDFs
docling ./papers/*.pdf --to md --output ./output
```

### Python API

```python
from docling.document_converter import DocumentConverter
from pathlib import Path

converter = DocumentConverter()

sources = list(Path("papers").glob("*.pdf"))
for result in converter.convert_all(sources):
    doc = result.document
    name = Path(result.input.file).stem
    doc.save_as_markdown(f"output/{name}.md")
```

### Concurrency Settings

```python
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions(
    # Batch sizes for threaded pipeline stages
    ocr_batch_size=4,
    layout_batch_size=4,
    table_batch_size=4,
    batch_polling_interval_seconds=0.5,
    queue_max_size=100,
    # Hardware
    accelerator_options=AcceleratorOptions(
        num_threads=8,
        device="auto",
    ),
)
```

---

## Comparison with Marker and MarkItDown

| Feature | Docling | Marker | MarkItDown |
|---------|---------|--------|------------|
| **Developer** | IBM Research | Datalab | Microsoft |
| **Approach** | Modular pipeline (layout + OCR + table + VLM) | Surya deep-learning models | pdfminer/pdfplumber text extraction |
| **GPU required** | No (but helps) | No (but much faster) | No |
| **Speed (10-page PDF)** | 20s - 5min (depends on tier) | 2-17 min | 1-5 seconds |
| **OCR capability** | Multiple engines (Rapid, Easy, Tesseract, macOS) | Built-in (Surya) | Only via Azure |
| **Math/LaTeX** | Yes (formula enrichment) | Built-in (equation recognition) | Only via Azure |
| **Image extraction** | Yes (with classification + description) | Yes (JPEG/PNG) | No (PDF) |
| **Table extraction** | TableFormer (accurate + fast modes) | ML-based (TableRecModel) | Basic (position clustering) |
| **VLM integration** | Full pipeline (Granite, SmolDocling, etc.) | Text correction, table fixing | Image captions only |
| **Layout models** | 5 sizes (Heron → Egret XL) | Surya layout detection | None |
| **Pipeline customization** | Highly modular (per-stage config) | Config file based | Minimal |
| **Export formats** | MD, JSON, YAML, HTML, Text, DocTags | MD + images | MD only |
| **Page range** | Yes (`page_range=(1, 9)`) | Yes (`--page_range "0-8"`) | No |
| **Local processing** | Yes (all tiers) | Yes (all tiers) | Yes (except Azure) |
| **Cost** | Free (all local tiers) | Free (except LLM tier) | Free (except Azure/LLM) |

### When to Use Docling Over Marker

- You need **maximum pipeline customization** (swap OCR engines, layout models, etc.)
- You want **multiple export formats** (JSON, YAML, HTML, not just Markdown)
- You need **VLM-based page understanding** (Granite Docling, SmolDocling)
- You want **image classification and description** built into the pipeline
- You're on **macOS** and want native Vision framework OCR
- You need **formula enrichment** without full OCR overhead

### When to Use Docling Over MarkItDown

- You need **any** AI-based processing (layout, OCR, tables, formulas)
- You need **image extraction or description**
- You need **structured table output**
- You need **math/formula rendering**
- You want **multiple output formats**
- You need **page-range selection**

### When to Use Marker or MarkItDown Instead

- **Marker:** When you need battle-tested Surya OCR, simple config, and proven arXiv paper handling
- **MarkItDown:** When speed is everything and you only need plain text from clean digital PDFs

---

## arXiv-Specific Tips

### Recommended: Tier 3 (Default) for Quick Parsing

```bash
docling paper.pdf --to md --output ./output
```

### Recommended: Tier 4 for Full Quality

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    LayoutOptions,
    DOCLING_LAYOUT_EGRET_LARGE,
    TableStructureOptions,
    TableFormerMode,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions(
    layout_options=LayoutOptions(
        model_spec=DOCLING_LAYOUT_EGRET_LARGE,
    ),
    do_ocr=True,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.ACCURATE,
    ),
    do_code_enrichment=True,
    do_formula_enrichment=True,
    generate_picture_images=True,
    accelerator_options=AcceleratorOptions(
        device="auto",
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert("paper.pdf", page_range=(1, 9))
result.document.save_as_markdown("output.md", image_mode="referenced")
```

### Limitations for arXiv Papers

1. **Two-column layouts** — layout model handles these well but some merging can occur
2. **Dense math** — formula enrichment helps but complex nested equations may still be imperfect
3. **Algorithm pseudocode** — detected as code blocks with enrichment enabled
4. **Footnotes/references** — may be merged with body text depending on layout detection
5. **Supplementary material** — long appendices may slow VLM processing significantly

### Page Range

Docling uses **1-indexed** page ranges (unlike Marker which is 0-indexed):

```python
# First 9 pages
result = converter.convert("paper.pdf", page_range=(1, 9))

# Pages 5-10
result = converter.convert("paper.pdf", page_range=(5, 10))
```

---

## Summary

Docling is the right tool when you need a **highly customizable, modular pipeline** with multiple quality tiers and export formats. Its stage-based architecture lets you mix and match layout models, OCR engines, table extractors, and VLMs to find the right speed/quality tradeoff.

For arXiv paper parsing specifically:
- **Quick text extraction?** → Tier 1-2 (seconds, basic structure)
- **Good tables + OCR?** → Tier 3 (default, balanced)
- **Complex math + code + figures?** → Tier 4 (enhanced pipeline)
- **Best possible fidelity?** → Tier 5 (VLM pipeline, slowest)
- **Processing hundreds of papers?** → Tier 3 with batch processing and GPU acceleration

---

## Tested Code & Benchmarks

The following code was tested on an **Apple M2 Pro (16GB unified RAM, macOS)** parsing the first 9 pages of an arXiv paper (`2512.24601v2.pdf`, 38 pages total, "Recursive Language Models").

### Benchmark Results

| Tier | Time | Output Size | Lines | Notes |
|------|------|-------------|-------|-------|
| 1 | 14.5s | 42,143 chars | 217 | Backend text only, no models |
| 2 | 2.8s | 42,143 chars | 217 | Heron layout (models cached from Tier 1) |
| 3 | 42.4s | 45,363 chars | 356 | Heron + OCR + fast tables |
| 4 | 52.4s | 46,440 chars | 412 | Egret Large + formulas + code + accurate tables |
| 5 | 19m 41s | 40,121 chars | 292 | SmolDocling 256M on CPU (MLX had version issues) |

> **Note:** Tier 1 was slower than Tier 2 because Tier 1 included model download/caching time. On subsequent runs Tier 1 would be faster. Tier 5 took ~20 minutes on CPU; with a working MLX setup or CUDA GPU it would be significantly faster.

### Tier 1: Backend Text Only (Tested)

```python
import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

paper = "/path/to/paper.pdf"
output = "/path/to/output/docling_tier1/paper.md"

pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=False,
    do_code_enrichment=False,
    do_formula_enrichment=False,
    do_picture_classification=False,
    do_picture_description=False,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

start = time.time()
result = converter.convert(paper, page_range=(1, 9))
elapsed = time.time() - start

md = result.document.export_to_markdown()
with open(output, "w") as f:
    f.write(md)

print(f"Tier 1 done in {elapsed:.1f}s")
print(f"Output: {len(md)} chars, {md.count(chr(10))} lines")
```

### Tier 2: Heron Layout, No OCR (Tested)

```python
import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, LayoutOptions, DOCLING_LAYOUT_HERON
from docling.datamodel.base_models import InputFormat

paper = "/path/to/paper.pdf"
output = "/path/to/output/docling_tier2/paper.md"

pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=False,
    do_code_enrichment=False,
    do_formula_enrichment=False,
    do_picture_classification=False,
    do_picture_description=False,
    layout_options=LayoutOptions(
        model_spec=DOCLING_LAYOUT_HERON,
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

start = time.time()
result = converter.convert(paper, page_range=(1, 9))
elapsed = time.time() - start

md = result.document.export_to_markdown()
with open(output, "w") as f:
    f.write(md)

print(f"Tier 2 done in {elapsed:.1f}s")
print(f"Output: {len(md)} chars, {md.count(chr(10))} lines")
```

### Tier 3: Default Pipeline with Fast Tables (Tested)

```python
import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat

paper = "/path/to/paper.pdf"
output = "/path/to/output/docling_tier3/paper.md"

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.FAST,
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

start = time.time()
result = converter.convert(paper, page_range=(1, 9))
elapsed = time.time() - start

md = result.document.export_to_markdown()
with open(output, "w") as f:
    f.write(md)

print(f"Tier 3 done in {elapsed:.1f}s")
print(f"Output: {len(md)} chars, {md.count(chr(10))} lines")
```

### Tier 4: Enhanced Pipeline (Tested)

```python
import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    LayoutOptions,
    DOCLING_LAYOUT_EGRET_LARGE,
    TableStructureOptions,
    TableFormerMode,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat

paper = "/path/to/paper.pdf"
output = "/path/to/output/docling_tier4/paper.md"

pipeline_options = PdfPipelineOptions(
    layout_options=LayoutOptions(
        model_spec=DOCLING_LAYOUT_EGRET_LARGE,
    ),
    do_ocr=True,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.ACCURATE,
    ),
    do_code_enrichment=True,
    do_formula_enrichment=True,
    generate_picture_images=True,
    accelerator_options=AcceleratorOptions(
        num_threads=4,
        device="auto",
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

start = time.time()
result = converter.convert(paper, page_range=(1, 9))
elapsed = time.time() - start

md = result.document.export_to_markdown()
with open(output, "w") as f:
    f.write(md)

print(f"Tier 4 done in {elapsed:.1f}s")
print(f"Output: {len(md)} chars, {md.count(chr(10))} lines")
```

### Tier 5: VLM Pipeline (Tested — Slow on CPU)

```python
import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    AcceleratorOptions,
    smoldocling_vlm_conversion_options,
)
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline

paper = "/path/to/paper.pdf"
output = "/path/to/output/docling_tier5/paper.md"

pipeline_options = VlmPipelineOptions(
    vlm_options=smoldocling_vlm_conversion_options,
    generate_page_images=True,
    accelerator_options=AcceleratorOptions(
        num_threads=4,
        device="cpu",
    ),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

start = time.time()
result = converter.convert(paper, page_range=(1, 9))
elapsed = time.time() - start

md = result.document.export_to_markdown()
with open(output, "w") as f:
    f.write(md)

print(f"Tier 5 done in {elapsed:.1f}s ({elapsed/60:.1f}m)")
print(f"Output: {len(md)} chars, {md.count(chr(10))} lines")
```

> **Warning:** On Apple M2 Pro (CPU mode), Tier 5 took ~20 minutes for 9 pages (~2.2 min/page). The MLX backend (`smoldocling_vlm_mlx_conversion_options`) should be much faster on Apple Silicon but had `mlx-vlm` version compatibility issues at time of testing. On NVIDIA GPU with CUDA, expect ~30s-1min per page.

### Known Issues (as of v2.72.0)

1. **Layout model constants** — Use `DOCLING_LAYOUT_HERON`, `DOCLING_LAYOUT_EGRET_LARGE`, etc. imported directly from `docling.datamodel.pipeline_options`. They are **not** enum members on `LayoutModelConfig`.
2. **MLX VLM weight mismatch** — `mlx-vlm >= 0.3.x` may produce `ValueError: Expected shape` errors when loading Granite Docling or SmolDocling MLX models. Downgrade `mlx-vlm` or use the transformers backend as a workaround.
3. **transformers 5.x breaking change** — Installing `mlx-lm` may upgrade `transformers` to 5.x, which removes `AutoModelForVision2Seq`. Pin `transformers<5` for Docling compatibility: `pip install 'transformers>=4.40,<5'`.
