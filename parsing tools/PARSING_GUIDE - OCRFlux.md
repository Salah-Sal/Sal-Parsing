# OCRFlux Parsing Guide

> Comprehensive guide to all parsing options for scientific paper (arXiv PDF → Markdown) conversion using OCRFlux.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture — Three-Stage Pipeline](#2-architecture--three-stage-pipeline)
3. [Installation](#3-installation)
4. [Deployment Modes](#4-deployment-modes)
5. [Quality-Affecting Parameters](#5-quality-affecting-parameters)
6. [Parsing Tiers — Lowest to Highest Quality](#6-parsing-tiers--lowest-to-highest-quality)
7. [CLI Reference](#7-cli-reference)
8. [Output Format](#8-output-format)
9. [Evaluation & Benchmarks](#9-evaluation--benchmarks)
10. [Docker Deployment](#10-docker-deployment)
11. [Known Limitations](#11-known-limitations)

---

## 1. Overview

**OCRFlux** is a 3B-parameter Vision-Language Model (VLM) for converting PDFs and images to Markdown, developed by the **ChatDOC** team. It is based on the Qwen2.5-VL architecture.

| Property | Value |
|---|---|
| Model | `ChatDOC/OCRFlux-3B` |
| Parameters | 3 billion |
| Architecture | Qwen2.5-VL chat template |
| License | Apache 2.0 |
| Languages | English, Chinese |
| HuggingFace | https://huggingface.co/ChatDOC/OCRFlux-3B |
| Demo | https://ocrflux.pdfparser.io/ |

**Key differentiator**: OCRFlux is the only open-source tool with native **cross-page table and paragraph merging** — it can detect and reassemble tables/paragraphs that span page boundaries.

---

## 2. Architecture — Three-Stage Pipeline

Unlike single-pass OCR tools, OCRFlux uses a **three-stage pipeline**:

| Stage | Task | Prompt | Skippable? |
|---|---|---|---|
| **1** | Page-to-Markdown | Renders each page as an image, sends to VLM with OCR prompt | No |
| **2** | Cross-Page Element Merge Detection | Identifies paragraph/table pairs that should merge across consecutive pages | Yes (`--skip_cross_page_merge`) |
| **3** | HTML Table Merge | Merges split table fragments into single coherent HTML tables | Yes (`--skip_cross_page_merge`) |

### Stage 1: Page-to-Markdown

Each PDF page is rendered to a PNG image (via `pdftoppm`) and sent to the model with this prompt:

> "Below is the image of one page of a document. Just return the plain text representation of this document as if you were reading it naturally. ALL tables should be presented in HTML format. If there are images or figures in the page, present them as `<Image>(left,top),(right,bottom)</Image>`. Present all titles and headings as H1 headings. Do not hallucinate."

The model returns a structured JSON response:

```python
@dataclass(frozen=True)
class PageResponse:
    primary_language: Optional[str]       # e.g., "English", "Chinese"
    is_rotation_valid: bool               # Page orientation correctness
    rotation_correction: int              # 0, 90, 180, or 270 degrees
    is_table: bool                        # Page contains tables
    is_diagram: bool                      # Page contains diagrams
    natural_text: Optional[str]           # The actual markdown content
```

### Stage 2: Element Merge Detection

For each pair of consecutive pages, the model receives both pages' numbered elements and returns merge pairs:

```
Input:  Page N elements [0, 1, 2, ...] + Page N+1 elements [0, 1, 2, ...]
Output: [(elem_idx_page_N, elem_idx_page_N+1), ...] or []
```

### Stage 3: HTML Table Merge

For detected table pairs, the model merges two HTML fragments into one coherent table, handling header repetition and cell splitting.

---

## 3. Installation

### System Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install \
    poppler-utils poppler-data \
    ttf-mscorefonts-installer msttcorefonts \
    fonts-crosextra-caladea fonts-crosextra-carlito \
    gsfonts lcdf-typetools
```

### Python Environment

```bash
conda create -n ocrflux python=3.11
conda activate ocrflux
git clone https://github.com/chatdoc-com/OCRFlux.git
cd OCRFlux
pip install -e . --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
```

### Key Dependencies

| Package | Version | Notes |
|---|---|---|
| `torch` | >= 2.5.1 | CUDA required |
| `vllm` | 0.7.3 (pinned) | Inference engine |
| `transformers` | 4.50.0 (pinned) | HuggingFace |
| `pypdf` | >= 5.2.0 | PDF page counting |
| `pypdfium2` | — | PDF rendering |
| `Pillow` | — | Image processing |
| `poppler-utils` | system | `pdftoppm` for PDF→PNG |

---

## 4. Deployment Modes

OCRFlux offers three ways to run inference:

### Mode A: Offline Inference (Direct vLLM)

Load the model directly in Python. Best for quick single-file parsing.

```python
from vllm import LLM
from ocrflux.inference import parse

llm = LLM(
    model="ChatDOC/OCRFlux-3B",
    gpu_memory_utilization=0.8,
    max_model_len=8192
)

result = parse(
    llm,
    file_path="paper.pdf",
    skip_cross_page_merge=False,
    max_page_retries=4
)

if result:
    print(result['document_text'])  # Full merged markdown
```

### Mode B: Batch Pipeline (vLLM Server)

Designed for large-scale processing. Automatically starts a vLLM server, distributes work across async workers.

```bash
# Run pipeline (auto-starts vLLM server)
python -m ocrflux.pipeline ./workspace \
    --data paper.pdf \
    --model ChatDOC/OCRFlux-3B

# Convert JSONL results to markdown files
python -m ocrflux.jsonl_to_markdown ./workspace
```

Output: `./workspace/markdowns/{filename}/{filename}.md`

### Mode C: Client-Server (Remote)

Start a vLLM server, then send requests from a client.

```bash
# Start server
bash ocrflux/server.sh ChatDOC/OCRFlux-3B 30024
```

```python
# Client
import asyncio
from argparse import Namespace
from ocrflux.client import request

args = Namespace(
    model="ChatDOC/OCRFlux-3B",
    skip_cross_page_merge=False,
    max_page_retries=1,
    url="http://localhost",
    port=30024,
)
result = asyncio.run(request(args, "paper.pdf"))
```

---

## 5. Quality-Affecting Parameters

These parameters control the quality/speed tradeoff:

### 5.1 Image Resolution (`--target_longest_image_dim`)

Controls the pixel size of PDF page renders sent to the model.

| Value | Effect | Use Case |
|---|---|---|
| 512 | Lowest resolution, fastest | Quick draft, text-heavy papers |
| 768 | Low-medium resolution | Readable text, blurry equations |
| **1024** (default) | Balanced | Good for most papers |
| 1536 | High resolution | Complex equations, small text |
| 2048 | Maximum resolution | Dense tables, fine details |

Higher values = more input tokens = slower inference + better detail capture.

### 5.2 Cross-Page Merge (`--skip_cross_page_merge`)

| Setting | Effect |
|---|---|
| Enabled (default) | Runs all 3 stages — detects and merges split tables/paragraphs |
| `--skip_cross_page_merge` | Skips stages 2 & 3 — simply concatenates per-page results |

Disabling is significantly faster but loses cross-page context. For arXiv papers, tables rarely span pages, so skipping may be acceptable.

### 5.3 Retry Count (`--max_page_retries`)

Failed pages are retried with progressively increasing temperature:

```
Attempt:     0    1    2    3    4    5    6    7    8
Temperature: 0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8
```

| Value | Effect |
|---|---|
| 0 | No retries — fastest, but failed pages are lost |
| 4 | Moderate retries |
| **8** (default) | Maximum retries — most robust |

### 5.4 Model Precision (`--dtype`)

| Value | VRAM Usage | Speed | Quality |
|---|---|---|---|
| `float16` / `half` | Low | Fast | May have numerical issues |
| **`auto`** (default) | Medium | Balanced | vLLM auto-detects (usually bf16) |
| `bfloat16` | Medium | Balanced | Best for A100/H100 |
| `float32` / `float` | High (2x) | Slow | Maximum precision |

### 5.5 Context Length (`--model_max_context`)

| Value | Effect |
|---|---|
| 8192 | Faster, sufficient for most pages |
| **16384** (default) | Handles dense pages with large tables |

### 5.6 GPU Memory (`--gpu_memory_utilization`)

| Value | Effect |
|---|---|
| 0.5 | Conservative — leaves room for other processes |
| **0.8** (default) | Balanced |
| 0.9–0.95 | Aggressive — use with 24GB+ GPUs |

### 5.7 Error Rate Threshold (`--max_page_error_rate`)

Default: `0.004` (1 page per 250). If more pages fail than this threshold, the entire document is discarded. Lower = stricter quality control.

---

## 6. Parsing Tiers — Lowest to Highest Quality

### Tier 1: Draft (Fastest)

Skip cross-page merge, low resolution, no retries, FP16 precision.

```bash
python -m ocrflux.pipeline ./workspace \
    --data paper.pdf \
    --model ChatDOC/OCRFlux-3B \
    --skip_cross_page_merge \
    --target_longest_image_dim 512 \
    --max_page_retries 0 \
    --dtype half \
    --model_max_context 8192
```

**Or with Python:**
```python
llm = LLM(model="ChatDOC/OCRFlux-3B", gpu_memory_utilization=0.8, max_model_len=8192, dtype="half")
result = parse(llm, "paper.pdf", skip_cross_page_merge=True, max_page_retries=0)
```

| Aspect | Value |
|---|---|
| Resolution | 512px |
| Cross-page merge | Disabled |
| Retries | 0 |
| Precision | float16 |
| Context | 8192 |
| Best for | Quick text extraction, drafts |

### Tier 2: Basic

Skip cross-page merge, default resolution, minimal retries.

```bash
python -m ocrflux.pipeline ./workspace \
    --data paper.pdf \
    --model ChatDOC/OCRFlux-3B \
    --skip_cross_page_merge \
    --target_longest_image_dim 1024 \
    --max_page_retries 2
```

**Or with Python:**
```python
llm = LLM(model="ChatDOC/OCRFlux-3B", gpu_memory_utilization=0.8, max_model_len=8192)
result = parse(llm, "paper.pdf", skip_cross_page_merge=True, max_page_retries=2)
```

| Aspect | Value |
|---|---|
| Resolution | 1024px |
| Cross-page merge | Disabled |
| Retries | 2 |
| Precision | auto |
| Context | 8192 |
| Best for | Text-heavy papers without cross-page tables |

### Tier 3: Balanced (Default)

Cross-page merge enabled, default resolution, moderate retries.

```bash
python -m ocrflux.pipeline ./workspace \
    --data paper.pdf \
    --model ChatDOC/OCRFlux-3B \
    --target_longest_image_dim 1024 \
    --max_page_retries 4
```

**Or with Python:**
```python
llm = LLM(model="ChatDOC/OCRFlux-3B", gpu_memory_utilization=0.8, max_model_len=8192)
result = parse(llm, "paper.pdf", skip_cross_page_merge=False, max_page_retries=4)
```

| Aspect | Value |
|---|---|
| Resolution | 1024px |
| Cross-page merge | Enabled |
| Retries | 4 |
| Precision | auto |
| Context | 8192 |
| Best for | Most arXiv papers |

### Tier 4: High Quality

Higher resolution, full retries, extended context.

```bash
python -m ocrflux.pipeline ./workspace \
    --data paper.pdf \
    --model ChatDOC/OCRFlux-3B \
    --target_longest_image_dim 1536 \
    --max_page_retries 8 \
    --model_max_context 16384
```

**Or with Python:**
```python
llm = LLM(model="ChatDOC/OCRFlux-3B", gpu_memory_utilization=0.8, max_model_len=16384)
result = parse(llm, "paper.pdf", skip_cross_page_merge=False, max_page_retries=8)
```

| Aspect | Value |
|---|---|
| Resolution | 1536px |
| Cross-page merge | Enabled |
| Retries | 8 |
| Precision | auto |
| Context | 16384 |
| Best for | Papers with complex equations, dense tables |

### Tier 5: Maximum Quality

Maximum resolution, full retries, extended context, FP32 precision.

```bash
python -m ocrflux.pipeline ./workspace \
    --data paper.pdf \
    --model ChatDOC/OCRFlux-3B \
    --target_longest_image_dim 2048 \
    --max_page_retries 8 \
    --model_max_context 16384 \
    --dtype float32 \
    --gpu_memory_utilization 0.9
```

**Or with Python:**
```python
llm = LLM(model="ChatDOC/OCRFlux-3B", gpu_memory_utilization=0.9, max_model_len=16384, dtype="float32")
result = parse(llm, "paper.pdf", skip_cross_page_merge=False, max_page_retries=8)
```

| Aspect | Value |
|---|---|
| Resolution | 2048px |
| Cross-page merge | Enabled |
| Retries | 8 |
| Precision | float32 |
| Context | 16384 |
| GPU Memory | 0.9 |
| Best for | Maximum fidelity — complex layouts, small text, dense tables |

### Tier Summary

| Tier | Resolution | Cross-Page | Retries | Precision | Context | Speed | Quality |
|---|---|---|---|---|---|---|---|
| **1 — Draft** | 512 | No | 0 | fp16 | 8192 | Fastest | Lowest |
| **2 — Basic** | 1024 | No | 2 | auto | 8192 | Fast | Low |
| **3 — Balanced** | 1024 | Yes | 4 | auto | 8192 | Medium | Good |
| **4 — High** | 1536 | Yes | 8 | auto | 16384 | Slow | High |
| **5 — Maximum** | 2048 | Yes | 8 | fp32 | 16384 | Slowest | Highest |

---

## 7. CLI Reference

### Pipeline (Batch Processing)

```bash
python -m ocrflux.pipeline <workspace> [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `workspace` | (required) | Directory for results and intermediate files |
| `--task` | `pdf2markdown` | Task type: `pdf2markdown`, `merge_pages`, `merge_tables` |
| `--data` | — | Input file paths (PDFs, images, or JSON) |
| `--model` | `ChatDOC/OCRFlux-3B` | HuggingFace model ID or local path |
| `--target_longest_image_dim` | 1024 | Image render size (longest side, pixels) |
| `--max_page_retries` | 8 | Max retry attempts per page |
| `--max_page_error_rate` | 0.004 | Max acceptable page failure rate |
| `--skip_cross_page_merge` | False | Skip stages 2 & 3 |
| `--model_max_context` | 16384 | Max context length (tokens) |
| `--model_chat_template` | `qwen2-vl` | vLLM chat template |
| `--gpu_memory_utilization` | 0.8 | Fraction of GPU VRAM to use |
| `--tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism |
| `--dtype` | `auto` | Model precision: auto, half, float16, bfloat16, float32 |
| `--workers` | 8 | Concurrent worker processes |
| `--pages_per_group` | 500 | Target pages per work batch |
| `--port` | 40078 | vLLM server port |

### JSONL to Markdown

```bash
python -m ocrflux.jsonl_to_markdown <workspace> [--show_page_result]
```

| Argument | Description |
|---|---|
| `workspace` | Same workspace used in pipeline |
| `--show_page_result` | Also write per-page markdown files |

### Server Launch

```bash
bash ocrflux/server.sh <model_path> <port>
# Example:
bash ocrflux/server.sh ChatDOC/OCRFlux-3B 30024
```

Internally runs:
```bash
vllm serve <model_path> --port <port> --max-model-len 8192 --gpu_memory_utilization 0.8
```

---

## 8. Output Format

### JSONL (Pipeline Output)

Each processed document produces one JSON line in `workspace/results/output_{hash}.jsonl`:

```json
{
    "orig_path": "/path/to/paper.pdf",
    "num_pages": 19,
    "document_text": "# Title\n\nAbstract text...\n\n## Section 1\n\n...",
    "page_texts": {
        "0": "# Title\n\nAbstract...",
        "1": "## Section 1\n\nContent..."
    },
    "fallback_pages": [5, 12]
}
```

| Field | Description |
|---|---|
| `orig_path` | Original input file path |
| `num_pages` | Total page count |
| `document_text` | Final merged markdown (all 3 stages applied) |
| `page_texts` | Per-page markdown (0-indexed string keys) |
| `fallback_pages` | Pages that failed all retries (0-indexed) |

### Markdown (After Conversion)

Location: `workspace/markdowns/{filename}/{filename}.md`

**Formatting conventions:**
- All headings rendered as `#` (H1)
- Tables rendered as HTML `<table>` blocks
- Images as `<Image>(left,top),(right,bottom)</Image>` placeholders (stripped in final output)
- Paragraphs separated by `\n\n`

### Post-Processing Details

- Image placeholders (`<Image>...</Image>`) are stripped from final markdown
- Cross-page paragraph merges: space added unless previous char is `-` or CJK character
- Table format: Internal matrix format (`<l>`, `<t>`, `<lt>` markers) converted to standard HTML with `colspan`/`rowspan`

---

## 9. Evaluation & Benchmarks

### Published Benchmarks

#### Single-Page Parsing (Edit Distance Similarity)

| Model | English | Chinese | Overall |
|---|---|---|---|
| **OCRFlux-3B** | **0.971** | **0.962** | **0.967** |
| olmOCR-7B | 0.885 | 0.859 | 0.872 |
| Nanonets-OCR-s | 0.870 | 0.846 | 0.858 |
| MonkeyOCR | 0.828 | 0.731 | 0.780 |

#### Table Parsing (TEDS Score)

| Model | Simple | Complex | Overall |
|---|---|---|---|
| **OCRFlux-3B** | **0.912** | **0.807** | **0.861** |

#### Cross-Page Element Detection

| Language | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| English | 0.992 | 0.964 | 0.978 | 0.978 |
| Chinese | 1.000 | 0.988 | 0.994 | 0.994 |

#### Cross-Page Table Merging (TEDS Score)

| Type | Score |
|---|---|
| Simple | 0.965 |
| Complex | 0.935 |
| Overall | 0.950 |

### Running Evaluation Locally

```bash
# Page-to-Markdown evaluation
python -m eval.eval_page_to_markdown ./workspace --gt_file data.jsonl --n_jobs 40

# Table-to-HTML evaluation
python -m eval.eval_table_to_html ./workspace --gt_file data.jsonl --n_jobs 40
```

### Evaluation Datasets

| Dataset | URL |
|---|---|
| OCRFlux-bench-single | https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-single |
| OCRFlux-pubtabnet-single | https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-single |
| OCRFlux-bench-cross | https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-cross |
| OCRFlux-pubtabnet-cross | https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-cross |

---

## 10. Docker Deployment

### Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- Pre-downloaded OCRFlux-3B model

### Quick Start

```bash
docker run -it --gpus all \
    -v /path/to/workspace:/workspace \
    -v /path/to/pdfs:/pdfs \
    -v /path/to/OCRFlux-3B:/OCRFlux-3B \
    --entrypoint bash \
    chatdoc/ocrflux:latest

# Inside container:
python3.12 -m ocrflux.pipeline /workspace \
    --data /pdfs/*.pdf \
    --model /OCRFlux-3B
```

### Docker Image

- **Base**: Ubuntu 24.04
- **Python**: 3.12
- **Registry**: `chatdoc/ocrflux:latest`
- **Entry point**: `python3.12 -m ocrflux.pipeline`

---

## 11. Known Limitations

1. **CUDA-only**: Requires NVIDIA GPU. No MPS (Apple Silicon), CPU, or AMD ROCm support. The `vllm` dependency is CUDA-exclusive.
2. **All headings are H1**: The model prompt instructs "Present all titles and headings as H1 headings" — no heading hierarchy (H1/H2/H3) is preserved.
3. **Tables as HTML**: All tables are output as HTML, not Markdown tables. This is by design for complex table support but may not suit all downstream consumers.
4. **Image placeholders only**: Figures are detected with bounding boxes (`<Image>(left,top),(right,bottom)</Image>`) but not extracted as actual image files.
5. **Fixed DPI**: PDF pages are rendered at 72 DPI via `pdftoppm` before resizing to `target_longest_image_dim`. The base DPI is not configurable.
6. **No page range selection**: The pipeline processes all pages of a PDF. There is no built-in `--pages` flag to select a subset.
7. **Linux-first**: System dependencies (`poppler-utils`, font packages) are Ubuntu/Debian-oriented. macOS would need Homebrew equivalents.
8. **Pinned dependencies**: `vllm==0.7.3` and `transformers==4.50.0` are strictly pinned, which may conflict with other tools in the same environment.
9. **Max tokens**: Hard-coded at 8192 tokens per page response. Very dense pages may be truncated.
10. **Anti-hallucination**: The prompt includes "Do not hallucinate" but VLMs can still produce artifacts, especially at higher temperatures (retries).
