# DeepSeek-OCR Parsing Guide

## 1. Overview

**DeepSeek-OCR** (by DeepSeek AI) is a Vision-Language Model for document OCR that investigates the role of vision encoders from an LLM-centric viewpoint. It uses a **dual-encoder** architecture combining SAM-ViT-B and CLIP-ViT-L for high-quality optical character recognition with layout detection.

- **Paper**: [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)
- **Model**: `deepseek-ai/DeepSeek-OCR` (HuggingFace)
- **Successor**: [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) (released 2026/01/27)

| Property | Value |
|---|---|
| Vision Encoders | SAM-ViT-B (768-dim, 12 layers) + CLIP-ViT-L (1024-dim, 24 layers) |
| Projector | Linear (2048 → 1280) |
| Language Model | DeepSeek V2/V3 (MoE) |
| Max Context | 8192 tokens |
| Throughput | ~2500 tokens/s (A100-40G, vLLM, Gundam mode) |
| GPU Requirement | NVIDIA CUDA (flash-attn required) |

---

## 2. Architecture

```
PDF/Image → [Page Rendering (144 DPI)] → [Resize/Crop to Resolution Mode]
                                              ↓
                              ┌────────────────┴────────────────┐
                              │                                 │
                        SAM-ViT-B                         CLIP-ViT-L
                     (768-dim patches)              (1024-dim + CLS token)
                              │                                 │
                              └────────────────┬────────────────┘
                                               │
                                    Concatenate (2048-dim)
                                               │
                                    Linear Projector (→1280)
                                               │
                                    Global + Local Views
                                    (with newline/separator tokens)
                                               │
                                    DeepSeek V2/V3 LLM
                                               │
                                    Markdown + Layout Tags
```

**Key Design Points:**
- SAM encoder provides high-resolution spatial features (patch embeddings → conv downsampling 256→512→1024)
- CLIP encoder provides semantic features (class token + patch features)
- Features are concatenated along the channel dimension and projected
- Global view always processed; local crops only in Gundam (dynamic resolution) mode
- N-gram no-repeat logits processor prevents repetitive output

---

## 3. Installation

### Environment
```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

### Dependencies
```bash
# CUDA 11.8 example (download vllm whl from GitHub releases)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

### requirements.txt
```
transformers==4.46.3
tokenizers==0.20.3
PyMuPDF
img2pdf
einops
easydict
addict
Pillow
numpy
```

> **Note**: The `transformers>=4.51.1` error from vLLM can be ignored — both vLLM and Transformers backends work in the same environment.

---

## 4. Resolution Modes

DeepSeek-OCR supports 5 resolution modes that directly control quality and speed. Each mode determines how many **vision tokens** the encoders produce.

| Mode | base_size | image_size | crop_mode | Vision Tokens | Description |
|---|---|---|---|---|---|
| **Tiny** | 512 | 512 | False | 64 | Fastest, lowest quality |
| **Small** | 640 | 640 | False | 100 | Good for simple text |
| **Base** | 1024 | 1024 | False | 256 | Balanced quality/speed |
| **Large** | 1280 | 1280 | False | 400 | High quality, single view |
| **Gundam** | 1024 | 640 | True | Variable (n×100 + 256) | Dynamic resolution, best quality |

**How Resolution Works:**
- `base_size`: The global view is padded to this size (determines the "overview" resolution)
- `image_size`: For Gundam mode, each crop tile is this size
- `crop_mode=True` (Gundam): The image is split into multiple 640×640 tiles based on aspect ratio, plus one 1024×1024 global view
- Vision tokens per view = `ceil((size / patch_size) / downsample_ratio)²` where `patch_size=16`, `downsample_ratio=4`

**Gundam Mode Details:**
- Image is split into `n` tiles of `image_size × image_size` where `n` ranges from `MIN_CROPS` (2) to `MAX_CROPS` (default 6, max 9)
- Tile layout is chosen by finding the closest aspect ratio match
- Small images (≤640×640) skip cropping even in Gundam mode
- More crops = more tokens = better quality but slower and more VRAM

---

## 5. Prompt Modes

The prompt determines what kind of output the model produces.

| Prompt | Use Case | Layout Detection | Output Format |
|---|---|---|---|
| `<image>\n<|grounding\|>Convert the document to markdown.` | **Documents** (recommended for papers) | Yes | Markdown with `<\|ref\|>...<\|det\|>` tags |
| `<image>\nFree OCR.` | Plain text extraction | No | Raw text/markdown without layout |
| `<image>\n<|grounding\|>OCR this image.` | Non-document images | Yes | Markdown with layout tags |
| `<image>\nParse the figure.` | Figures/charts in documents | No | Structured figure description |
| `<image>\nDescribe this image in detail.` | General image captioning | No | Natural language description |
| `<image>\nLocate <\|ref\|>xxxx<\|/ref\|> in the image.` | Object localization | Yes | Coordinates |

**For arXiv papers, use**: `<image>\n<|grounding|>Convert the document to markdown.`

---

## 6. Backends

### 6.1 HuggingFace/Transformers Backend

Single-image inference with streaming-like output. Simpler setup.

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'path/to/image.jpg'
output_path = 'path/to/output/'

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,       # Resolution mode
    image_size=640,       # Crop tile size (Gundam)
    crop_mode=True,       # Enable dynamic resolution
    save_results=True,
    test_compress=True
)
```

**HF `model.infer()` Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `base_size` | int | Global view resolution (512/640/1024/1280) |
| `image_size` | int | Crop tile size (matches base_size for fixed modes, 640 for Gundam) |
| `crop_mode` | bool | Enable dynamic resolution cropping |
| `save_results` | bool | Save output files to `output_path` |
| `test_compress` | bool | Enable compression testing |

### 6.2 vLLM Backend

High-throughput batch processing. Requires editing `config.py` to set parameters.

**Three vLLM run scripts:**

| Script | Input | Use Case |
|---|---|---|
| `run_dpsk_ocr_image.py` | Single image | Streaming output, async |
| `run_dpsk_ocr_pdf.py` | PDF file | Batch all pages concurrently |
| `run_dpsk_ocr_eval_batch.py` | Image directory | Batch evaluation |

### 6.3 Upstream vLLM (Native Support)

Since Oct 2025, DeepSeek-OCR is natively supported in upstream vLLM:

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

image = Image.open("page.png").convert("RGB")
prompt = "<image>\nFree OCR."

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # <td>, </td>
    ),
    skip_special_tokens=False,
)

outputs = llm.generate(
    [{"prompt": prompt, "multi_modal_data": {"image": image}}],
    sampling_params
)
print(outputs[0].outputs[0].text)
```

---

## 7. Quality-Affecting Parameters

### 7.1 Resolution Parameters (config.py)

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `BASE_SIZE` | 1024 | 512/640/1024/1280 | Global view resolution |
| `IMAGE_SIZE` | 640 | 512/640/1024/1280 | Crop tile size |
| `CROP_MODE` | True | True/False | Enable dynamic resolution |
| `MIN_CROPS` | 2 | 1-9 | Minimum tiles in Gundam mode |
| `MAX_CROPS` | 6 | 1-9 | Maximum tiles in Gundam mode |

### 7.2 Inference Parameters

| Parameter | Default (PDF) | Default (Image) | Effect |
|---|---|---|---|
| `temperature` | 0.0 | 0.0 | Greedy decoding (deterministic) |
| `max_tokens` | 8192 | 8192 | Maximum output tokens per page |
| `max_model_len` | 8192 | 8192 | vLLM KV-cache context length |
| `gpu_memory_utilization` | 0.9 | 0.75 | GPU memory for KV cache |
| `MAX_CONCURRENCY` | 100 | N/A | Max concurrent sequences (vLLM) |
| `NUM_WORKERS` | 64 | N/A | Image preprocessing threads |

### 7.3 N-Gram No-Repeat Processor

Prevents repetitive output by banning n-grams that already appeared in a sliding window.

| Parameter | PDF Script | Image Script | Eval Script | Effect |
|---|---|---|---|---|
| `ngram_size` | 20 | 30 | 40 | Size of n-gram to check (larger = more permissive) |
| `window_size` | 50 | 90 | 90 | Lookback window (larger = stricter) |
| `whitelist_token_ids` | {128821, 128822} | {128821, 128822} | {128821, 128822} | Tokens exempt from ban (`<td>`, `</td>`) |

> **Trade-off**: Smaller `ngram_size` = more aggressive de-duplication but may break legitimate repetitions. The PDF script uses more aggressive settings (20/50) for throughput; image/eval use relaxed settings (30-40/90) for quality.

### 7.4 Other Parameters

| Parameter | Default | Effect |
|---|---|---|
| `SKIP_REPEAT` | True | Skip pages where model didn't produce EOS (likely repetition failure) |
| `PRINT_NUM_VIS_TOKENS` | False | Debug: print vision token counts |
| `block_size` | 256 | vLLM block size for paged attention |
| `enforce_eager` | False | Disable CUDA graphs (set True for debugging) |
| `tensor_parallel_size` | 1 | Multi-GPU parallelism |
| `swap_space` | 0 | CPU swap space for vLLM (GB) |

---

## 8. Parsing Tiers

### Tier 1 — Draft (Fastest)

Minimal resolution, no layout detection, most aggressive n-gram filtering.

**Config:**
- Mode: **Tiny** (base_size=512, image_size=512, crop_mode=False)
- Prompt: `<image>\nFree OCR.`
- Vision tokens: 64 per page

**HuggingFace:**
```python
res = model.infer(
    tokenizer,
    prompt="<image>\nFree OCR. ",
    image_file='page.jpg',
    output_path='output/',
    base_size=512,
    image_size=512,
    crop_mode=False,
    save_results=True
)
```

**vLLM (config.py):**
```python
BASE_SIZE = 512
IMAGE_SIZE = 512
CROP_MODE = False
PROMPT = '<image>\nFree OCR.'
```

---

### Tier 2 — Basic

Small resolution with layout detection. Good for simple documents.

**Config:**
- Mode: **Small** (base_size=640, image_size=640, crop_mode=False)
- Prompt: `<image>\n<|grounding|>Convert the document to markdown.`
- Vision tokens: 100 per page

**HuggingFace:**
```python
res = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown. ",
    image_file='page.jpg',
    output_path='output/',
    base_size=640,
    image_size=640,
    crop_mode=False,
    save_results=True
)
```

**vLLM (config.py):**
```python
BASE_SIZE = 640
IMAGE_SIZE = 640
CROP_MODE = False
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
```

---

### Tier 3 — Standard

Base resolution, good balance of quality and speed. Single global view at 1024×1024.

**Config:**
- Mode: **Base** (base_size=1024, image_size=1024, crop_mode=False)
- Prompt: `<image>\n<|grounding|>Convert the document to markdown.`
- Vision tokens: 256 per page

**HuggingFace:**
```python
res = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown. ",
    image_file='page.jpg',
    output_path='output/',
    base_size=1024,
    image_size=1024,
    crop_mode=False,
    save_results=True
)
```

**vLLM (config.py):**
```python
BASE_SIZE = 1024
IMAGE_SIZE = 1024
CROP_MODE = False
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
```

---

### Tier 4 — High Quality

Large resolution, single view. Best non-dynamic option.

**Config:**
- Mode: **Large** (base_size=1280, image_size=1280, crop_mode=False)
- Prompt: `<image>\n<|grounding|>Convert the document to markdown.`
- Vision tokens: 400 per page

**HuggingFace:**
```python
res = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown. ",
    image_file='page.jpg',
    output_path='output/',
    base_size=1280,
    image_size=1280,
    crop_mode=False,
    save_results=True
)
```

**vLLM (config.py):**
```python
BASE_SIZE = 1280
IMAGE_SIZE = 1280
CROP_MODE = False
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
```

---

### Tier 5 — Maximum Quality (Gundam)

Dynamic resolution with global 1024×1024 view plus up to 9 local 640×640 crops. Best quality, highest VRAM usage.

**Config:**
- Mode: **Gundam** (base_size=1024, image_size=640, crop_mode=True)
- Prompt: `<image>\n<|grounding|>Convert the document to markdown.`
- Vision tokens: Variable (256 global + n×100 local + separators)
- MAX_CROPS: 6 (default) or 9 (maximum)
- Relaxed n-gram settings for quality

**HuggingFace:**
```python
res = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown. ",
    image_file='page.jpg',
    output_path='output/',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True
)
```

**vLLM (config.py):**
```python
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 9          # Maximum crops (default 6, GPU memory permitting)
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
```

> **VRAM Warning**: MAX_CROPS=9 requires significant GPU memory. Use MAX_CROPS=6 if you have ≤40GB VRAM.

---

## 9. Processing PDFs (Full Workflow)

### Step 1: Edit config.py

```python
# In DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py

# Choose your resolution mode (see Tier descriptions above)
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6
MAX_CONCURRENCY = 100
NUM_WORKERS = 64
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # or local path

INPUT_PATH = '/path/to/your/paper.pdf'
OUTPUT_PATH = '/path/to/output/'

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
```

### Step 2: Run

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_pdf.py
```

### Step 3: Output Files

| File | Description |
|---|---|
| `{name}.mmd` | Clean markdown (layout tags stripped, images replaced with `![](...)`) |
| `{name}_det.mmd` | Raw output with `<\|ref\|>...<\|det\|>` layout detection tags |
| `{name}_layouts.pdf` | Visual PDF with bounding boxes drawn on each page |
| `images/` | Cropped images extracted from detected image regions |

**Post-processing applied automatically:**
- `<|ref|>image<|/ref|><|det|>...<|/det|>` → `![](images/X_Y.jpg)`
- Other `<|ref|>...<|det|>` tags are stripped
- `\coloneqq` → `:=`, `\eqqcolon` → `=:`
- Excessive newlines collapsed
- Pages separated by `<--- Page Split --->`

---

## 10. Output Format

### Clean Markdown (.mmd)
```markdown
# Title of Paper

## Abstract

This paper presents...

\[E = mc^2\]

| Column 1 | Column 2 |
|---|---|
| Data | Data |

![](images/0_0.jpg)

<--- Page Split --->

## 1. Introduction
...
```

### Detection Output (_det.mmd)
```markdown
<|ref|>title<|/ref|><|det|>[[10, 50, 990, 120]]<|/det|>
# Title of Paper

<|ref|>text<|/ref|><|det|>[[10, 130, 990, 400]]<|/det|>
This paper presents...

<|ref|>image<|/ref|><|det|>[[100, 410, 500, 700]]<|/det|>
```

The detection output preserves bounding box coordinates in `[x1, y1, x2, y2]` format (0-999 normalized).

---

## 11. Tier Comparison Summary

| Tier | Mode | Vision Tokens | Layout | Quality | Speed | VRAM |
|---|---|---|---|---|---|---|
| 1 (Draft) | Tiny 512 | 64 | No | Low | Fastest | Lowest |
| 2 (Basic) | Small 640 | 100 | Yes | Medium-Low | Fast | Low |
| 3 (Standard) | Base 1024 | 256 | Yes | Medium | Moderate | Medium |
| 4 (High) | Large 1280 | 400 | Yes | High | Slower | High |
| 5 (Maximum) | Gundam 1024+640 | 256 + n×100 | Yes | Highest | Slowest | Highest |

---

## 12. Known Limitations

1. **CUDA-only**: Requires NVIDIA GPU with flash-attn. No MPS (Apple Silicon), CPU, or ROCm support.
2. **No page range selection**: PDF processing always converts all pages. To process a subset, pre-extract pages with a tool like `pdftk` or PyMuPDF.
3. **Config via module-level constants**: Resolution mode, paths, and prompts are set by editing `config.py` (vLLM backend) or passing parameters to `model.infer()` (HF backend). No CLI arguments.
4. **vLLM v0 only**: The custom model code explicitly sets `VLLM_USE_V1 = '0'`. The upstream vLLM integration (v0.11+) does not have this limitation.
5. **Max output tokens = 8192**: Hard-coded in all scripts. Very long pages may be truncated.
6. **PDF rendering at 144 DPI**: Fixed in `run_dpsk_ocr_pdf.py` via PyMuPDF. Not configurable without code changes.
7. **N-gram processor settings vary by script**: PDF uses more aggressive de-duplication (ngram=20, window=50) than image/eval scripts (ngram=30-40, window=90). May need tuning per use case.
8. **No cross-page merging**: Unlike OCRFlux, DeepSeek-OCR processes each page independently. Tables or paragraphs spanning pages are not reassembled.
9. **SKIP_REPEAT behavior**: Pages where the model fails to produce an EOS token (likely repetition) are silently dropped from the clean output when `SKIP_REPEAT=True`.
10. **Layout tags use custom format**: The `<|ref|>...<|det|>` grounding format is specific to DeepSeek-OCR and requires post-processing to convert to standard markdown.
11. **Pinned dependency versions**: `transformers==4.46.3`, `vllm==0.8.5` — may conflict with other VLM tools in the same environment.
