# dots.ocr Parsing Guide

> A comprehensive guide to all parsing options in dots.ocr, ordered from lowest to highest quality.

**Model**: dots.ocr — a 1.7B parameter VLM based on Qwen2.5-VL, by RedNote HiLab
**Architecture**: Single unified model (no multi-stage pipeline)
**Supported Inputs**: PDF (multi-page), JPG, JPEG, PNG
**Output**: Markdown (`.md`), structured layout JSON (`.json`), annotated visualization (`.jpg`)

---

## Table of Contents

1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Parsing Modes (Lowest to Highest Quality)](#2-parsing-modes-lowest-to-highest-quality)
3. [Inference Backends](#3-inference-backends)
4. [Quality-Tuning Parameters](#4-quality-tuning-parameters)
5. [Running the Parser (CLI)](#5-running-the-parser-cli)
6. [Web UIs](#6-web-uis)
7. [Docker Deployment](#7-docker-deployment)
8. [Tier Summary Table](#8-tier-summary-table)
9. [Tips for Scientific Papers](#9-tips-for-scientific-papers)
10. [Hardware Requirements & Compatibility](#10-hardware-requirements--compatibility)
11. [Running on Apple Silicon (M1/M2/M3)](#11-running-on-apple-silicon-m1m2m3)
12. [Known Limitations](#12-known-limitations)

---

## 1. Prerequisites & Installation

```bash
# Create environment
conda create -n dots_ocr python=3.12
conda activate dots_ocr

# Install PyTorch (CUDA 12.8 example — adjust for your platform)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

# Install dots.ocr
cd /path/to/dots.ocr
pip install -e .

# Download the model weights
python3 tools/download_model.py                          # from HuggingFace (default)
python3 tools/download_model.py -t modelscope            # from ModelScope (alternative)
```

> **Note**: The model is saved to `./weights/DotsOCR`. The directory name must NOT contain periods (`.`) due to a transformers integration quirk.

---

## 2. Parsing Modes (Lowest to Highest Quality)

dots.ocr uses **prompt modes** to control what the model extracts. There are four modes, listed here from fastest/simplest to most complete.

---

### Tier 1 — `prompt_ocr` (Text Extraction Only)

**What it does**: Extracts raw text from the document. No layout detection, no bounding boxes, no structural information. Page headers and footers are excluded.

**Best for**: Quick text dumps, content indexing, search corpus creation.

**Output**: Plain text (no structured JSON).

```bash
python3 dots_ocr/parser.py input.pdf --prompt prompt_ocr
```

**Quality**: Lowest structural fidelity. You get the words but lose all formatting, tables, formulas, and reading order information.

---

### Tier 2 — `prompt_layout_only_en` (Layout Detection Only)

**What it does**: Detects the layout structure — bounding boxes and category labels for every element on the page. Does NOT extract any text.

**Best for**: Document structure analysis, layout benchmarking, preprocessing for downstream OCR.

**Output**: JSON with `bbox` and `category` fields (no `text` field).

```bash
python3 dots_ocr/parser.py input.pdf --prompt prompt_layout_only_en
```

**Detected categories** (11 types):
| Category | Description |
|----------|-------------|
| Title | Document/section title |
| Section-header | Section headings |
| Text | Body text paragraphs |
| List-item | Bulleted/numbered list entries |
| Caption | Figure/table captions |
| Footnote | Footnotes |
| Formula | Mathematical equations |
| Table | Tabular data |
| Picture | Images/figures |
| Page-header | Running headers |
| Page-footer | Running footers |

**Quality**: Layout structure only — useful when you need to know *where* things are but don't need the content itself.

---

### Tier 3 — `prompt_grounding_ocr` (Region-Specific OCR)

**What it does**: Extracts text from a specific rectangular region of the page, defined by a bounding box `[x1, y1, x2, y2]`.

**Best for**: Targeted extraction of specific sections, tables, or formulas. Useful when you already know where the content is.

**Output**: Text extracted from the specified region.

```bash
python3 dots_ocr/parser.py input.png --prompt prompt_grounding_ocr --bbox 163 241 1536 705
```

> **Note**: Coordinates are in pixel space relative to the input image dimensions.

**Quality**: Focused and accurate for targeted regions, but requires manual bbox specification. Not suitable for full-document parsing.

---

### Tier 4 — `prompt_layout_all_en` (Full Layout + OCR) **[DEFAULT]**

**What it does**: Performs complete document analysis — detects all layout elements with bounding boxes, categorizes them, AND extracts text content. Elements are sorted in human reading order. This is the **default mode**.

**Best for**: Full document conversion, scientific paper parsing, archival conversion.

**Output**:
- `.json` — Structured layout data: `[{bbox, category, text}, ...]`
- `.md` — Markdown with formatted content
- `_nohf.md` — Markdown excluding page headers/footers
- `.jpg` — Annotated visualization with colored bounding boxes

```bash
python3 dots_ocr/parser.py input.pdf --prompt prompt_layout_all_en
```

**Text formatting by category**:
| Category | Format |
|----------|--------|
| Formula | LaTeX (`$$...$$`) |
| Table | HTML (`<table>...</table>`) |
| Picture | Embedded base64 image (`![](data:image/png;base64,...)`) |
| Text, Title, Section-header, etc. | Markdown |

**Quality**: Highest. Full structural + content extraction with reading order preservation.

---

## 3. Inference Backends

dots.ocr supports two inference backends, which significantly affect speed and throughput.

### Option A — vLLM Server (Recommended for Production)

The fastest option. Runs the model as an OpenAI-compatible API server.

**Launch the server**:
```bash
# Using the provided launch script:
bash demo/launch_model_vllm.sh

# Or manually:
vllm serve rednote-hilab/dots.ocr \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --chat-template-content-format string \
    --served-model-name model \
    --trust-remote-code
```

> Since **vLLM v0.11.0**, dots.ocr is officially integrated — no out-of-tree registration needed.
> For vLLM v0.9.1–v0.10.x, the launch script patches the entrypoint automatically.

**Run the parser against the server**:
```bash
python3 dots_ocr/parser.py input.pdf \
    --ip localhost --port 8000 \
    --model_name model \
    --prompt prompt_layout_all_en \
    --num_thread 16
```

**Advantages**: Multi-threaded PDF processing (up to 128 threads), high GPU utilization, production-ready.

---

### Option B — HuggingFace Local Inference (No Server Needed)

Loads the model directly via transformers. Simpler setup but slower — single-threaded only.

```bash
python3 dots_ocr/parser.py input.pdf --use_hf true --prompt prompt_layout_all_en
```

Or use the standalone demo:
```bash
python3 demo/demo_hf.py
```

**Model loading configuration** (in demo_hf.py):
```python
model = AutoModelForCausalLM.from_pretrained(
    "./weights/DotsOCR",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
```

**Key constraint**: When `use_hf=True`, `num_thread` is forced to 1 (sequential page processing).

**Advantages**: No server setup, simpler deployment. Good for small jobs.

---

## 4. Quality-Tuning Parameters

Beyond the prompt mode, these parameters control output quality and resolution.

### Resolution Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--dpi` | 200 | Any integer | PDF-to-image conversion DPI. Higher = sharper input but slower. |
| `--min_pixels` | 3136 | 3136+ | Minimum total pixel count after resize. Floor: 56x56. |
| `--max_pixels` | 11289600 | Up to 11289600 | Maximum total pixel count. Ceiling: ~3360x3360. |
| `--no_fitz_preprocess` | *(off)* | Flag | Disables automatic DPI upsampling via PyMuPDF. |

**Image resize rules**:
- Dimensions are rounded to multiples of 28 (`IMAGE_FACTOR`)
- Aspect ratio must be < 200:1
- Images are smart-resized to fit within `[min_pixels, max_pixels]` while preserving aspect ratio

**Resolution quality ladder**:
```
Low:    --max_pixels 50000                  # Fast, lossy
Medium: (defaults)                          # Balanced
High:   --min_pixels 100000 --dpi 300       # Sharper, slower
Max:    --min_pixels 100000 --max_pixels 11289600 --dpi 300  # Best possible
```

### Model Generation Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--temperature` | 0.1 | Sampling randomness. Lower = more deterministic/consistent. |
| `--top_p` | 1.0 | Nucleus sampling. 1.0 = no filtering. Lower = more focused. |
| `--max_completion_tokens` | 16384 | Max output tokens. Increase for very dense pages. |

**For scientific papers**, keep `temperature=0.1` (or even `0.0`) for maximum reproducibility.

### Performance Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--num_thread` | 16 | Parallel page processing threads (vLLM only). |
| `--gpu-memory-utilization` | 0.95 | vLLM GPU memory allocation (server-side). |

---

## 5. Running the Parser (CLI)

### Full CLI Reference

```bash
python3 dots_ocr/parser.py <input_path> [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input_path` | positional | *required* | Path to PDF or image file |
| `--output` | str | `./output` | Output directory |
| `--prompt` | str | `prompt_layout_all_en` | Parsing mode |
| `--bbox` | 4 ints | — | Bounding box for grounding OCR: `x1 y1 x2 y2` |
| `--protocol` | str | `http` | Server protocol (`http` or `https`) |
| `--ip` | str | `localhost` | vLLM server IP |
| `--port` | int | `8000` | vLLM server port |
| `--model_name` | str | `model` | Model name on server |
| `--temperature` | float | `0.1` | Sampling temperature |
| `--top_p` | float | `1.0` | Top-p sampling |
| `--dpi` | int | `200` | PDF conversion DPI |
| `--max_completion_tokens` | int | `16384` | Max output tokens |
| `--num_thread` | int | `16` | Parallel threads |
| `--no_fitz_preprocess` | flag | — | Disable fitz DPI upsampling |
| `--min_pixels` | int | — | Min image pixels |
| `--max_pixels` | int | — | Max image pixels |
| `--use_hf` | bool | `False` | Use local HF model instead of vLLM |

### Example Commands

```bash
# Fastest: text-only extraction
python3 dots_ocr/parser.py paper.pdf --prompt prompt_ocr --num_thread 64

# Layout detection only (no text)
python3 dots_ocr/parser.py paper.pdf --prompt prompt_layout_only_en

# Full parsing (default) — best quality
python3 dots_ocr/parser.py paper.pdf --prompt prompt_layout_all_en --output ./results

# Full parsing with max resolution
python3 dots_ocr/parser.py paper.pdf \
    --prompt prompt_layout_all_en \
    --dpi 300 \
    --max_pixels 11289600 \
    --output ./results

# Region-specific OCR
python3 dots_ocr/parser.py page.png --prompt prompt_grounding_ocr --bbox 100 200 800 600

# Local HF inference (no server)
python3 dots_ocr/parser.py paper.pdf --use_hf true --prompt prompt_layout_all_en
```

---

## 6. Web UIs

dots.ocr ships with three web interfaces for interactive use.

### Gradio UI (Single File)

```bash
python3 demo/demo_gradio.py
```
- Upload PDF or image
- Select prompt mode
- Configure server connection and pixel parameters
- View: rendered markdown, raw markdown, layout JSON
- Download results as ZIP

### Gradio Batch UI (Multi-File)

```bash
python3 demo/demo_gradio_batch.py
```
- Batch upload multiple files
- Up to 6 concurrent workers (configurable)
- Editable markdown/JSON outputs with auto-save
- Automatic retry with exponential backoff (up to 5 retries)
- Custom export scripts via built-in Python API

### Gradio Annotation UI (Region OCR)

```bash
python3 demo/demo_gradio_annotion.py
```
- Draw bounding boxes on images
- Automatically switches to `prompt_grounding_ocr` mode
- Single-box annotation (DELETE to clear)
- Useful for extracting specific tables or figures

### Streamlit UI (Minimal)

```bash
streamlit run demo/demo_streamlit.py
```
- Simple single-image interface
- Three input modes: upload, URL/path, test image
- Sidebar configuration panel
- Two-column result display (image + markdown)

---

## 7. Docker Deployment

### Build and Run

```bash
cd docker

# Build image (based on vllm/vllm-openai:v0.9.1)
docker build -t dots-ocr:latest .

# Place model weights in ./model/dots.ocr/, then:
docker compose up
```

**docker-compose.yml** exposes port `8000` and mounts `./model/dots.ocr` as the model weights directory. GPU passthrough is configured for device `0`.

**For vLLM v0.11.0+**, you can skip the custom Docker image entirely and use the official vLLM image:
```bash
docker run --gpus all -p 8000:8000 \
    -v /path/to/weights:/weights \
    vllm/vllm-openai:v0.11.0 \
    --model /weights/DotsOCR \
    --trust-remote-code \
    --gpu-memory-utilization 0.95
```

---

## 8. Tier Summary Table

| Tier | Prompt Mode | Layout | Text | Reading Order | Formula | Tables | Speed | Use Case |
|------|-------------|--------|------|---------------|---------|--------|-------|----------|
| 1 | `prompt_ocr` | No | Yes (plain) | No | No | No | Fastest | Text dumps, search indexing |
| 2 | `prompt_layout_only_en` | Yes (bbox + category) | No | Yes | Detected but not extracted | Detected but not extracted | Fast | Layout analysis, preprocessing |
| 3 | `prompt_grounding_ocr` | No (user-specified region) | Yes (region) | N/A | If in region | If in region | Fast | Targeted extraction |
| 4 | `prompt_layout_all_en` | Yes (bbox + category) | Yes (formatted) | Yes | LaTeX | HTML | Slowest | Full document conversion |

**Resolution ladder** (apply to any tier):

| Level | Settings | Notes |
|-------|----------|-------|
| Low | `--max_pixels 50000` | Fast, lower accuracy on small text |
| Default | *(no overrides)* | Balanced for most documents |
| High | `--dpi 300 --min_pixels 100000` | Better for dense pages |
| Maximum | `--dpi 300 --max_pixels 11289600` | Best quality, highest memory/compute |

**Backend choice**:

| Backend | Threads | Speed | Setup |
|---------|---------|-------|-------|
| vLLM server | Up to 128 | Fast (parallel pages) | Requires running server |
| HuggingFace local | 1 (forced) | Slow (sequential) | No server needed |

---

## 9. Tips for Scientific Papers

1. **Use `prompt_layout_all_en`** — it's the only mode that extracts formulas as LaTeX and tables as HTML.

2. **Increase DPI for dense papers**: arXiv papers with small fonts or complex equations benefit from `--dpi 300`.

3. **Keep temperature low** (`0.1` or `0.0`) for reproducible, deterministic output.

4. **Use `_nohf.md` output** — the "no header/footer" variant strips running headers and page numbers, giving cleaner markdown.

5. **For long papers**, increase `--max_completion_tokens` to `24000` or higher to avoid truncation on dense pages.

6. **Multi-thread for speed**: With vLLM, set `--num_thread` to at least the number of pages in your PDF.

7. **Check the visualization**: The `.jpg` output shows colored bounding boxes overlaid on each page — useful for verifying that formulas, tables, and figures were correctly detected.

8. **Region OCR for problem areas**: If full-page parsing misses a table or equation, use `prompt_grounding_ocr` with the bbox coordinates from the layout JSON to re-extract that specific region.

---

## 10. Hardware Requirements & Compatibility

### Minimum Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU** | NVIDIA CUDA-capable GPU | **Mandatory** for vLLM backend; hardcoded in HF backend |
| **VRAM** | ~4 GB+ (fp16) | 1.7B params = ~3.4 GB in float16, plus inference overhead |
| **RAM** | 16 GB+ | For PDF preprocessing, image loading, multi-threading |
| **Python** | >= 3.10 | 3.12 recommended |
| **CUDA** | 12.x | Install instructions target CUDA 12.8 wheels |

### Hard CUDA Dependencies

These libraries **will not install** without an NVIDIA GPU and CUDA toolkit:

| Dependency | Version | Why It's CUDA-Only |
|------------|---------|-------------------|
| `flash-attn` | 2.8.0.post2 | Flash Attention 2 — NVIDIA-only optimized attention kernel |
| `vllm` | 0.9.1+ | Inference server — no CPU/MPS backend |
| PyTorch (as installed) | 2.7.0+cu128 | The install instructions use CUDA 12.8 wheels specifically |

### Code-Level CUDA Lock-In

The codebase hardcodes CUDA in multiple places:

- **`demo/demo_hf.py`**: `inputs = inputs.to("cuda")` — crashes on non-CUDA systems
- **`dots_ocr/model/inference.py`**: Same `.to("cuda")` pattern
- **`dots_ocr/parser.py`**: `attn_implementation="flash_attention_2"` — fails without flash-attn
- **Model dtype**: `torch.bfloat16` — limited operation support on non-CUDA backends

### Platform Compatibility Matrix

| Platform | vLLM Backend | HF Backend | Status |
|----------|-------------|------------|--------|
| Linux + NVIDIA GPU | Full support | Full support | **Recommended** |
| Linux + CPU only | No | Theoretically possible (with code patches) | Extremely slow |
| macOS + Apple Silicon | No | No (hardcoded CUDA) | **Not supported** |
| macOS + Intel | No | No | Not supported |
| Windows + NVIDIA GPU | Partial (vLLM Linux-only) | Possible with patches | Untested |

---

## 11. Running on Apple Silicon (M1/M2/M3)

> **Tested on**: Apple M2 Pro, 16 GB unified RAM, macOS

### The Short Answer

**dots.ocr cannot run natively on Apple Silicon.** All four parsing tiers hit the same blockers — the bottleneck is the inference backend, not the prompt mode.

### What Blocks Native Execution

| Blocker | Detail |
|---------|--------|
| `flash-attn` won't install | No ARM/Apple Silicon wheels. `pip install flash-attn` fails immediately. |
| `.to("cuda")` hardcoded | Both HF and demo scripts send tensors to `"cuda"`. Crashes with `RuntimeError: No CUDA GPUs available`. |
| vLLM requires NVIDIA | vLLM has zero MPS or CPU support. Cannot serve the model. |
| `bfloat16` limited on MPS | Apple MPS has incomplete bf16 operator coverage. Many ops would error or silently fall back to CPU. |

### Could You Patch It for MPS?

In theory, the 1.7B model (~3.4 GB in fp16) fits comfortably in 16 GB unified memory. A patch would need to:

1. Remove `flash-attn` from requirements (use default SDPA attention)
2. Change all `.to("cuda")` → `.to("mps")` or `.to("cpu")`
3. Change `torch.bfloat16` → `torch.float16`
4. Remove `attn_implementation="flash_attention_2"` from model loading
5. Use HF backend only (skip vLLM)

**However**, the model uses custom remote code (`trust_remote_code=True`) from Qwen2.5-VL which may contain CUDA-specific kernels. MPS compatibility is not guaranteed and would require testing.

### Practical Alternatives for Apple Silicon Users

| Option | Effort | Cost | Notes |
|--------|--------|------|-------|
| **Google Colab (free T4)** | Low | Free | dots.ocr includes `demo/demo_colab_remote_server.ipynb` — upload your PDF, run in Colab, download results |
| **Google Colab Pro (A100)** | Low | ~$10/mo | Much faster than free T4 for batch processing |
| **Cloud GPU rental** (RunPod, Vast.ai, Lambda Labs) | Medium | ~$0.50–1.50/hr | Spin up an instance, run vLLM server, connect remotely |
| **Patch for MPS** | High | Free | Modify inference code as described above — fragile, no guarantee of correctness |
| **Use other local tools** | None | Free | Marker, Docling, MarkItDown, and SciPDF all run natively on Apple Silicon |

### Recommended Approach: Google Colab

The lowest-friction path is the included Colab notebook:

1. Open `demo/demo_colab_remote_server.ipynb` in Google Colab
2. Select a GPU runtime (T4 free, or A100 with Pro)
3. Run the notebook to start a vLLM server with an ngrok tunnel
4. Point your local parser at the Colab server:
   ```bash
   python3 dots_ocr/parser.py paper.pdf \
       --ip <ngrok-url> --port 443 --protocol https \
       --prompt prompt_layout_all_en
   ```

---

## 12. Known Limitations

- **High character-to-pixel ratio**: Very small text can cause failures. Workaround: increase DPI to 200+ or enlarge the image.
- **Repetitive special characters**: Sequences like `...`, `___`, or `---` can cause infinite generation loops. Use a lower `max_completion_tokens` as a safety net.
- **Complex tables**: Deeply nested or merged-cell tables may not convert perfectly to HTML.
- **Picture content**: Detected and bounded but not semantically described (no image captioning).
- **Max resolution**: 11,289,600 total pixels (~3360x3360). Larger images are automatically downscaled.
- **Aspect ratio**: Must be < 200:1 or the image is rejected.
- **No CPU inference for vLLM**: The vLLM backend requires a CUDA GPU. HF local mode supports CPU but is extremely slow.
- **100+ languages supported**, but prompt modes are English-only (`_en` suffix). Chinese prompts may be available in future releases.
