# Chandra OCR Parsing Guide (Datalab's Marker successor)

A guide for [Chandra](https://github.com/datalab-to/chandra) (`chandra-ocr`), Datalab's newer document-OCR VLM, which they position as **higher accuracy than Marker**. PDF/image → Markdown / HTML / JSON, with layout, tables, math, and handwriting.

> ⚠️ **Read this first — hardware reality on 16 GB.** The open model `datalab-to/chandra` is **8.77 B parameters** (~17.5 GB in fp16/bf16). That **exceeds 16 GB unified memory before any activations**, so it does **not** run locally on a 16 GB Apple-silicon machine. In [our benchmark](comparison_reports/new_parsers_benchmark_M2Pro.md), a *2 B* model (QARI) already drove the MPS driver allocation to ~17.6 GB and spilled into swap, and hard-OOM'd at full resolution. Chandra is **4× larger**. On an M2 Pro / 16 GB it will OOM at load or swap-thrash to tens of minutes per page. **Use the hosted API, or a ≥ 32 GB (ideally 48 GB) machine.**

---

## What it is

- **Model:** `datalab-to/chandra`, 8.77 B-param VLM. Code Apache-2.0; weights modified OpenRAIL-M (free for research / personal / startups < $2 M).
- **Outputs:** Markdown, HTML, JSON (+ images, metadata, token counts).
- **Reported quality** (vendor benchmarks, March 2026): olmOCR **85.9**, ArXiv-math **90.2**, tables **89.9**, Arabic **68.4%** — i.e. better than Marker, and notably better Arabic than Marker's Surya path.
- **Relationship:** same team as **Marker** and **Surya**; Chandra is the higher-accuracy successor.

---

## Install

```bash
uv venv .venv-chandra --python 3.12
uv pip install --python .venv-chandra/bin/python "chandra-ocr[hf]"   # [hf] is required for the local backend
```

The base `chandra-ocr` install does **not** include the local-inference deps — without the `[hf]` extra the `--method hf` path raises `ImportError: ... requires additional dependencies`.

---

## Usage

### Hosted API (recommended on ≤ 16 GB)
Use the Datalab API (managed, zero data retention / SOC-2 per their docs). This is the practical path on a 16 GB Mac — no 17 GB download, no OOM.

### Local (only on ≥ 32 GB)
```bash
export TORCH_DEVICE=mps        # or cuda
chandra input.pdf ./output --method hf --max-output-tokens 4000 \
        --page-range "1-5" --save-html
```
| Flag | Purpose |
|---|---|
| `--method hf` | local HuggingFace model (needs `[hf]` extra + ≥ 32 GB) |
| `--method vllm` | vLLM server (CUDA) |
| `--page-range` | subset of PDF pages |
| `--max-output-tokens` | cap per-page generation |
| `--save-html` / `--no-headers-footers` / `--include-images` | output controls |

---

## Verdict for this repo

On the target hardware (**M2 Pro / 16 GB**), Chandra is **API-only** — it is the best-on-paper of the Datalab line but is locked out locally by the 16 GB ceiling. If/when run on a larger machine (or via API), it is the natural upgrade from Marker for mixed science + Arabic documents. Until then, **MinerU2.5** is the local default for science PDFs (see its guide), and the Arabic text problem is best handled by a [hybrid pipeline](RESEARCH%20REPORT%20-%20Marker%20Arabic%20OCR%20Optimization.md).
