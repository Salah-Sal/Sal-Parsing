# MinerU2.5 PDF Parsing Guide: Best Local Engine for Scientific PDFs

A practical guide for parsing arXiv-style papers with [MinerU](https://github.com/opendatalab/MinerU) (`mineru` 3.3.1, which ships the **MinerU2.5** 1.2 B document-parsing VLM). Benchmarked on an Apple **M2 Pro / 16 GB** — see [the benchmark report](comparison_reports/new_parsers_benchmark_M2Pro.md).

> **TL;DR for this repo:** MinerU2.5 was the strongest *local* parser tested for scientific PDFs — publication-quality math LaTeX, excellent tables, and the **only local tool that survived the quantum bra-ket / string-diagram page** (Marker without `--use_llm` truncated those equations). Peak ~3 GB RAM, so it fits 16 GB comfortably.

---

## Quick Start

```bash
# Install (isolated env recommended; needs Python ≤ 3.13 — NOT 3.14 yet)
uv venv .venv-mineru --python 3.12
uv pip install --python .venv-mineru/bin/python "mineru[core]"

# Convert on Apple Silicon (MPS)
export MINERU_DEVICE_MODE=mps
export MINERU_MODEL_SOURCE=huggingface
mineru -p paper.pdf -o ./output -b vlm-engine
```

Output lands in `./output/<name>/vlm/<name>.md` (+ `_content_list.json`, `_layout.pdf`, `_middle.json`, extracted images).

---

## Backends

| Backend (`-b`) | Model | Device | Best for |
|---|---|---|---|
| **`vlm-engine`** | MinerU2.5 (1.2 B VLM) | MPS / CUDA / CPU | **Default for science PDFs** — math, tables, layout. What this repo recommends. |
| `pipeline` | classic layout + OCR + formula | CPU / MPS | Has a `-l arabic` (and other) language hint; lighter; weaker on complex math |
| `vlm-vllm-*` / `*-http-client` | MinerU2.5 via vLLM/server | CUDA | Throughput on NVIDIA; **vLLM is not an MPS path** |

On Apple Silicon use **`vlm-engine`** (runs the VLM through transformers on MPS). The `vllm` backends are CUDA-only.

---

## Measured on Apple M2 Pro / 16 GB (June 2026)

| Metric | Value |
|---|---|
| Backend | `vlm-engine` (MinerU2.5 1.2 B), `MINERU_DEVICE_MODE=mps` |
| Peak resident memory | **3.02 GB** (0 swaps) — fits 16 GB easily |
| Model load | ~67 s (first call per process) |
| Inference | ~1–3 min/page (first page slower due to MPS warmup) |
| `sci-table` | ✅ full HTML table incl. colspan group-headers and every ± cell |
| `sci-math-std` (vMF/Bessel) | ✅ clean LaTeX (Newton iteration, Kummer `M(1/2,p/2,κ)`, eigenvalue conditions) |
| `sci-math-quantum` (bra-ket + diagram glyphs) | 🟢 full multi-line derivation as a LaTeX array; exotic spider glyphs approximated, never dropped |
| Arabic (`ar-table`, `ar-prose`) | 🟡 layout/table structure preserved, but letter/diacritic errors — use an Arabic-tuned pass for the text |

> Each `mineru` invocation spins a short-lived FastAPI worker and reloads the model (~67 s). To amortise that, point `-p` at a **folder** of PDFs and convert them in one call.

---

## Tips for arXiv papers

1. **Start with `vlm-engine`.** It handles body text, headings, references, tables, and most display math out of the box.
2. **Math-heavy / category-theory / quantum papers:** this is where MinerU2.5 shines vs. local Marker — it keeps multi-line aligned derivations as LaTeX arrays. Custom inline diagram glyphs (DisCoCat spiders, etc.) are *approximated*, so spot-check those.
3. **Batch to amortise load:** `mineru -p ./pdfs_folder -o ./out -b vlm-engine` loads the model once for the whole folder.
4. **Arabic / RTL:** `vlm-engine` is not Arabic-tuned. Either try `-b pipeline -l arabic`, or (better) use MinerU/Marker for *layout* and an Arabic VLM (QARI; hosted Baseer) for *text* — see the [Arabic survey](RESEARCH%20REPORT%20-%20Arabic%20OCR%20Landscape%20Survey.md).
5. **Python version:** the ML stack does not yet support Python 3.14 — pin 3.12/3.13 in the venv.

---

## When to use MinerU2.5 vs. the others

```
Scientific PDF, want local + free?
├── Heavy math / tables / quantum notation → MinerU2.5 (vlm-engine)   ← best local
├── Mostly text + tables, standard math    → MinerU2.5 or Marker (both good)
└── Exotic display-math glyphs, need perfection → Marker --use_llm, or hosted Chandra
Arabic / RTL?
└── Don't rely on a single 16 GB-local tool → hybridise (layout + Arabic-tuned text)
```
