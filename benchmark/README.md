# benchmark/

Scripts used for the [New Parsers Benchmark](../parsing%20tools/comparison_reports/new_parsers_benchmark_M2Pro.md) (MinerU2.5 / Marker / QARI on Apple M2 Pro / 16 GB).

Outputs, sample images, logs, model caches, and venvs are **gitignored** — only the scripts are tracked. Regenerate with:

| File | Purpose |
|---|---|
| `prep_samples.py` | Render representative sample pages (1-page PDF + 300-DPI PNG) from a science PDF and an Arabic PDF. Set `ARABIC_PDF=/path/to/arabic.pdf`. |
| `run_qwenvl_ocr.py` | Generic Qwen-VL OCR runner (QARI / Qwen2-VL or Qwen2.5-VL). Records wall time + peak MPS memory to a `.meta.json`. |
| `marker_tier4.json`, `tier1.json`, `tier2.json` | Marker tier configs (see the [Marker guide](../parsing%20tools/PARSING_GUIDE-%20Marker.md)). |

Each tool runs in its own `uv` venv (Python 3.12; the ML stack does not yet support 3.14):

```bash
uv venv .venv-mineru --python 3.12 && uv pip install --python .venv-mineru/bin/python "mineru[core]"
uv venv .venv-marker --python 3.12 && uv pip install --python .venv-marker/bin/python marker-pdf psutil
# QARI uses a base venv with: torch transformers pillow accelerate pymupdf
```
