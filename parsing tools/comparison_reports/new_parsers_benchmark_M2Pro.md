# Benchmark: New Parsers (MinerU2.5, Chandra, QARI/Baseer) on Apple M2 Pro

**Date:** June 2026
**Purpose:** Evaluate three tools that were missing from this repo — **MinerU2.5**, **Chandra** (Datalab's Marker successor), and the Arabic doc-VLM tier (**Baseer / QARI**) — on shared sample pages, *on real consumer Apple-silicon hardware*, and decide whether to add guides for them.

> Companion to the existing tier guides. Unlike the arXiv-only tier tests, this run includes **heavy math, quantum bra-ket notation, and fully-diacritized Arabic**, which are the failure modes that matter for the downstream corpus.

---

## Hardware & Environment

| Component | Detail |
|---|---|
| **Machine** | MacBook Pro, Apple **M2 Pro** |
| **CPU** | 12-core (8 performance + 4 efficiency) |
| **GPU / accelerator** | Apple integrated GPU via **Metal (MPS)** backend |
| **Unified memory** | **16 GB** |
| **Disk free** | ~765 GB |
| **OS** | macOS 26.5 (build 25F71) |
| **Python** | 3.12.8 (via `uv`; system default 3.14 is too new for the ML stack) |
| **Torch** | 2.12.0, `torch.backends.mps.is_available() == True` |
| **transformers** | 5.11.0 |
| **Isolation** | one `uv` venv per tool (`.venv`, `.venv-mineru`, `.venv-chandra`) to avoid pin conflicts |

**Why 16 GB is the binding constraint.** All three tools are vision-language models. At fp16, weight size alone is ≈ 2× the parameter count in GB. So a 1.2 B model needs ~2.5 GB (fine), a 3–4 B model ~6–8 GB (tight), and a ~9 B model ~17.5 GB (**will not fit** — forces swap or OOM). Peak resident memory is reported per tool below, measured with `/usr/bin/time -l` (`maximum resident set size`) for CLI tools and `torch.mps` allocator counters for the script-driven runs.

---

## Samples (hand-selected after visually inspecting candidates)

All five were chosen by *rendering candidate pages and looking at them*, not by blind heuristics. Each is provided as a 1-page PDF + 300-DPI PNG under `samples/`.

| Sample | Source | What it stresses |
|---|---|---|
| `sci-table` | *Recursive Language Models* (arXiv 2512.24601v2) p5 | Complex multi-section comparison table (colspan group headers, ± cells); matches the repo's existing benchmark paper |
| `sci-math-std` | Sra (2016), *Directional Statistics in ML* p5 | Clean typeset display math — vMF density, Bessel functions, fractions |
| `sci-math-quantum` | Coecke et al. (2020), *Meaning updating of density matrices* p10 | Bra-ket ⟨ψ\|, Σᵢxᵢ²\|i⟩⟨i\|, **inline string-diagram glyphs**, multi-line aligned derivations — mirrors the downstream quantum-NLP corpus |
| `ar-table` | Naseef (2018), *Kināyah in the Qurʾān* p191 | Fully-diacritized Qurʾanic verses + English in a table; RTL + mixed-script |
| `ar-prose` | Naseef (2018) p175 | Dense diacritized Arabic prose with footnotes |

---

## Tools under test

| Tool | Version / model | Params | Backend used | Role |
|---|---|---|---|---|
| **MinerU2.5** | `mineru` 3.3.1, `vlm-engine` | 1.2 B | transformers on MPS | Science PDFs (math/tables/layout) |
| **Chandra** | `chandra-ocr`, `datalab-to/chandra` | **8.77 B** | hf (local) on MPS | Datalab's higher-accuracy Marker successor |
| **QARI-OCR v0.3** | `NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct` | 2 B (Qwen2-VL) | transformers on MPS | Arabic doc-VLM (open stand-in for Baseer) |
| **Baseer** | — | 3 B | **not runnable locally** | See note below |

### Note on Baseer
The printed-document **Baseer** model (arXiv:2509.18174, the SOTA Arabic doc→Markdown VLM) is **not released as open weights**. The only public weights under the Misraj org, `Misraj/Baseer__Nakba`, are a *different* model: "Baseer-Nakba HTR," a fine-tune for the NAKBA 2026 **handwritten historical-manuscript** competition (the README itself directs document-extraction users to the hosted service at baseerocr.com). Benchmarking the handwriting variant on printed pages would be misleading, so Baseer is documented as **hosted-only** and **QARI-OCR v0.3** is benchmarked as the openly-runnable Arabic doc-VLM in its place.

---

## Results

### A. Does it run on 16 GB, and how heavy is it?

| Tool | Params | Peak memory (MPS) | Fits 16 GB? | Model load | Per-page inference | Notes |
|---|---|---|---|---|---|---|
| **MinerU2.5** (`vlm-engine`) | 1.2 B | **3.02 GB** RSS, 0 swaps | ✅ comfortably | ~67 s | ~1–3 min (first page slower; MPS warmup) | cleanest fit of all VLMs tested |
| **Marker** (Tier-4 local) | Surya stack | **1.59 GB** parent RSS¹ | ✅ | ~30 s | **~160 s/page** (807 s / 5, Tier-4 `force_ocr` @ 300 DPI; lighter tiers faster) | ¹multiprocessing — parent RSS understates true footprint (docs cite ~3.5 GB/worker) |
| **QARI-OCR v0.3** | 2 B | alloc 4.4 GB / **driver 17.6–17.8 GB** | ⚠️ only with swap | ~6 s | **85 s** (table) … **334 s** (dense prose) | at full 300-DPI it **hard-OOM'd** (`MTLBuffer 10 GB` alloc fail); had to cap to 1.0 MP |
| **Chandra** | **8.77 B** | ≈ 17.5 GB weights alone | ❌ **no** | — | — | exceeds physical RAM; **not run** (see §C) |

> Memory was measured with `/usr/bin/time -l` (`maximum resident set size`) and `torch.mps` allocator counters. The MPS *driver* allocation is the honest figure for "how close to the limit" — QARI's 17.6 GB shows a 2 B model already spilling into swap on this machine.

### B. Quality on the shared samples

Rating: ✅ excellent · 🟢 good · 🟡 usable-with-errors · 🔴 fails. Each cell reflects the *actual* output (saved under `benchmark/results/<tool>/`).

| Sample | MinerU2.5 | Marker (local) | QARI v0.3 |
|---|---|---|---|
| `sci-table` (complex table) | ✅ full HTML table, colspan group-headers, every ± cell; inline `$2\times$` kept | 🟢 markdown table extracted (21 rows), body reordered slightly | — (not Arabic; not run) |
| `sci-math-std` (vMF/Bessel) | ✅ clean LaTeX — Newton iter, Kummer `M(1/2,p/2,κ)`, eigenvalue conds | ✅ equally clean | — |
| `sci-math-quantum` (bra-ket + **diagram glyphs**) | 🟢 **full multi-line derivation as LaTeX array**, conjugates preserved; exotic spider glyphs approximated (graceful) | 🔴 **truncates the display equations** — drops the `⊙(Σxᵢ²\|i⟩⟨i\|)=` body, eq (14) → just `(14)`, proof derivation dropped | — |
| `ar-table` (Qurʾan + tashkeel + table) | 🟡 **table kept**, but Arabic letter/diacritic errors (حَتَّى→خَتَّى, الصَّلاةَ garbled) | 🟡 **best Arabic glyphs** (تَقْرَبُوا، جُنُبًا، الْغائِط، صَعيدًا، المائدة all correct; kept ﴿﴾) but **span duplication + RTL reorder** | 🟡 decent Arabic text, several errors; **flattened the table** + spurious `<u>` tags |
| `ar-prose` (dense diacritized) | 🟡 structure + English perfect, Arabic errors | 🟡 Surya glyphs decent, some duplication | 🔴 **repetition collapse** — looped `وَفَلَمَّا` to the token cap |

### C. Why Chandra was not run (16 GB hard limit)

Chandra's open model `datalab-to/chandra` is **8.77 B params** → ~17.5 GB in fp16/bf16, which **exceeds the 16 GB unified memory before any activations**. This is not speculation: in this same run, **QARI at only 2 B already drove the MPS driver allocation to 17.6 GB and spilled into swap**, and at full resolution it hard-OOM'd on a 10 GB Metal buffer. A 4×-larger model will OOM at load or swap-thrash to tens of minutes per page. We therefore did **not** download the 17 GB weights. **Chandra locally needs ≥ 32 GB (ideally 48 GB) or 4-bit quantization (and bitsandbytes 4-bit doesn't run on MPS) — otherwise use the Datalab hosted API.**

### D. Cross-cutting findings

1. **Best local science parser on this hardware = MinerU2.5.** Publication-quality math, excellent tables, 3 GB footprint, and — decisively — it is the **only local tool that survived the quantum diagram-glyph page**, emitting the full multi-line derivation where Marker (without `--use_llm`) truncated the equations.
2. **Marker's weak spot is local equation handling on exotic notation**, not tables or text. Its tier guide already says to escalate to `--use_llm` for hard math; this benchmark shows *why* — the local Texify/Surya path drops the diagram-embedded display equations.
3. **Arabic is nuanced, and the repo's "Surya Arabic is weak" framing needs a caveat.** On *clean printed, fully-diacritized* text, Marker/Surya actually produced the **best character/diacritic accuracy** of the three — better than the Arabic-specialized QARI here. The real failures were *layout* (RTL reorder, span duplication for Marker; table-flattening and repetition-collapse for QARI), not glyphs. The survey's weak-Arabic finding holds for **scanned** dictionaries, less so for clean digital Arabic.
4. **The 16 GB ceiling is the dominant constraint.** It rules Chandra out entirely, forces QARI to a reduced resolution (which hurts tiny diacritics and triggered the repetition collapse), and means any ≥7–9 B doc-VLM is API-only on this machine. MinerU2.5's 1.2 B size is what makes it the practical winner here.

## Verdict & repo actions

| Tool | Verdict on M2 Pro / 16 GB | Repo action |
|---|---|---|
| **MinerU2.5** | **Best local engine for the science PDFs.** 3 GB, strong math+tables, handles quantum notation. | **Add `PARSING_GUIDE - MinerU.md`** (vlm-engine, MPS env vars, the quantum-math win) |
| **Marker** | Excellent for tables/text/standard-math; **truncates exotic display math locally** → use `--use_llm` there. Surprisingly strong on clean printed Arabic glyphs. | Update the Marker guide with: the quantum-math local limitation, and the clean-print Arabic result |
| **Chandra** | **Not locally viable on 16 GB** (8.77 B). API-only here. | **Add `PARSING_GUIDE - Chandra.md`** noting it needs API / ≥32 GB; document the size math |
| **QARI v0.3** | Runs but **swaps hard** (2 B → 17.6 GB driver), needs resolution cap, **collapsed on the dense page**. Good glyphs when it behaves; loses table structure. | Add this 16 GB caveat + repetition-collapse note to the Arabic survey |
| **Baseer** | **Not runnable locally** — open weights are the NAKBA *handwriting* variant; printed-doc Baseer is hosted-only. | Correct the survey: Baseer = hosted (baseerocr.com), not open weights |

**One-line recommendation for this hardware:** use **MinerU2.5** as the default local parser for the scientific PDFs; reach for **Marker `--use_llm`** (or hosted Chandra) only for the most equation-dense / diagram-heavy pages; for Arabic, expect to **hybridise** (a layout-faithful parser for structure + an Arabic-tuned pass for glyphs) rather than trusting any single 16 GB-local tool — exactly the hybrid the [Marker Arabic optimization report](../RESEARCH%20REPORT%20-%20Marker%20Arabic%20OCR%20Optimization.md) already proposed.

---

*All outputs, per-tool timing logs, and `/usr/bin/time` captures are under `benchmark/` (gitignored: regenerate with the scripts in `benchmark/`). Hardware: Apple M2 Pro, 16 GB, macOS 26.5, June 2026.*
