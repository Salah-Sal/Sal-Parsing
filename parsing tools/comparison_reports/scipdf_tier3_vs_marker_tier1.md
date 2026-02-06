# SciPDF Tier 3 vs Marker Tier 1

**Paper:** "Recursive Language Models" (2512.24601v2)
**Machine:** Apple M2 Pro, 16GB unified RAM, macOS

**Important caveat:** SciPDF cannot limit to a page range — it processed all 19 pages. Marker Tier 1 processed only the first 9 pages. This means SciPDF's output includes appendices, benchmark details, system prompts, and additional trajectory examples that Marker never saw. The comparison below focuses on the overlapping content (first 9 pages) where possible, but output size metrics reflect the full outputs.

---

## Overview

| Metric | SciPDF Tier 3 | Marker Tier 1 |
|--------|--------------|---------------|
| **Tool** | SciPDF Parser v0.1.1 (GROBID v0.7.3) | Datalab Marker v1.10.1 |
| **Method** | GROBID CRF models via Docker, dict-to-markdown conversion | Surya layout detection, no OCR |
| **Time** | 58.9s (full 19 pages) | ~2m 14s (first 9 pages) |
| **Output size** | 94,853 chars (452 lines) | 46,282 chars (263 lines) |
| **Pages processed** | All 19 | First 9 only |

---

## 1. Title and Author Extraction

| Tool | Result |
|------|--------|
| **SciPDF** | `# Recursive Language Models` (H1), then `**Authors:** Alex L Zhang; Tim Kraska; Omar Khattab; Jaech Openai; A Richardson; C Hudson; C M De Bourcy; C Chan; F Wang; Von Lohmann; G Zhao; G Leclerc; O ' Connell; J Rizzo; J Gordon...` — correctly detects 3 actual authors but then appends **dozens of names from the Singh et al. (2025) reference list** as co-authors |
| **Marker** | `# Recursive Language Models` (H1), then `## Alex L. Zhang <sup>1</sup> Tim Kraska <sup>1</sup> Omar Khattab <sup>1</sup>` — correct 3 authors with superscript affiliations |

**Winner: Marker** — SciPDF's GROBID badly over-extracted authors, confusing names from the GPT-5 reference (Singh et al. 2025) as paper co-authors. Marker correctly identified exactly 3 authors.

---

## 2. Section Headings

| Tool | Hierarchy | Subsections |
|------|-----------|-------------|
| **SciPDF** | All sections as `## Heading` (flat H2) | Detected: Introduction, Recursive Language Models, Scaling Long Context Tasks, Tasks, Methods and Baselines, Results and Discussion, etc. — but subsections (3.1, 3.2, 4.1) are promoted to the same level as parent sections |
| **Marker** | H1 title → H2 numbered sections → H4 subsections (`#### 3.1. Tasks`) | Multi-level: 3.1, 3.2, 4.1 correctly nested under their parent sections |

SciPDF also produced spurious section headings from appendix benchmark content (e.g., `## Task 3`, `## Task 4`, ..., `## Task 20`) — these are OOLONG-Pairs benchmark tasks that GROBID misidentified as paper sections.

**Winner: Marker** — proper multi-level heading hierarchy. SciPDF flattens everything to H2 and misidentifies benchmark content as sections.

---

## 3. Body Text Quality

Both tools produce clean, well-merged paragraph text. No mid-sentence line breaks in either. Reading order across the two-column layout is correct in both.

**SciPDF:**
> We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling.

**Marker:**
> We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling.

One key difference: SciPDF occasionally merges adjacent sections' text into one blob. For example, the "Results and Discussion" section in SciPDF contains text from Observations 1-6 as a single continuous paragraph without the observation sub-headings that structure the text in the original PDF. Marker preserves the paragraph breaks and observation headers.

**Winner: Marker** — slightly better paragraph separation and preserved sub-structure within sections.

---

## 4. Inline Formatting

| Feature | SciPDF | Marker |
|---------|--------|--------|
| **Italic** | None | Yes — `*context rot*`, `*environment*`, `*programmatically*`, `*reasoning models*` |
| **Bold** | `**Authors:**`, `**Date:**`, `**DOI:**` (from our conversion script only) | None in body text |
| **Superscript** | None | Yes — `<sup>1</sup>`, `2 <sup>13</sup>` |

**Winner: Marker** — captures italic emphasis and superscript notation from the PDF. SciPDF/GROBID extracts plain text only with no inline formatting.

---

## 5. Mathematical Notation

| Tool | Inline math | Quality |
|------|-------------|---------|
| **SciPDF** | `P ∈ Σ ⋆`, `|P | ≫ K`, `Ω(|P |)`, `Ω(|P | 2)` | Unicode preserved, slightly looser spacing |
| **Marker** | `P ∈ Σ ⋆`, `\|P\| ≫ K`, `Ω(\|P\|)`, `Ω(\|P\| 2)` | Unicode preserved, tighter spacing, escaped pipes |

Neither tool produces LaTeX. Both use Unicode math symbols from the PDF's text layer.

**Winner: Tie** — both handle Unicode math comparably. Marker has slightly tighter spacing; SciPDF has cleaner (unescaped) pipe characters.

---

## 6. Algorithm/Pseudocode Blocks

**SciPDF:**
The algorithm blocks are not rendered. Algorithm 1's pseudocode is partially embedded in body text. The key lines (`state ← InitREPL`, `code ← LLMM(hist)`, etc.) are **missing** from the output — GROBID failed to extract the algorithm environment. Only a garbled fragment appears in the Formulas section:
```
← LLM M (hist) if action is Finish then return val // Flaw #2 out ← RUN(action, val) // Flaw #3...
```
This is actually part of Algorithm 2 crammed into a single formula line.

**Marker:**
```
Algorithm 1 A recursive language model, around LLM M
Input: prompt P
Output: response Y
state ← InitREPL(prompt=P)
state ← AddFunction(state, sub_RLMM)
hist ← [Metadata(state)]
while True do
   code ← LLMM(hist)
   (state, stdout) ← REPL(state, code)
   hist ← hist ∥ code ∥ Metadata(stdout)
   if state[Final] is set then
      return state[Final]
```
Clean, properly indented, complete fenced code block. Algorithm 2 is similarly well-extracted.

**Winner: Marker** — by a wide margin. Algorithms are clean and complete. SciPDF/GROBID essentially lost the algorithm blocks.

---

## 7. Table (Table 1 — Main Results)

**SciPDF:**
Table 1 data is extracted as a single blob inside a fenced code block in the Figures section:
```
ModelCodeQABrowseComp+ (1K) OOLONG OOLONG-PairsTask Length N (tokens)23K-4.2M6M-11M131K32KGPT-5...Base Model24.0 * ($0.13 ± $0.07)0.0 * (N/A)...
```
All data is present but crammed into one unstructured text block with no row/column separation. Unreadable as a table.

**Marker:**
```
| Model | CodeQA | BrowseComp+ (1K) | OOLONG | OOLONG-Pairs |
|-------|--------|-------------------|--------|--------------|
| Base Model | 24.0∗ | 0.0∗ | 44.0 | 0.1 |
| ($0.13 ± $0.07) | (N/A) ± (N/A) | ($0.14 ± $0.02) | ($0.16 ± $0.10) |
...
```
Proper multi-row markdown pipe table with headers and aligned columns. All values correct.

**Winner: Marker** — structured, readable pipe table. SciPDF extracted the data but without any tabular structure.

---

## 8. Figure Content

| Tool | Handling |
|------|----------|
| **SciPDF** | Figure captions extracted as separate objects: `**1** (figure): Figure1. A comparison of GPT-5...` with full caption text. 22 figure/table objects detected total, including labeled and unlabeled ones |
| **Marker** | Figure captions extracted as italic text: `*Figure 1.* A comparison of GPT-5...`. No figure content (images disabled at Tier 1) |

SciPDF extracts more figures (22 total including some unlabeled ones from the appendix) due to processing all 19 pages. However, many appear as empty entries (`**** (figure):` with no caption), suggesting GROBID detected figure regions but couldn't extract their text.

**Winner: SciPDF** — extracts more figure metadata as structured objects. But Marker's formatting is cleaner for the figures it does capture.

---

## 9. Citations and References

| Feature | SciPDF | Marker |
|---------|--------|--------|
| **Inline citations** | `(Hong et al., 2025)` — plain parenthetical, embedded in body text | `[\(Hong et al.,\)](#page-9-0) [\(2025\)](#page-9-0)` — hyperlinked with page anchors |
| **Cross-references** | `Figure 1`, `Table 1`, `§3` — plain text | `Figure [1](#page-0-0)`, `Table [1](#page-4-1)`, `[§3.](#page-3-0)` — clickable links |
| **Reference list** | Structured: `- Y Bai; S Tu; J Zhang... (2025). Longbench v2... **` — author; year; title; journal as separate fields | `<span id="page-8-4">` anchors + `[hyperlinked URLs](url)` |
| **Reference URLs** | None — no URLs in the reference list | Full URLs: `[https://arxiv.org/abs/2412.15204](url)` |

SciPDF's references are structurally richer (parsed into author/title/year/journal fields), but the markdown rendering has issues: many journal fields are empty (`**` at the end), and some references have garbled author lists (the DeepSeek-AI reference includes 100+ author names as a single entry).

**Winner: Marker** — hyperlinked citations, cross-references, and properly formatted reference URLs. SciPDF has structured reference data but poor markdown rendering and no hyperlinks.

---

## 10. Footnotes

| Tool | Handling |
|------|----------|
| **SciPDF** | Footnote text merged into body text, not distinguished. The footnote "This is key: it forces M to rely on..." appears inline without any footnote marker |
| **Marker** | `<sup>1</sup>This is key: it forces M to rely on...` with `<span id="page-2-1">` anchor |

**Winner: Marker** — superscript footnote numbers with anchor IDs. SciPDF loses footnote markers entirely.

---

## 11. Formulas

| Tool | Handling |
|------|----------|
| **SciPDF** | 1 formula extracted as a separate object with coordinates: `← LLM M (hist) if action is Finish then return val...` (coords: [3.0, 322.78, 325.52, 218.66, 85.18]) |
| **Marker** | No separate formula extraction — math notation is inline Unicode |

SciPDF extracted only 1 formula from the entire 19-page paper, and it's actually a garbled fragment of Algorithm 2. GROBID's formula detection was very poor on this paper.

**Winner: Tie** — neither tool handles formulas well. Marker has cleaner inline Unicode; SciPDF's formula extraction is broken.

---

## 12. Structured Data (SciPDF Advantage)

SciPDF outputs a structured JSON dict with programmatic access to:
- 41 sections as separate objects with `heading`, `text`, `publication_ref`, `figure_ref`, `table_ref`
- 41 references as separate objects with `ref_id`, `title`, `journal`, `year`, `authors`
- 22 figure/table objects with `figure_label`, `figure_type`, `figure_caption`, `figure_data`
- Cross-reference IDs linking citations to bibliography entries

Marker outputs flat markdown text — you'd need to parse it to extract this structure.

**Winner: SciPDF** — the structured dict format is fundamentally more useful for programmatic access, search indexing, and NLP pipelines.

---

## 13. Metadata

| Feature | SciPDF | Marker |
|---------|--------|--------|
| **Publication date** | `2026-01-28` | `*Preprint. January 29, 2026.*` (inline italic) |
| **DOI** | `10.1561/1500000019` | Not extracted |
| **Affiliations** | Not extracted separately | `<sup>1</sup>MIT CSAIL` inline |

**Winner: SciPDF** — extracts date and DOI as structured fields. Marker has the date as inline text only.

---

## 14. Appendix Content

| Tool | Handling |
|------|----------|
| **SciPDF** | Extracted all 19 pages including: Appendices A-F, training details, negative results, full system prompts for all methods, 20 OOLONG-Pairs benchmark tasks, RLM trajectory examples, CodeAct prompts |
| **Marker** | Only processed first 9 pages — no appendix content |

This is not a fair comparison since SciPDF processed the full PDF. However, it shows that GROBID successfully extracted the appendix content, though it misidentified benchmark tasks as paper sections.

**Winner: N/A** — different page ranges, not comparable.

---

## 15. Speed

| Tool | Time | Pages | Per-page |
|------|------|-------|----------|
| **SciPDF Tier 3** | 58.9s | 19 | ~3.1s/page |
| **Marker Tier 1** | ~2m 14s | 9 | ~14.9s/page |

**Winner: SciPDF** — approximately 5x faster per page. GROBID's CRF models (even via x86 emulation on ARM) are faster than Marker's Surya deep learning models.

---

## 16. Completeness

SciPDF extracted all body text from the full 19-page paper. Within the first 9 pages (overlapping content), both tools capture the complete body text. SciPDF additionally extracts DOI and structured reference metadata that Marker doesn't.

However, SciPDF missed: algorithm blocks, table structure, inline formatting, footnote markers, and most formulas.

**Winner: Tie** — SciPDF has more metadata; Marker has more formatting fidelity.

---

## Scorecard Summary

| Dimension | SciPDF Tier 3 | Marker Tier 1 |
|-----------|--------------|---------------|
| Title/Authors | 2nd (over-extracted authors) | 1st |
| Section Headings | 2nd (flat H2, spurious sections) | 1st |
| Body Text Quality | 2nd | 1st |
| Inline Formatting | 2nd | 1st |
| Math Notation | Tie | Tie |
| Algorithm Blocks | 2nd (lost) | 1st |
| Table Extraction | 2nd (unstructured blob) | 1st |
| Figure Content | 1st (more metadata) | 2nd |
| Citations/Links | 2nd | 1st |
| Footnotes | 2nd | 1st |
| Formulas | Tie | Tie |
| Structured Data | 1st | 2nd |
| Metadata (DOI/date) | 1st | 2nd |
| Speed (per page) | 1st | 2nd |
| Completeness | Tie | Tie |

**Wins:**
- **Marker Tier 1:** 8 wins
- **SciPDF Tier 3:** 4 wins
- **Tie:** 3

---

## Verdict

**Marker Tier 1** produces significantly more polished, readable markdown than **SciPDF Tier 3** for this arXiv paper. Marker wins on every formatting dimension: headings, inline emphasis, algorithms, tables, citations, and footnotes. For human reading or document review, Marker is the clear choice.

**SciPDF Tier 3** has a fundamentally different strength: **structured data**. Its output is a JSON dictionary with sections, references, and figures as separate programmatic objects with cross-reference IDs. This makes it far better for search indexing, NLP pipelines, citation analysis, and automated paper processing. It's also ~5x faster per page.

### The Fundamental Tradeoff

These tools solve different problems:
- **Marker** produces a document you can *read* — formatted markdown that looks like the original paper
- **SciPDF** produces data you can *query* — structured objects you can programmatically access

SciPDF's GROBID backend was not designed for markdown rendering. It excels at bibliographic extraction (41 parsed references with author/title/year/journal fields) and document structure detection, but it loses formatting, algorithms, and table structure in the process.

### Recommendation

| Use Case | Best Choice |
|----------|-------------|
| **Reading/reviewing a paper** | Marker Tier 1 |
| **Building a paper search index** | SciPDF Tier 3 |
| **Citation analysis / bibliometrics** | SciPDF Tier 3 |
| **Extracting specific sections programmatically** | SciPDF Tier 3 |
| **Papers with complex tables/algorithms** | Marker Tier 1 |
| **Speed-sensitive batch processing** | SciPDF Tier 3 (~3s/page vs ~15s/page) |
| **Preserving formatting fidelity** | Marker Tier 1 |
