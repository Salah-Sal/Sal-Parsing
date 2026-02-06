# MarkItDown Tier 1 vs Marker Tier 1 — Comparison Report

**Paper:** "Recursive Language Models" (arXiv: 2512.24601v2)
**Pages parsed:** 0-8 (9 pages)
**Date:** 2026-02-05

| Tool | Version | Engine | Time |
|------|---------|--------|------|
| **MarkItDown** (Microsoft) | 0.1.5b1 | pdfminer.six + pdfplumber | ~1.5 seconds |
| **Marker** (Datalab) | 1.10.1 | Surya layout/detection models (MPS) | ~2 min 14s |

---

## 1. High-Level Summary

| Metric | MarkItDown Tier 1 | Marker Tier 1 |
|--------|-------------------|---------------|
| **Time** | ~1.5s | ~2m 14s |
| **Speed ratio** | 1x (baseline) | ~90x slower |
| **Output size** | 40 KB | 46 KB |
| **Lines of markdown** | 1,027 | 263 |
| **Images extracted** | 0 | 0 |
| **Markdown headings** | None (plain text) | Yes (`#`, `##`, `####`) |
| **Formatting** | None | Rich (italic, bold, links, anchors) |
| **Tables** | Plain text dump | Proper markdown pipe tables |
| **Citations** | Plain text | Hyperlinked with anchors |

---

## 2. Title and Metadata

### MarkItDown
- Title: `Recursive Language Models` (plain text, no heading markup)
- Authors: `Alex L. Zhang 1 Tim Kraska 1 Omar Khattab 1` (superscript numbers as plain text)
- The arXiv sidebar stamp is extracted **character-by-character** as vertical text, producing 40 lines of garbage:
  ```
  6
  2
  0
  2

  n
  a
  J

  8
  2
  ...
  ```
  This is the rotated arXiv ID watermark (`arXiv:2512.24601v2 [cs.AI] 28 Jan 2026`) broken into individual characters.

### Marker
- Title: `# Recursive Language Models` (proper H1 heading)
- Authors: `## Alex L. Zhang <sup>1</sup> Tim Kraska <sup>1</sup> Omar Khattab <sup>1</sup>` (H2 with HTML superscripts)
- The arXiv sidebar stamp is **not extracted** — Marker's layout model correctly identifies and skips it.

### Verdict
**Marker wins decisively.** MarkItDown's extraction of the arXiv sidebar as character-per-line garbage is a significant flaw. Marker cleanly suppresses it and produces proper heading markup.

---

## 3. Document Structure & Headings

### MarkItDown
- **No markdown headings at all.** Every section title is plain text:
  ```
  Abstract
  ...
  1. Introduction
  ...
  2. Recursive Language Models
  ...
  3. Scaling Long Context Tasks
  ...
  3.1. Tasks
  ```
- There is no structural distinction between a section heading and a regular paragraph — they're all just lines of text.
- Page numbers appear inline as isolated numbers (`1`, `2`, `3`, etc.) and the running header `Recursive Language Models` repeats on every page, mixed into body text.

### Marker
- Proper markdown heading hierarchy:
  - `# Recursive Language Models` (H1 — title)
  - `## Abstract`, `## 1. Introduction`, `## 2. Recursive Language Models` (H2 — sections)
  - `#### 3.1. Tasks` (H4 — subsections)
  - `### 3.2. Methods and Baselines` (H3)
- Page numbers and running headers are **suppressed** — they don't appear in the output.

### Verdict
**Marker wins decisively.** Without heading markup, MarkItDown's output is just a flat wall of text. Marker produces navigable document structure with proper hierarchy.

---

## 4. Line Count Paradox (1,027 vs 263)

MarkItDown produces **4x more lines** despite the same content. Why?

1. **No paragraph joining.** MarkItDown preserves the PDF's internal line breaks, so every line in the two-column layout becomes a separate line in the output. For example:
   ```
   Frontier reasoning models have limited context windows
   tend to exhibit context
   and, even within their limits,
   rot (Hong et al., 2025), a phenomenon illustrated in Fig-
   ure 1 where quality degrades steeply as prompts get longer.
   ```
   Marker joins these into a single flowing paragraph:
   ```
   Frontier reasoning models have limited context windows and, even within
   their limits, tend to exhibit *context rot* (Hong et al., 2025), a phenomenon
   illustrated in Figure 1 where quality degrades steeply as prompts get longer.
   ```

2. **The arXiv sidebar** adds ~40 junk lines in MarkItDown.

3. **Page numbers and running headers** add repeated lines (e.g., `Recursive Language Models` appears 9 times).

4. **Table data** is spread across many lines instead of compact pipe tables.

### Verdict
**Marker wins.** Fewer lines but denser, cleaner text. MarkItDown's raw line breaks make the output harder to consume downstream.

---

## 5. Two-Column Layout Handling

This is one of the most critical differences for arXiv papers.

### MarkItDown
- pdfminer extracts text in reading order but **does not always merge two-column text correctly**. Most paragraphs flow properly, but some columns bleed into each other.
- Hyphenated line breaks are preserved: `Re-\ncursive` stays as `Re-` on one line and `cursive` on the next, rather than being rejoined to `Recursive`.
- Mid-sentence line breaks from the PDF columns are preserved verbatim.

### Marker
- Surya's layout detection model correctly identifies two-column structure and **merges text into flowing paragraphs**.
- Hyphenated words at column breaks are rejoined (e.g., `long-\ncontext` → `longcontext` or kept as `long-context` appropriately).
- Reading order is correct across column boundaries.

### Verdict
**Marker wins.** Proper column detection is essential for academic papers. MarkItDown's line-level extraction leaves fragments that break downstream NLP.

---

## 6. Mathematics

### MarkItDown
- Unicode symbols extracted directly from PDF text:
  - `P ∈ Σ⋆` (correct Unicode)
  - `Ω(|P|)` and `Ω(|P|2)` (superscript missing — just `2`)
  - `score(ˆy) = 0.75|y−ˆy|` (no superscript — exponent is inline)
- No HTML tags, no LaTeX — everything is plain Unicode.
- Generally readable but loses superscripts/subscripts.

### Marker
- Same Unicode extraction from embedded PDF text, plus HTML for some formatting:
  - `P ∈ Σ ⋆` (Unicode with extra spaces)
  - `Ω(|P|)` and `Ω(|P| 2 )` (superscript missing — plain `2`)
  - `score(ˆy) = 0.75<sup>|</sup>y−yˆ<sup>|</sup>` (attempts HTML superscript but garbles it)
  - `2 <sup>13</sup> to 2 <sup>18</sup>` (HTML superscript tags for exponents)

### Verdict
**Roughly tied, both poor.** Neither tool produces LaTeX math (that requires Marker Tier 4 or MarkItDown Tier 4/Azure). MarkItDown's plain Unicode is slightly cleaner — Marker's attempted HTML superscripts sometimes garble the expressions (the scoring function is worse in Marker). Both lose superscript information on inline math like `Ω(|P|²)`.

---

## 7. Table 1 (Main Results Table)

This is the most data-dense element in the paper.

### MarkItDown
- **No markdown table structure.** The table is extracted as blocks of plain text:
  ```
  Model

  CodeQA

  BrowseComp+ (1K) OOLONG OOLONG-Pairs

  Task Length N (tokens)

  23K-4.2M

  6M-11M

  131K

  32K

  GPT-5 (with RLM sub-calls to GPT-5-mini)

  Base Model
  CodeAct (+ BM25)
  ...
  ($0.13 ± $0.07)
  ...
  24.0∗
  22.0∗
  ```
- Values are present but **not associated with their column headers** — it's impossible to tell which number belongs to which benchmark without cross-referencing the original PDF.
- The layout is a vertical dump of cell values, roughly in order, but a machine or human cannot reconstruct the table structure.

### Marker
- **Proper markdown pipe table** with headers, separator, and aligned cells:
  ```markdown
  | Model | CodeQA | BrowseComp+ (1K) | OOLONG | OOLONG-Pairs |
  |-------|--------|-------------------|--------|--------------|
  | Base Model | 24.0∗ | 0.0∗ | 44.0 | 0.1 |
  | | (\$0.13 ± \$0.07) | (N/A) ± (N/A) | ... | ... |
  ```
- All cells populated with correct values.
- Multi-line cells use `<br>` tags for sub-rows (cost underneath score).
- Column alignment is preserved.

### Verdict
**Marker wins decisively.** This is arguably the most important difference. MarkItDown's table output is essentially unusable — a flat dump of values with no structure. Marker produces a complete, parseable markdown table with all data correctly placed in cells.

---

## 8. Algorithm Pseudocode

### MarkItDown
- Algorithms 1 and 2 are extracted as plain text with correct content:
  ```
  Algorithm 1 A recursive language model, around LLM M
  Input: prompt P
  Output: response Y
  state ← InitREPL(prompt=P)
  state ← AddFunction(state, sub_RLMM)
  hist ← [Metadata(state)]
  while True do
  ```
- All lines present including the body of both algorithms.
- Unicode arrows (←) preserved.
- No code block formatting — just plain text paragraphs.

### Marker
- Both algorithms rendered inside **fenced code blocks** (` ``` `):
  ```
  Algorithm 1 A recursive language model, around LLM M
  Input: prompt P
  Output: response Y
  state ← InitREPL(prompt=P)
  ...
  ```
- Algorithm 2 is **complete** with all lines including the loop body.
- Flaw comments (`// Flaw #1`, `// Flaw #2`, `// Flaw #3`) preserved.

### Verdict
**Marker wins slightly.** Both extract the full algorithm text, but Marker wraps them in code blocks making them visually distinct from body text. The content is identical.

---

## 9. Citations and References

### MarkItDown
- Inline citations are plain text: `(Hong et al., 2025)`
- Reference list entries are plain text paragraphs.
- URLs are plain text strings: `https://arxiv.org/abs/2412.15204`
- No hyperlinks, no anchors, no internal cross-referencing.

### Marker
- Inline citations are **hyperlinked** with HTML anchors:
  `[\(Hong et al.,](#page-9-0) [2025\)](#page-9-0)`
- Reference entries have `<span id="page-X-X">` anchors.
- URLs are **clickable markdown links**:
  `[https://arxiv](https://arxiv.org/abs/2412.15204).org/abs/2412.15204`
- Internal cross-references work: clicking a citation jumps to the reference.

### Verdict
**Marker wins.** Hyperlinked citations and references are a major usability advantage for navigating academic papers.

---

## 10. Text Formatting and Emphasis

### MarkItDown
- **No formatting whatsoever.** Everything is plain text:
  - Italicized terms in the PDF (`context rot`, `environment`) → plain text
  - Bold text → plain text
  - All emphasis is lost

### Marker
- Italics preserved: `*context rot*`, `*environment*`, `*programmatically*`
- Bold rarely used but preserved where detected
- Conference names italicized: `*The Twelfth International Conference on Learning Representations*`

### Verdict
**Marker wins.** Emphasis is part of the document's meaning (e.g., defined terms are italicized in academic writing). MarkItDown strips all formatting.

---

## 11. Figures and Figure Captions

### MarkItDown
- Figure captions extracted as plain text paragraphs
- **Figure chart data is extracted as garbled axis tick labels:**
  ```
  8k16k33k66k131k262k524k1M020406080100Score (%)GPT-5OOLONGOOLONG-PairsS-NIAH8k16k33k66k131k262k524k1MInput Context Length (log scale)020406080100Score (%)RLM(GPT-5)OOLONGOOLONG-PairsS-NIAH
  ```
  This is pdfminer extracting the text layer from the chart SVG/image — axis labels, legend entries, and data all concatenated into a single unreadable string.

### Marker
- Figure captions preserved with span anchors and italic formatting:
  `<span id="page-0-0"></span>*Figure 1.* A comparison of GPT-5 and...`
- Figure chart data is **also extracted as garbled text** (same issue — both tools extract embedded text from figures).
- No images in either (both Tier 1 configs disabled image extraction).

### Verdict
**Tie on content, Marker wins on formatting.** Both extract garbled chart text. But Marker at least formats the captions with proper figure labels and anchors.

---

## 12. Footnotes

### MarkItDown
- Footnotes are extracted inline as part of the text flow.
- Footnote markers (superscript numbers) appear as plain numbers mixed into text.
- Footnote text appears wherever pdfminer encounters it (often at the bottom of a column, mid-paragraph in the output).

### Marker
- Footnote markers use HTML: `<sup>1</sup>`, `<sup>2</sup>`
- Footnote text is wrapped in `<span>` tags with IDs.
- Cross-referencing between marker and footnote body is possible.

### Verdict
**Marker wins.** Footnotes are properly marked and linkable.

---

## 13. Page Boundaries and Running Headers

### MarkItDown
- **Page numbers appear inline** as isolated numbers: `1`, `2`, `3`, etc.
- The running header `Recursive Language Models` repeats 9 times mixed into the body text.
- No indication of where pages break.

### Marker
- Page numbers and running headers are **suppressed** — they don't pollute the body text.
- No page boundary markers (Tier 1 doesn't use `paginate_output`).

### Verdict
**Marker wins.** MarkItDown's inline page numbers and repeated headers create noise in the text.

---

## 14. Text Accuracy

### MarkItDown
- Author email correct: `altzhang@mit.edu` (embedded in text but not separately formatted)
- Author affiliations: `1MIT CSAIL, Cambridge, MA, USA`
- All body text accurately extracted from embedded PDF text.
- No OCR errors (no OCR used).

### Marker
- Author email correct: `altzhang@mit.edu`
- Author affiliations in footnote format with proper anchoring.
- All body text accurately extracted.
- No OCR errors.

### Verdict
**Tie.** Both extract embedded PDF text accurately. Neither introduces errors.

---

## 15. Downstream Usability

Consider common use cases for parsed papers:

| Use Case | MarkItDown | Marker |
|----------|-----------|--------|
| Feed to LLM for Q&A | Usable but noisy (junk lines, no structure) | Clean, structured, ready to use |
| Extract table data | Impossible (unstructured dump) | Possible (markdown table) |
| Build citation graph | Manual only (plain text) | Parseable (hyperlinked references) |
| Search/index content | Works (text is all there) | Works (cleaner text) |
| Render as HTML | Poor (no headings, no formatting) | Good (proper markdown renders well) |
| Bulk processing pipeline | Excellent speed (~1.5s/paper) | Slow (~2m 14s/paper) |

---

## 16. Overall Scoring

| Dimension | MarkItDown Tier 1 | Marker Tier 1 | Winner |
|-----------|-------------------|---------------|--------|
| **Speed** | ~1.5s | ~2m 14s | MarkItDown (90x faster) |
| **Heading structure** | None (plain text) | Proper markdown | Marker |
| **Paragraph flow** | Raw line breaks | Joined paragraphs | Marker |
| **Two-column handling** | Basic (some bleeding) | Proper layout detection | Marker |
| **Table extraction** | Unstructured dump | Proper pipe tables | Marker |
| **Math** | Plain Unicode | Unicode + HTML sup | Tie (both poor) |
| **Algorithm pseudocode** | Plain text (complete) | Code blocks (complete) | Marker (slightly) |
| **Citations** | Plain text | Hyperlinked | Marker |
| **References** | Plain text, no links | Linked with anchors | Marker |
| **Text formatting** | None | Italic/bold preserved | Marker |
| **Figure handling** | Garbled chart text | Garbled + formatted captions | Marker (slightly) |
| **Page noise** | Page numbers + headers in text | Suppressed | Marker |
| **arXiv sidebar** | 40 lines of character garbage | Correctly suppressed | Marker |
| **Text accuracy** | Accurate | Accurate | Tie |
| **Dependencies** | pdfminer + pdfplumber only | PyTorch + Surya models | MarkItDown (lighter) |
| **GPU required** | No | No (but uses MPS when available) | MarkItDown |

**Score: Marker 11, MarkItDown 3, Tie 2**

---

## 17. Final Assessment

**MarkItDown Tier 1** is the right choice when:
- You need **raw speed** — 90x faster means processing hundreds of papers in the time Marker does one
- You only need **approximate text content** for keyword search or full-text indexing
- You're in a **minimal environment** without GPU or PyTorch
- The downstream consumer (e.g., an LLM) can handle noisy, unstructured text
- You plan to do your own post-processing on the raw text

**Marker Tier 1** is the right choice when:
- You need **structured markdown** with headings, tables, and formatting
- **Table data** must be extractable and machine-readable
- **Citation navigation** matters
- The output will be **rendered** or displayed to humans
- You need correct **two-column layout** merging for academic papers
- Processing time is acceptable (~2 min per paper on Apple M2 Pro)

**Bottom line:** For arXiv paper parsing, Marker Tier 1 produces dramatically better structured output at the cost of ~90x longer processing time. MarkItDown is a text dump tool — useful for speed but not for structure. If you need both speed and structure, consider using MarkItDown for bulk filtering/search and Marker for the papers you actually need to read carefully.
