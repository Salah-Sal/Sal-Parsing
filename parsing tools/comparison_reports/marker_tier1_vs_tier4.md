# Tier 1 vs Tier 4 Parsing Comparison Report

**Paper:** "Recursive Language Models" (arXiv: 2512.24601v2)
**Pages parsed:** 0-8 (9 pages)
**Date:** 2026-02-05

### Machine Specs

| Component | Details |
|-----------|---------|
| **Chip** | Apple M2 Pro |
| **CPU** | 12 cores (8 performance + 4 efficiency), arm64 |
| **GPU** | 19-core integrated GPU, Metal 4 |
| **RAM** | 16 GB unified memory |
| **OS** | macOS 26.2 (Build 25C56) |
| **PyTorch** | 2.9.0 |
| **Compute device** | MPS (Metal Performance Shaders) |
| **Marker version** | 1.10.1 |

> Note: The M2 Pro's table recognition model falls back to CPU (`TableRecEncoderDecoderModel` is not compatible with MPS). All other models run on MPS.

---

## 1. High-Level Summary

| Metric | Tier 1 (Fastest) | Tier 4 (High Quality) |
|--------|-------------------|-----------------------|
| **Time** | ~2 min 14s | ~17 min 1s |
| **Speed ratio** | 1x (baseline) | ~7.6x slower |
| **Output size** | 46 KB (.md only) | 44 KB (.md) + 405 KB (3 images) |
| **Lines of markdown** | 263 | 338 |
| **Images extracted** | 0 | 3 (JPEG) |
| **OCR regions processed** | 0 | 274 (text) + 81 (tables) |
| **Page separators** | None | Yes (`{0}---`, `{1}---`, etc.) |

---

## 2. Document Structure & Headings

### Tier 1
- Title: `# Recursive Language Models`
- All major sections use `##` (e.g., `## Abstract`, `## 1. Introduction`)
- Subsections use `####` (e.g., `#### 3.1. Tasks`)
- Consistent, flat heading hierarchy

### Tier 4
- Title: `# **Recursive Language Models**` (bold inside heading)
- Major sections inconsistently use `#`, `###`, or `####` (e.g., `# **Abstract**`, `### 1. Introduction`, `# 2. Recursive Language Models`)
- Subsections use `####` with bold: `#### **3.1.** Tasks`
- Heading hierarchy is less consistent — OCR re-interprets visual font sizes

### Verdict
**Tier 1 wins on heading consistency.** Because Tier 1 extracts embedded PDF text directly, the heading hierarchy is more uniform. Tier 4's forced OCR re-interprets headings from visual appearance, leading to inconsistent levels (mixing `#`, `###`, `####` for same-level sections).

---

## 3. Figures and Images

### Tier 1
- No images extracted (disabled via `--disable_image_extraction`)
- Figure captions preserved as text paragraphs
- Figures referenced with HTML anchors: `<span id="page-0-0"></span>*Figure 1.* A comparison of...`
- Figure 4 (code trajectory examples): **Not rendered** — only the caption text exists

### Tier 4
- 3 images extracted as JPEG files:
  - `_page_0_Figure_11.jpeg` (88 KB) — Figure 1 (performance plots)
  - `_page_1_Figure_1.jpeg` (210 KB) — Figure 2 (RLM architecture diagram)
  - `_page_5_Figure_1.jpeg` (107 KB) — Figure 3 (cost comparison plots)
- Images referenced with markdown syntax: `![](_page_0_Figure_11.jpeg)`
- Figure 4 (code trajectory examples): **Fully rendered** as code blocks with actual Python code visible

### Verdict
**Tier 4 wins decisively.** The extracted figures are essential for understanding the paper (performance charts, architecture diagram). Tier 4 also captured Figure 4's code snippets as proper code blocks, which Tier 1 missed entirely.

---

## 4. Mathematics

This is one of the most significant differences between the two tiers.

### Tier 1 (PDF embedded text extraction)
Uses Unicode characters and HTML `<sup>` tags:
- `P ∈ Σ ⋆` (Unicode symbols, extra space around star)
- `|P| ≫ K` (Unicode much-greater-than)
- `Ω(|P|)` and `Ω(|P| 2)` (superscript missing — just plain "2")
- `2 <sup>13</sup> to 2 <sup>18</sup>` (HTML superscript tags)
- `score(ˆy) = 0.75<sup>|</sup>y−yˆ<sup>|</sup>` (garbled — hard to read)

### Tier 4 (OCR with Surya models)
Uses LaTeX `$...$` notation:
- `$P \in \Sigma^*$` (proper LaTeX)
- `$|P| \gg K$` (proper LaTeX)
- `$\Omega(|P|)$` and `$\Omega(|P|^2)$` (correct superscript)
- `$2^{13}$` to `$2^{18}$` (proper LaTeX exponents)
- `$\mathtt{score}(\hat{y}) = 0.75^{|y-\hat{y}|}$` (clean, correct LaTeX)

### Specific Examples

| Expression | Tier 1 | Tier 4 |
|-----------|--------|--------|
| Set membership | `P ∈ Σ ⋆` | `$P \in \Sigma^*$` |
| Much greater than | `\|P\| ≫ K` | `$\|P\| \gg K$` |
| Big-Omega quadratic | `Ω(\|P\| 2)` | `$\Omega(\|P\|^2)$` |
| Scoring function | `score(ˆy) = 0.75<sup>\|</sup>y−yˆ<sup>\|</sup>` | `$\mathtt{score}(\hat{y}) = 0.75^{\|y-\hat{y}\|}$` |
| Model M | `M` (plain text) | `$\mathcal{M}$` (calligraphic) |
| Environment E | `E` (plain text) | `$\mathcal{E}$` (calligraphic) |
| 2x performance | `2×` | `$2\times$` |
| 3x cheaper | `3×` | `$3\times$` |

### Verdict
**Tier 4 wins decisively.** LaTeX output is standard for academic papers and renders correctly in any markdown viewer with math support. Tier 1's Unicode/HTML mix is inconsistent and sometimes garbled (especially the scoring function). The calligraphic `\mathcal{M}` for the model variable is only captured by Tier 4.

---

## 5. Tables

Both tiers successfully extracted Table 1 (the main results table), but with notable differences.

### Tier 1
- Table structure preserved with pipes
- Multi-line cells use `<br>` tags: `Task Length<br>N<br>(tokens)`
- Cost values use `\$` escaping: `(\$0.13 ± \$0.07)`
- All cells populated with values
- No bold marking for best results

### Tier 4
- Table structure preserved with pipes
- Multi-line cells use LaTeX: `Task Length $N$ (tokens)`
- Cost values wrapped in LaTeX: `$0.0^*$ (N/A) $\pm$ (N/A)`
- **Some cells are empty** where Tier 1 has values (e.g., OOLONG-Pairs for "Summary agent" and "RLM (no sub-calls)" rows)
- Best results marked with `<b>` bold tags

### Specific Missing Data in Tier 4
These cells have values in Tier 1 but are empty in Tier 4:
- Summary agent / OOLONG-Pairs (GPT-5): `0.1` missing
- RLM (no sub-calls) / OOLONG-Pairs (GPT-5): `43.9` missing
- CodeAct (+ BM25) / OOLONG-Pairs (Qwen3-Coder): `0.3` missing
- RLM / OOLONG-Pairs (Qwen3-Coder): `23.1` missing
- RLM (no sub-calls) / OOLONG-Pairs (Qwen3-Coder): `17.3` missing
- RLM / OOLONG-Pairs (Qwen3-8B): `4.3` missing
- RLM (fine-tuned) / OOLONG-Pairs (Qwen3-8B): `5.2` missing

### Verdict
**Mixed.** Tier 4 has better formatting (bold best results, LaTeX math in cells) but **drops data from several table cells**. This is a significant accuracy issue — missing values in the rightmost column. Tier 1 preserves all table data. For data completeness, Tier 1 is more reliable here; for rendering quality, Tier 4 is better.

---

## 6. Algorithm Pseudocode

### Tier 1
- Both Algorithm 1 and Algorithm 2 rendered as code blocks
- Algorithm 2 is **complete** — all lines including the body of the while loop:
  ```
  (action, val) ← LLMM(hist)
  if action is Finish then
     return val // Flaw #2
  out ← RUN(action, val) // Flaw #3
  hist ← hist ∥ (action, val, out)
  if Tok(hist) > K then
     hist ← Compact(hist)
  ```

### Tier 4
- Both algorithms rendered as code blocks
- Algorithm 2 is **truncated** — cuts off after `while True do`:
  ```
  actions \leftarrow {Finish, Exec, Search, sub_LLM_M}
  hist \leftarrow [Metadata(actions), P] // Flaw #1
  while True do
  ```
  The body of the loop (Flaws #2 and #3) is completely missing.

### Verdict
**Tier 1 wins.** Algorithm 2's body is critical for understanding the paper's argument about three design flaws. Tier 4's OCR missed the indented body of the algorithm, likely because forced OCR at high DPI re-interpreted the layout and lost the continuation.

---

## 7. Citations and References

### Tier 1
- Inline citations are hyperlinked with HTML anchors:
  `[\(Hong et al.,](#page-9-0) [2025\)](#page-9-0)`
- Reference list entries have `<span id="page-X-X">` anchors enabling in-document linking
- URLs in references are clickable markdown links:
  `[https://arxiv](https://arxiv.org/abs/2412.15204).org/abs/2412.15204`

### Tier 4
- Inline citations are plain text: `(Hong et al., 2025)`
- No anchor tags — no in-document linking
- URLs in references are plain text (not clickable):
  `https://arxiv.org/abs/2412.15204`

### Verdict
**Tier 1 wins for interactivity.** Hyperlinked citations and clickable URLs are valuable for navigating the document. Tier 4's plain text citations are cleaner to read but lose navigability.

---

## 8. Text Formatting and Emphasis

### Tier 1
- Italics: `*context rot*`, `*environment*` — consistently preserved
- Bold: Rarely used
- Model names: Plain text (`RLM-Qwen3-8B`)
- Key terms: Italicized

### Tier 4
- Italics: `*context rot*`, `*environment*` — consistently preserved
- Bold: Used more aggressively (`**Re**cursive Language Models`, `**RLM-Qwen3-8B**`, `**S-NIAH**`)
- Model names: Often bolded
- Key terms: Mix of bold and italic

### Verdict
**Tier 4 slightly better.** The bolding of key terms like benchmark names (`**S-NIAH**`, `**OOLONG**`) and model names matches the original PDF's visual emphasis more closely.

---

## 9. OCR Errors and Text Accuracy

### Tier 1 (no OCR — uses embedded PDF text)
- Author email correct: `altzhang@mit.edu`
- GitHub URL split but linked: `[https:](https://github.com/alexzhang13/rlm) //github.[com/alexzhang13/rlm](...)`
- "OOLONG-Pairs" section label present and complete
- No OCR-introduced errors

### Tier 4 (forced OCR at 300 DPI)
- Author email has OCR error: `altrhang@mit.edu` (should be `altzhang`)
- GitHub URL split and unlinked: `https:` / `//github.com/alexzhang13/rlm.`
- "OOLONG-Pairs" section label **missing** — paragraph starts with "OOLONG to include 20 new queries..." instead of "**OOLONG-Pairs.** We modify the trec\_coarse split of OOLONG to include..."
- HTML entity artifacts: `<sup>&amp;lt;sup>` in footnotes

### Verdict
**Tier 1 wins on text accuracy for clean digital PDFs.** Embedded PDF text is more reliable than OCR for well-formed digital documents. Tier 4's forced OCR introduced a misspelling in an email address and dropped a section label.

---

## 10. Page Structure

### Tier 1
- Continuous document — no page breaks
- No indication of where page boundaries fall

### Tier 4
- Page separators: `{0}---`, `{1}---`, ..., `{8}---`
- Clear horizontal rules between pages
- Useful for referencing back to the original PDF

### Verdict
**Tier 4 wins.** Page markers are helpful for cross-referencing with the original PDF, especially in academic contexts.

---

## 11. Content from Figure 4 (Code Trajectory Examples)

This is a uniquely interesting case. Figure 4 shows code snippets from RLM trajectories.

### Tier 1
- Only the figure caption is preserved as text
- The actual code snippets within the figure are completely absent
- A reader would have no idea what the code examples look like

### Tier 4
- Three code blocks fully extracted with actual Python code:
  - (a) Keyword search/regex filtering code
  - (b) Batch LLM query classification code
  - (c) Final result stitching code with `FINAL_VAR()`
- Sub-captions for each code block preserved
- This is arguably some of the most interesting content in the paper

### Verdict
**Tier 4 wins decisively.** The code examples in Figure 4 are essential for understanding the paper's core claims about emergent RLM behavior. Tier 1 loses this content entirely.

---

## 12. Overall Quality Scoring (for this arXiv paper)

| Dimension | Tier 1 | Tier 4 | Winner |
|-----------|--------|--------|--------|
| **Speed** | 2m 14s | 17m 1s | Tier 1 |
| **Heading consistency** | Uniform `##` hierarchy | Mixed `#`/`###`/`####` | Tier 1 |
| **Figures** | None | 3 JPEG images | Tier 4 |
| **Math rendering** | Unicode/garbled | Clean LaTeX | Tier 4 |
| **Table completeness** | All cells present | 7+ cells missing | Tier 1 |
| **Table formatting** | Plain | Bold best, LaTeX cells | Tier 4 |
| **Algorithm pseudocode** | Complete | Algorithm 2 truncated | Tier 1 |
| **Citations** | Hyperlinked | Plain text | Tier 1 |
| **Text accuracy** | No errors | OCR typo in email, dropped label | Tier 1 |
| **Bold/emphasis** | Minimal | Matches PDF visual style | Tier 4 |
| **Page markers** | None | Yes | Tier 4 |
| **Figure 4 code** | Missing | Fully extracted | Tier 4 |
| **Cost** | Free | Free | Tie |

### Final Assessment

**Tier 1** is better when:
- You need the raw text content quickly and accurately
- Table data completeness matters (e.g., for data extraction)
- You want working hyperlinks in citations
- The PDF is clean digital text (not scanned)
- You trust embedded PDF text over OCR

**Tier 4** is better when:
- You need figures/images (charts, diagrams)
- LaTeX math rendering matters (academic/research use)
- You need content from inside figures (like code examples)
- Page-level structure is important
- Visual emphasis (bold, formatting) should match the original

**Neither tier is perfect.** The ideal approach for this paper would be:
- **Tier 5 (LLM-enhanced)** to get the best of both worlds — accurate tables with no missing cells, proper math, extracted images, and correct OCR
- Alternatively, use **Tier 1 for text + Tier 4 for images**, and merge the results

---

## 13. Recommendations for arXiv Paper Parsing

1. **If you only need body text and references:** Tier 1 or Tier 3 is sufficient and 7x faster.
2. **If you need math in LaTeX format:** Tier 4 minimum, Tier 5 recommended.
3. **If you need figures:** Tier 3+ (don't disable image extraction).
4. **If you need complete tables:** Do NOT use `--force_ocr` on clean digital PDFs — the embedded text (Tier 1/3) preserves table data more reliably.
5. **If you need everything:** Use Tier 5 with `--use_llm` for LLM-corrected tables, equations, and images.
6. **Always spot-check:** Both tiers can drop content. Compare critical sections against the original PDF.
