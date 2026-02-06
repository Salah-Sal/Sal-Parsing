# Marker Tier 1 vs Docling Tier 4

**Paper:** "Recursive Language Models" (2512.24601v2), first 9 pages
**Machine:** Apple M2 Pro, 16GB unified RAM, macOS

This comparison pits Marker's fastest tier against Docling's enhanced pipeline to see whether Docling's heavier processing (Egret Large layout + OCR + accurate tables + formula/code enrichment) can outperform Marker's lightweight Surya-based extraction.

---

## Overview

| Metric | Marker Tier 1 | Docling Tier 4 |
|--------|---------------|----------------|
| **Tool** | Datalab Marker 1.10.1 | IBM Docling 2.72.0 |
| **Method** | Surya layout detection, no OCR, no image extraction | Egret Large layout + auto OCR + accurate TableFormer + formula/code enrichment |
| **Time** | ~2m 14s | 52.4s |
| **Output size** | 46,282 chars (263 lines) | 46,440 chars (412 lines) |

---

## 1. Title and Author Extraction

| Tool | Result |
|------|--------|
| **Marker** | `# Recursive Language Models` (H1) then `## Alex L. Zhang <sup>1</sup> Tim Kraska <sup>1</sup> Omar Khattab <sup>1</sup>` — proper heading hierarchy, superscript affiliations |
| **Docling** | `## Recursive Language Models` (H2) then `Alex L. Zhang 1 Tim Kraska 1 Omar Khattab 1` — flat H2, raw affiliation numbers, no superscripts |

**Winner: Marker** — proper H1/H2 hierarchy and superscript affiliation markers.

---

## 2. Section Headings

| Tool | Hierarchy | Subsections |
|------|-----------|-------------|
| **Marker** | H1 title → H2 sections → H4 subsections (`#### 3.1. Tasks`) | Yes — 3.1, 3.2, 4.1 all detected |
| **Docling** | H2 for everything (title, sections, subsections all `##`) | Yes — 3.1, 3.2, 4.1 detected but same level as parent sections |

**Winner: Marker** — multi-level heading hierarchy. Docling flattens everything to H2.

---

## 3. Body Text Quality

Both tools produce clean, well-merged paragraph text with proper reading order across the two-column layout. No mid-sentence line breaks in either.

One difference: Docling adds occasional extra whitespace around punctuation and terms (e.g., `RLM s` instead of `RLMs`, `28 . 3%` instead of `28.3%`, `Ω( | P | )` instead of `Ω(|P|)`). Marker has tighter spacing.

**Winner: Marker** — slightly cleaner text with no spurious spaces.

---

## 4. Inline Formatting

| Feature | Marker | Docling |
|---------|--------|---------|
| **Italic** | Yes — `*context rot*`, `*environment*`, `*programmatically*`, `*reasoning models*` | None |
| **Bold** | None | None |
| **Superscript** | Yes — `<sup>1</sup>`, `2 <sup>13</sup>` | None |

**Winner: Marker** — captures italic emphasis and superscript notation from the PDF. Docling Tier 4 produces plain text despite having more models enabled.

---

## 5. Mathematical Notation

| Tool | Inline math | Quality |
|------|-------------|---------|
| **Marker** | `P ∈ Σ ⋆`, `\|P\| ≫ K`, `Ω(\|P\|)`, `Ω(\|P\| 2)` | Clean Unicode, tight spacing |
| **Docling** | `P ∈ Σ ⋆`, `\| P \|≫ K`, `Ω( \| P \| )`, `Ω( \| P \| 2 )` | Unicode preserved but extra spaces around symbols |

Neither tool produces LaTeX at these tiers. Both use Unicode math symbols.

**Winner: Marker** — tighter, more readable math notation. Docling's formula enrichment didn't convert inline math to LaTeX for this paper.

---

## 6. Algorithm/Pseudocode Blocks

This is a critical test — the paper has two algorithm blocks (Algorithm 1 and Algorithm 2).

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
Clean, properly indented, all lines correct.

**Docling:**
```
=algorithm 1 A recursive language model, around LLM M
		=Input: prompt P
		Output: response Y
	) is
		state <- InitREPL (prompt=P)
		state <- AddFunction (state, sub_RLM.M)
		hist <- [Metadata (state)]
		while True do
		C*, an
		&
		and
		(state, stdout) <- REPL (state,  code)
		unded
		g. the
		[- return state[Final] is then
			[- return state[Final]]
```
Garbled — contains OCR noise (`C*, an`, `& `, `unded`, `g. the`), corrupted lines, mangled indentation. Algorithm 2 is even worse with mixed body text leaking in (`L pro-`, `the user`, `-RLM`).

**Winner: Marker** — by a wide margin. The algorithms are clean and readable. Docling's OCR-based approach introduced significant noise into the pseudocode blocks, likely because the OCR tried to read the two-column layout around the algorithm boxes and mixed in adjacent body text.

---

## 7. Table (Table 1 — Main Results)

The critical benchmark table with scores, costs, and model comparisons.

**Marker:** Proper multi-row markdown pipe table with correct column alignment:
```
| Model | CodeQA | BrowseComp+ (1K) | OOLONG | OOLONG-Pairs |
|-------|--------|-------------------|--------|--------------|
| Base Model | 24.0∗ | 0.0∗ | 44.0 | 0.1 |
| ($0.13 ± $0.07) | (N/A) ± (N/A) | ($0.14 ± $0.02) | ($0.16 ± $0.10) |
...
```
4 content columns, ~20 data rows, all values correct.

**Docling:** Also a multi-row pipe table, but with **duplicated columns** and misaligned data:
```
| Model | CodeQA | CodeQA | BrowseComp+ (1K) | BrowseComp+ (1K) | OOLONG | OOLONG | OOLONG-Pairs |
```
7-8 columns instead of 4, with scores and costs split into separate columns and some row merging issues (e.g., `Base Model CodeAct (+` on one line, `BM25)` on the next). The `Qwen3-8B` row header is repeated across all columns.

**Winner: Marker** — clean 4-column table. Docling detected the table but produced duplicate columns and misaligned rows from TableFormer's attempt to parse the complex multi-level header.

---

## 8. Figure Content Extraction

The paper's Figure 2 contains an illustrative RLM example with code snippets and REPL output.

| Tool | Handling |
|------|----------|
| **Marker** | Figure caption extracted as italic text. No figure content (images disabled at Tier 1) |
| **Docling** | Figure caption extracted. Additionally, OCR extracted the **text content inside the figure** — including the example prompt, code snippets (`part1, part2 = prompt.split("Chapter 2")`), and REPL output |

This is Docling Tier 4's biggest advantage — it OCR'd the text embedded within figures. Marker Tier 1 skips figure content entirely.

However, the extracted figure text is messy and interleaved with body text (e.g., `RLM (root / depth=0)`, `Environment E`, `Prompt`, followed by example text). It's useful for search/indexing but not clean reading.

**Winner: Docling** — extracts figure content that Marker completely misses. Messy but present.

---

## 9. Citations and References

| Feature | Marker | Docling |
|---------|--------|---------|
| **Inline citations** | Hyperlinked: `[\(Hong et al.,\)](#page-9-0) [\(2025\)](#page-9-0)` | Plain text: `(Hong et al., 2025)` |
| **Cross-references** | `Figure [1](#page-0-0)`, `Table [1](#page-4-1)`, `[§3.](#page-3-0)` — all clickable | `Figure 1`, `Table 1`, `§3` — plain text |
| **Reference list** | Full list with `<span id>` anchors and `[hyperlinked URLs](url)` | Full list with spaces in URLs: `https://arxiv . org/abs/2412 . 15204` |
| **HTML anchors** | `<span id="page-8-1">` for each reference | None |

**Winner: Marker** — fully hyperlinked citations, cross-references, and reference URLs. Docling has plain text citations with broken URL spacing.

---

## 10. Figure Captions

| Tool | Style |
|------|-------|
| **Marker** | `*Figure 1.* A comparison of GPT-5...` — italic label with period |
| **Docling** | `Figure 1. A comparison of GPT-5...` — plain text |

**Winner: Marker** — italic formatting matches the PDF original.

---

## 11. Footnotes

| Tool | Style |
|------|-------|
| **Marker** | `<sup>1</sup>This is key: it forces M to rely on...` with `<span id="page-2-1">` anchor |
| **Docling** | `1 This is key: it forces M to rely on...` — plain text with space after number |

**Winner: Marker** — superscript footnote numbers with anchor IDs.

---

## 12. Image Placeholders

| Tool | Handling |
|------|----------|
| **Marker** | No image tags (disabled at Tier 1) |
| **Docling** | `<!-- image -->` placeholder tags at figure locations |

**Winner: Docling** — marks where figures appear in the document flow.

---

## 13. Chart/Figure Legend Leakage

Docling's OCR extracted text from chart legends and axes that Marker didn't:

**Docling leaked (from Figure 3 bar chart):**
```
4
3-
1-
Base Model (GPT-5)
RLM(GPT-5) with REPL
RLM(GPT-5) with REPL (no sub-calls)
Summary Agent (GPT-5)
CodeAct (GPT-5) + BM25
CodeAct (GPT-5) + Subagents
-
25th
```

**And from Figure 4 code examples:**
```
Execution 1
Execution Time:
0.158s
# Let's scan the context for clues using keyword searches...
hits - O
for i, chunk in enumerate(context):
...
```

This is a double-edged sword: the content is real and potentially useful (you can see the actual RLM code examples from Figure 4), but it clutters the markdown with unstructured fragments.

**Winner: Tie** — Docling extracts more content (useful for search), Marker is cleaner (better for reading).

---

## 14. Speed

| Tool | Time | Notes |
|------|------|-------|
| **Marker Tier 1** | ~2m 14s | Surya layout model only |
| **Docling Tier 4** | 52.4s | Egret Large + OCR + TableFormer + code/formula enrichment |

**Winner: Docling** — 2.5x faster despite running more models. This is because Marker's Surya models run sequentially on CPU/MPS, while Docling's pipeline is threaded with batched inference.

---

## 15. Completeness

Both tools extract all body text from the 9 pages. Docling extracts additional content from inside figures (code examples, chart legends) that Marker skips entirely.

**Winner: Docling** — extracts more total content due to OCR on figure regions.

---

## Scorecard Summary

| Dimension | Marker Tier 1 | Docling Tier 4 |
|-----------|---------------|----------------|
| Title/Authors | 1st | 2nd |
| Section Headings | 1st | 2nd |
| Body Text Quality | 1st | 2nd |
| Inline Formatting | 1st | 2nd |
| Math Notation | 1st | 2nd |
| Algorithm Blocks | 1st | 2nd (garbled OCR) |
| Table Extraction | 1st | 2nd (duplicate columns) |
| Figure Content | 2nd | 1st |
| Citations/Links | 1st | 2nd |
| Figure Captions | 1st | 2nd |
| Footnotes | 1st | 2nd |
| Image Placeholders | 2nd | 1st |
| Chart Legend Handling | Tie | Tie |
| Speed | 2nd | 1st |
| Completeness | 2nd | 1st |

**Wins:**
- **Marker Tier 1:** 9 wins
- **Docling Tier 4:** 4 wins
- **Tie:** 2

---

## Verdict

**Marker Tier 1** still produces more polished, readable markdown than **Docling Tier 4** for this arXiv paper, despite Docling running significantly more AI models. Marker wins on formatting (headings, italic, superscript), text cleanliness (no spurious spaces), algorithm blocks (clean vs garbled OCR), table structure (4 clean columns vs duplicated columns), and citations (fully hyperlinked).

**Docling Tier 4** wins on speed (2.5x faster), figure content extraction (OCR'd text from inside diagrams), image placeholders, and raw completeness. Its OCR capability means it captures content that Marker's Tier 1 simply cannot see — like the code examples embedded in Figure 4.

### The Paradox

Docling Tier 4's OCR actually *hurts* some elements (algorithm blocks) while helping others (figure content). The OCR introduced noise into the pseudocode blocks by mixing in adjacent column text. Meanwhile, Marker Tier 1 — which uses no OCR — produces cleaner algorithms because it relies purely on the PDF's embedded text layer, which is already correct.

### Recommendation

| Use Case | Best Choice |
|----------|-------------|
| **Reading/reviewing a paper** | Marker Tier 1 (cleaner formatting, better tables) |
| **Extracting ALL content including figures** | Docling Tier 4 (OCR captures figure text) |
| **Speed-sensitive processing** | Docling Tier 4 (52s vs 2m14s) |
| **Citation-heavy work** | Marker Tier 1 (hyperlinked references) |
| **Papers with important code in figures** | Docling Tier 4 (extracts code from images) |
| **Papers with complex pseudocode** | Marker Tier 1 (no OCR noise in algorithms) |
