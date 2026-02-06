# SciPDF Tier 5 vs Marker Tier 1

**Paper:** "Recursive Language Models" (2512.24601v2)
**Machine:** Apple M2 Pro, 16GB unified RAM, macOS

**Important caveat:** SciPDF processed all 19 pages (no page range support). Marker processed only the first 9 pages. Output size comparisons reflect this difference. Content comparisons focus on overlapping sections where possible.

---

## Overview

| Metric | SciPDF Tier 5 | Marker Tier 1 |
|--------|--------------|---------------|
| **Tool** | SciPDF Parser v0.1.1 (GROBID v0.7.3) | Datalab Marker v1.10.1 |
| **Method** | GROBID CRF models + spaCy NLP + textstat readability analysis | Surya layout detection, no OCR |
| **Time** | 58.9s (GROBID) + 3.9s (analysis) = **62.8s total** (19 pages) | ~2m 14s (9 pages) |
| **Output size** | 99,601 chars (536 lines) | 46,282 chars (263 lines) |
| **Pages processed** | All 19 | First 9 only |
| **Unique feature** | Text analysis appendix (readability, POS, journal stats) | Formatted markdown with hyperlinks |

---

## What SciPDF Tier 5 Adds Over Tier 3

SciPDF Tier 5's base text extraction is **identical** to Tier 3. The only addition is a text analysis appendix (~4,750 extra chars, 84 extra lines) containing:

1. **Abstract readability metrics** (13 metrics)
2. **Abstract linguistic stats** (word/sentence/verb counts)
3. **Journal/reference features** (reference year statistics)
4. **Per-section readability table** (Flesch ease, FK grade, SMOG, word/sentence counts for every section)

This means all text extraction strengths and weaknesses from the Tier 3 comparison carry over. This report focuses on what the analysis appendix adds relative to Marker's output.

---

## 1. Title and Author Extraction

| Tool | Result |
|------|--------|
| **SciPDF** | Correct title, but **50+ incorrect authors** — GROBID confused names from the Singh et al. (2025) GPT-5 reference with paper co-authors |
| **Marker** | Correct title (H1), correct 3 authors with `<sup>1</sup>` affiliation markers |

**Winner: Marker** — SciPDF's GROBID catastrophically over-extracted authors.

---

## 2. Section Headings

| Tool | Hierarchy | Issues |
|------|-----------|--------|
| **SciPDF** | All flat `##` (H2) | 20 OOLONG-Pairs benchmark tasks misidentified as paper sections (`## Task 3`, `## Task 4`, ..., `## Task 20`) |
| **Marker** | H1 → H2 → H4 multi-level | Clean hierarchy: `## 1. Introduction` → `#### 3.1. Tasks` |

**Winner: Marker** — proper heading hierarchy without spurious section detection.

---

## 3. Body Text Quality

Both produce clean, well-merged paragraph text with correct two-column reading order. SciPDF merges some paragraph breaks within sections (e.g., Observations 1-6 appear as continuous text without sub-headings). Marker preserves paragraph breaks and observation headers.

**Winner: Marker** — better paragraph separation.

---

## 4. Inline Formatting

| Feature | SciPDF | Marker |
|---------|--------|--------|
| **Italic** | None | `*context rot*`, `*environment*`, `*programmatically*` |
| **Superscript** | None | `<sup>1</sup>`, `2 <sup>13</sup>` |

**Winner: Marker** — captures emphasis and superscript from the PDF. SciPDF is plain text only.

---

## 5. Mathematical Notation

Both use Unicode math from the PDF's text layer. Neither produces LaTeX at these tiers.

| Tool | Example |
|------|---------|
| **SciPDF** | `P ∈ Σ ⋆`, `\|P \| ≫ K`, `Ω(\|P \|)` |
| **Marker** | `P ∈ Σ ⋆`, `\|P\| ≫ K`, `Ω(\|P\|)` |

**Winner: Tie**

---

## 6. Algorithm/Pseudocode Blocks

**SciPDF:** Algorithm blocks are **lost**. Algorithm 1 is missing entirely. A garbled fragment of Algorithm 2 appears in the Formulas section as a single line.

**Marker:** Both algorithms extracted as clean fenced code blocks with proper indentation.

**Winner: Marker** — by a wide margin.

---

## 7. Table (Table 1 — Main Results)

**SciPDF:** Table data is present but dumped as a single unstructured text blob in a code block — all rows crammed together with no column separation.

**Marker:** Proper multi-row markdown pipe table with headers, column alignment, and all values correct.

**Winner: Marker** — readable structured table vs unreadable blob.

---

## 8. Citations and References

| Feature | SciPDF | Marker |
|---------|--------|--------|
| **Inline citations** | Plain text: `(Hong et al., 2025)` | Hyperlinked: `[\(Hong et al.,\)](#page-9-0)` |
| **Cross-references** | Plain text: `Figure 1`, `§3` | Clickable: `Figure [1](#page-0-0)`, `[§3.](#page-3-0)` |
| **Reference list** | Structured author/title/year fields, but many empty journal fields (`**`) | Full entries with `<span id>` anchors and clickable URLs |
| **Reference URLs** | None | `[https://arxiv.org/abs/...](url)` |

**Winner: Marker** — hyperlinked everything. SciPDF has structured fields but no hyperlinks and broken rendering.

---

## 9. Footnotes

| Tool | Handling |
|------|----------|
| **SciPDF** | Merged into body text, footnote markers lost |
| **Marker** | `<sup>1</sup>This is key: it forces M to rely on...` with anchor IDs |

**Winner: Marker**

---

## 10. Figure Captions

| Tool | Handling |
|------|----------|
| **SciPDF** | 22 figure objects detected, captions as separate entries: `**1** (figure): Figure1. A comparison...` Some entries empty (`**** (figure):`) |
| **Marker** | `*Figure 1.* A comparison of GPT-5...` — italic labels, inline with text flow |

**Winner: Marker** for readability. SciPDF for structured metadata.

---

## 11. Structured Data

SciPDF's JSON output provides:
- 41 sections as separate objects with cross-reference IDs (`publication_ref`, `figure_ref`, `table_ref`)
- 41 references parsed into `ref_id`, `title`, `journal`, `year`, `authors`
- 22 figures with `figure_label`, `figure_type`, `figure_caption`, `figure_data`
- DOI: `10.1561/1500000019`
- Publication date: `2026-01-28`

Marker outputs flat markdown — no programmatic access without parsing.

**Winner: SciPDF** — structured data is fundamentally more useful for programmatic processing.

---

## 12. Text Analysis (SciPDF Tier 5 Exclusive)

This is what Tier 5 adds that no Marker tier provides at all.

### Abstract Readability

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Flesch Reading Ease | 31.1 | Difficult (college/graduate level) |
| Flesch-Kincaid Grade | 14.5 | ~College sophomore |
| SMOG Index | 16.1 | Graduate-level |
| Coleman-Liau Index | 15.5 | Graduate-level |
| Gunning Fog | 17.9 | Post-graduate |
| Dale-Chall | 13.1 | College graduate |
| Text Standard | 15th-16th grade | Consistent across metrics |
| Avg sentence length | 22.5 words | Moderate |
| Difficult words | 46 | High density |

### Abstract Linguistic Stats

| Stat | Value |
|------|-------|
| Words | 167 |
| Sentences | 5 |
| Verbs | 18 |
| Avg words/sentence | 33.4 |

### Journal/Reference Features

| Stat | Value |
|------|-------|
| Total references | 41 |
| Unique journals | 4 |
| Avg reference year | 2024.0 |
| Median reference year | 2024.5 |
| Min reference year | 2018 |
| Max reference year | 2026 |

### Per-Section Readability Highlights

| Section | Flesch Ease | FK Grade | Words | Sentences |
|---------|-----------|----------|-------|-----------|
| Introduction | 40.8 | 11.8 | 1,065 | 34 |
| Recursive Language Models | 40.5 | 14.0 | 793 | 26 |
| Scaling Long Context Tasks | 62.0 | 7.6 | 161 | 5 |
| Methods and Baselines | 54.1 | 9.7 | 690 | 24 |
| Results and Discussion | 44.5 | 11.7 | 1,316 | 47 |
| Related Works | 54.1 | 8.6 | 422 | 12 |
| Conclusion | 18.6 | 18.8 | 150 | 4 |

The Conclusion section is the hardest to read (Flesch 18.6, FK grade 18.8), while Scaling Long Context Tasks is the easiest (Flesch 62.0, FK grade 7.6).

**Winner: SciPDF** — Marker has no text analysis capabilities. This is a unique and valuable feature for scientometrics, paper review, and corpus analysis.

---

## 13. Speed

| Tool | Time | Pages | Per-page |
|------|------|-------|----------|
| **SciPDF Tier 5** | 62.8s total (58.9s GROBID + 3.9s analysis) | 19 | ~3.3s/page |
| **Marker Tier 1** | ~134s | 9 | ~14.9s/page |

**Winner: SciPDF** — ~4.5x faster per page, even including the text analysis step.

---

## Scorecard Summary

| Dimension | SciPDF Tier 5 | Marker Tier 1 |
|-----------|--------------|---------------|
| Title/Authors | 2nd | 1st |
| Section Headings | 2nd | 1st |
| Body Text Quality | 2nd | 1st |
| Inline Formatting | 2nd | 1st |
| Math Notation | Tie | Tie |
| Algorithm Blocks | 2nd | 1st |
| Table Extraction | 2nd | 1st |
| Figure Metadata | 1st | 2nd |
| Citations/Links | 2nd | 1st |
| Footnotes | 2nd | 1st |
| Structured Data | 1st | 2nd |
| Text Analysis | 1st | N/A (not available) |
| Metadata (DOI/date) | 1st | 2nd |
| Speed (per page) | 1st | 2nd |

**Wins:**
- **Marker Tier 1:** 8 wins
- **SciPDF Tier 5:** 5 wins (including text analysis exclusive)
- **Tie:** 1

---

## Verdict

**Marker Tier 1** remains the better tool for producing readable, well-formatted markdown. It wins every formatting dimension: headings, emphasis, algorithms, tables, citations, and footnotes. If you need a document that looks like the original paper, Marker is the clear choice.

**SciPDF Tier 5** offers two capabilities that Marker cannot match:

1. **Structured data** — sections, references, and figures as separate programmatic objects with cross-reference IDs, DOI, and publication date
2. **Text analysis** — readability metrics (Flesch, SMOG, Gunning Fog), linguistic features (POS tagging, word counts), and bibliometric statistics (reference year distribution, unique journals)

### What the Text Analysis Reveals

The readability analysis shows this paper is written at a **15th-16th grade level** (graduate school), with the Introduction and Conclusion being the densest sections. The bibliography skews very recent (median year 2024.5, all refs from 2018-2026), with only 4 unique journals — typical of a fast-moving ML research area where most citations are preprints.

This kind of analysis is impossible with Marker's output without building a separate NLP pipeline.

### Recommendation

| Use Case | Best Choice |
|----------|-------------|
| **Reading/reviewing a paper** | Marker Tier 1 |
| **Scientometrics / paper review** | SciPDF Tier 5 |
| **Readability analysis** | SciPDF Tier 5 (exclusive) |
| **Citation analysis / bibliometrics** | SciPDF Tier 5 |
| **Building search indexes** | SciPDF Tier 5 |
| **Papers with complex tables/algorithms** | Marker Tier 1 |
| **Speed-sensitive batch processing** | SciPDF Tier 5 (~3.3s/page vs ~15s/page) |
| **Formatting fidelity** | Marker Tier 1 |
| **Corpus linguistics** | SciPDF Tier 5 (POS tagging, readability) |
