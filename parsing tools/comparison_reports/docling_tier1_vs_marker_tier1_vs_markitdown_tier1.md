# Tier 1 Comparison: Docling vs Marker vs MarkItDown

**Paper:** "Recursive Language Models" (2512.24601v2), first 9 pages
**Machine:** Apple M2 Pro, 16GB unified RAM, macOS

---

## Overview

| Metric | Docling Tier 1 | Marker Tier 1 | MarkItDown Tier 1 |
|--------|---------------|---------------|-------------------|
| **Tool** | IBM Docling 2.72.0 | Datalab Marker 1.10.1 | Microsoft MarkItDown 0.1.5b1 |
| **Method** | DoclingParse backend (no AI models) | Surya layout detection (no OCR) | pdfminer.six + pdfplumber |
| **Time** | 14.5s | ~2m 14s | ~1.6s |
| **Output size** | 42,143 chars | 46,282 chars | 40,762 chars |
| **Lines** | 217 | 263 | 1,027 |

---

## 1. Title and Author Extraction

| Tool | Result |
|------|--------|
| **Docling** | `## Recursive Language Models` then `## Alex L. Zhang 1 Tim Kraska 1 Omar Khattab 1` — proper H2 headings, but author line includes raw affiliation numbers |
| **Marker** | `# Recursive Language Models` then `## Alex L. Zhang <sup>1</sup> Tim Kraska <sup>1</sup> Omar Khattab <sup>1</sup>` — H1 for title, H2 for authors, superscript affiliation numbers |
| **MarkItDown** | `Recursive Language Models` then `Alex L. Zhang 1 Tim Kraska 1 Omar Khattab 1` — plain text, no markdown headings at all |

**Winner: Marker** — proper heading hierarchy, superscript formatting for affiliations.

---

## 2. Section Headings

| Tool | Heading detection | Format |
|------|-------------------|--------|
| **Docling** | All major sections detected (Abstract, 1. Introduction, 2. Recursive Language Models, etc.) | `## Section Name` — all H2 |
| **Marker** | All major sections detected, plus subsections (3.1, 3.2, 4.1) | H1 for title, H2 for sections, H4 for subsections |
| **MarkItDown** | Section numbers detected but no markdown heading syntax | Plain text: `1. Introduction`, `2. Recursive Language Models` |

**Winner: Marker** — proper heading hierarchy with multiple levels. Docling detects sections but flattens everything to H2. MarkItDown has no heading formatting at all.

---

## 3. Body Text Quality

| Tool | Paragraph handling | Line breaks | Reading order |
|------|-------------------|-------------|---------------|
| **Docling** | Clean paragraph blocks, well-merged lines | Proper paragraph breaks | Correct two-column reading order |
| **Marker** | Clean paragraph blocks, well-merged lines | Proper paragraph breaks | Correct two-column reading order |
| **MarkItDown** | Line breaks at PDF column width (mid-sentence breaks) | Preserves raw PDF line breaks | Mostly correct but some column merging issues |

**Example — same sentence across all three:**

**Docling:**
> We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling.

**Marker:**
> We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling.

**MarkItDown:**
> We study allowing large language models (LLMs)
> to process arbitrarily long prompts through the
> lens of inference-time scaling.

**Winner: Tie (Docling = Marker)** — both merge lines into proper paragraphs. MarkItDown preserves raw PDF line breaks, making text harder to read.

---

## 4. Inline Formatting (Bold, Italic, Emphasis)

| Tool | Bold | Italic | Superscript |
|------|------|--------|-------------|
| **Docling** | None detected | None detected | None |
| **Marker** | None | Yes — `*context rot*`, `*environment*`, `*programmatically*` | Yes — `<sup>1</sup>`, `2 <sup>13</sup>` |
| **MarkItDown** | None | None | None |

**Winner: Marker** — captures italic emphasis and superscript notation. Docling and MarkItDown produce plain text only.

---

## 5. Mathematical Notation

| Tool | Inline math | Symbols | Rendering |
|------|-------------|---------|-----------|
| **Docling** | `P ∈ Σ ⋆`, `\| P \|≫ K`, `Ω( \| P \| )` | Unicode symbols preserved | Readable but no LaTeX |
| **Marker** | `P ∈ Σ ⋆`, `\|P\| ≫ K`, `Ω(\|P\|)` | Unicode symbols preserved | Readable but no LaTeX |
| **MarkItDown** | `P ∈ Σ⋆`, `\|P \| ≫ K`, `Ω(\|P \|)` | Unicode symbols preserved | Readable but extra spaces |

**Winner: Tie** — all three handle Unicode math symbols comparably at Tier 1. No tool produces LaTeX at this tier. Marker has slightly better spacing.

---

## 6. Algorithm/Pseudocode Blocks

| Tool | Algorithm 1 | Algorithm 2 |
|------|-------------|-------------|
| **Docling** | Detected with `## Algorithm 1` heading, content in a fenced code block | Detected with heading, content in fenced code block |
| **Marker** | Detected with heading, content in fenced ``````` code block, properly indented | Detected with heading, content in fenced code block |
| **MarkItDown** | Plain text dump, no code block markers, mixed into body | Plain text dump, no code block markers |

**Winner: Tie (Docling = Marker)** — both wrap algorithms in fenced code blocks. MarkItDown dumps them as unformatted text.

---

## 7. Table (Table 1 — Main Results)

This is the critical benchmark table with performance scores across methods and tasks.

| Tool | Table detected? | Format | Readability | Cell accuracy |
|------|----------------|--------|-------------|---------------|
| **Docling** | Yes | Single-row markdown pipe table with all data crammed into one row | Very poor — one massive cell | Data present but unusable |
| **Marker** | Yes | Proper multi-row markdown pipe table with headers and aligned columns | Good — readable, properly structured | Most cells correct, some formatting noise with `$` signs |
| **MarkItDown** | Sort of | Raw text dump of cell contents | Poor — unstructured text blob | Data present but not tabular |

**Docling** produced:
```
| Model  CodeQA  BrowseComp+ (1K)  OOLONG  OOLONG-Pairs  Task Length... |
|------|
```
All data crammed into a single cell.

**Marker** produced:
```
| Model | CodeQA | BrowseComp+ (1K) | OOLONG | OOLONG-Pairs |
|-------|--------|-------------------|--------|--------------|
| Base Model | 24.0∗ | 0.0∗ | 44.0 | 0.1 |
...
```
Properly structured with rows and columns.

**Winner: Marker** — by a wide margin. The multi-row pipe table is readable and well-structured. Docling detected the table but collapsed it into a single row. MarkItDown had no table structure.

---

## 8. Citations and References

| Tool | Inline citations | Reference list | Hyperlinks |
|------|-----------------|----------------|------------|
| **Docling** | `(Hong et al., 2025)` — plain parenthetical | Full reference list at end | `https: //github . com/alexzhang13/rlm` (spaces in URLs) |
| **Marker** | `[\(Hong et al.,\)](#page-9-0) [\(2025\)](#page-9-0)` — hyperlinked with anchor IDs | Full reference list with `<span id="page-X-Y">` anchors and `[https://arxiv.org/abs/...](url)` links | Proper clickable hyperlinks |
| **MarkItDown** | `(Hong et al., 2025)` — plain parenthetical | Full reference list at end | `https://arxiv.org/abs/...` plain URLs |

**Winner: Marker** — cross-linked citations with page anchors and proper hyperlinked URLs. Docling has broken URL spacing. MarkItDown has plain text only.

---

## 9. Figure References and Image Placeholders

| Tool | Figure captions | Image handling |
|------|----------------|----------------|
| **Docling** | Figure captions extracted as body text (e.g., "Figure 1. A comparison of GPT-5...") | `<!-- image -->` placeholder tags |
| **Marker** | Figure captions with italic formatting (`*Figure 1.* A comparison...`) | No image tags (images disabled in Tier 1) |
| **MarkItDown** | Figure captions as plain text | No image handling |

**Winner: Marker** — italic figure labels. Docling is close with image placeholders. MarkItDown is plain text.

---

## 10. Footnotes

| Tool | Handling |
|------|----------|
| **Docling** | Footnote text appears inline at end of the relevant section, prefixed with number |
| **Marker** | Footnote text with superscript number and `<span>` anchor: `<sup>1</sup>This is key: it forces M to rely on...` |
| **MarkItDown** | Footnote text appears inline, prefixed with `1This is key:` |

**Winner: Marker** — proper superscript footnote numbers with anchor IDs.

---

## 11. arXiv Sidebar (Metadata)

The arXiv PDF has a sidebar with metadata (arXiv ID, submission date, category).

| Tool | Handling |
|------|----------|
| **Docling** | Not visible in output — cleanly stripped or not extracted |
| **Marker** | Clean: `*Preprint. January 29, 2026.*` — date extracted as italic text |
| **MarkItDown** | Garbled vertical text: individual characters on separate lines (`6`, `2`, `0`, `2`, `n`, `a`, `J`, ...) |

**Winner: Docling** — cleanest handling (omitted entirely). Marker extracts it neatly. MarkItDown outputs garbled character-per-line noise.

---

## 12. Chart/Plot Data Leakage

MarkItDown's pdfminer sometimes extracts raw data from chart axes embedded in the PDF.

| Tool | Handling |
|------|----------|
| **Docling** | No chart data leaked into text |
| **Marker** | No chart data leaked into text |
| **MarkItDown** | Leaked: `8k16k33k66k131k262k524k1M020406080100Score (%)GPT-5OOLONGOOLONG-PairsS-NIAH...` dumped as a single line |

**Winner: Tie (Docling = Marker)** — neither leaks chart data. MarkItDown dumps raw SVG/chart data into the text.

---

## 13. Output Structure and Readability

| Tool | Overall feel |
|------|--------------|
| **Docling** | Clean, well-structured markdown. Proper headings, paragraphs, code blocks. Reads like a document. Missing inline formatting and has some table issues. |
| **Marker** | Best-structured markdown. Heading hierarchy, italic/bold, hyperlinked citations, proper tables, code blocks, superscripts. Reads closest to the original paper. |
| **MarkItDown** | Raw text extraction. No markdown structure. Mid-sentence line breaks, garbled sidebar, leaked chart data. Reads like a text dump. |

---

## 14. Speed

| Tool | Time | Relative |
|------|------|----------|
| **MarkItDown** | ~1.6s | 1x (baseline) |
| **Docling** | 14.5s | ~9x slower |
| **Marker** | ~2m 14s | ~84x slower |

**Winner: MarkItDown** — by far the fastest. Docling is a solid middle ground. Marker is the slowest due to Surya model inference.

---

## 15. Completeness (Content Coverage)

All three tools extracted the full body text of the first 9 pages. No sections were missing from any output. References were fully captured by all three.

**Winner: Tie** — all three capture the complete text content.

---

## Scorecard Summary

| Dimension | Docling | Marker | MarkItDown |
|-----------|---------|--------|------------|
| Title/Authors | 2nd | 1st | 3rd |
| Section Headings | 2nd | 1st | 3rd |
| Body Text Quality | 1st (tie) | 1st (tie) | 3rd |
| Inline Formatting | 3rd (tie) | 1st | 3rd (tie) |
| Math Notation | 1st (tie) | 1st (tie) | 1st (tie) |
| Algorithm Blocks | 1st (tie) | 1st (tie) | 3rd |
| Table Extraction | 2nd | 1st | 3rd |
| Citations/Links | 2nd | 1st | 3rd |
| Figure References | 2nd | 1st | 3rd |
| Footnotes | 2nd | 1st | 3rd |
| arXiv Sidebar | 1st | 2nd | 3rd |
| Chart Data Leakage | 1st (tie) | 1st (tie) | 3rd |
| Overall Readability | 2nd | 1st | 3rd |
| Speed | 2nd | 3rd | 1st |
| Completeness | 1st (tie) | 1st (tie) | 1st (tie) |

**Wins:**
- **Marker:** 10 wins (best formatting, tables, citations, readability)
- **Docling:** 5 wins (clean text, sidebar handling, speed middle ground)
- **MarkItDown:** 2 wins (fastest, complete text)

---

## Verdict

**Marker Tier 1** produces the most polished, readable markdown despite being the slowest. It wins on formatting, structure, tables, citations, and overall readability — critical for academic papers.

**Docling Tier 1** is a strong middle option — 9x faster than Marker with clean paragraph text, proper headings, and code blocks. Its main weaknesses are table collapsing (single-row) and missing inline formatting (no italic/bold). A good choice when you need decent structure without the wait.

**MarkItDown Tier 1** is the fastest by far but produces the least usable output — raw text with mid-sentence line breaks, garbled sidebar metadata, and no markdown formatting. Best suited for quick text search/indexing, not for reading.

### Recommendation by Use Case

| Use Case | Best Tool |
|----------|-----------|
| **Reading/reviewing a paper** | Marker Tier 1 |
| **Quick structured extraction** | Docling Tier 1 |
| **Fast text search / indexing** | MarkItDown Tier 1 |
| **Processing hundreds of papers** | Docling Tier 1 (best speed/quality ratio) |
| **Table-heavy papers** | Marker Tier 1 (only tool with usable tables) |
