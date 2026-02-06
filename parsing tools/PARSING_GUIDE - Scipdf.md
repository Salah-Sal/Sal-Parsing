# SciPDF Parser — Parsing Guide

**Tool:** SciPDF Parser v0.1.1
**Author:** Titipat Achakulvisut
**Backend:** [GROBID](https://github.com/kermitt2/grobid) (machine learning service for scientific document extraction)
**Architecture:** Client-server — Python client sends PDFs to a GROBID Docker service for processing
**Output format:** Structured Python dictionaries (NOT markdown)

---

## How It Works

Unlike Marker, Docling, or MarkItDown, SciPDF Parser does **not** produce markdown. It uses a two-step architecture:

1. **GROBID server** (Docker container) — receives the PDF, runs ML models (CRF sequence labeling, deep learning) to extract structured TEI XML
2. **Python client** — parses the XML into structured Python dictionaries with separate fields for title, abstract, sections, references, figures, and formulas

This means you get **structured data** (JSON-like dicts) rather than a flat markdown document. Each section, reference, and figure is a separate object you can programmatically access.

---

## Prerequisites

```bash
# 1. Install Docker (required — GROBID runs as a container)
# macOS: brew install --cask docker

# 2. Install scipdf_parser
pip install git+https://github.com/titipata/scipdf_parser

# 3. Install spaCy model (for text analysis features)
python -m spacy download en_core_web_sm

# 4. Java (only needed for pdffigures2 figure image extraction)
# macOS: brew install openjdk
```

### Starting the GROBID Server

```bash
# On Apple Silicon (M1/M2/M3), use the standard image — it runs via Rosetta emulation.
# The ARM-specific tag (grobid:0.7.3-arm) does NOT exist on Docker Hub.
docker run --rm --init --ulimit core=0 -p 8070:8070 -d grobid/grobid:0.7.3

# Verify it's running (takes ~50s to initialize after container starts):
curl http://localhost:8070/api/isalive
```

The GROBID server must be running before any parsing. It listens on `http://localhost:8070` by default.

**Note:** The bundled `serve_grobid.sh` script references a `grobid:0.7.3-arm` tag that doesn't exist. On Apple Silicon, just use the standard `grobid:0.7.3` tag directly.

**Alternative — Cloud GROBID (no Docker needed):**
```python
grobid_url = "https://kermitt2-grobid.hf.space"
# or
grobid_url = "https://cloud.science-miner.com/grobid/"
```

---

## Quality Tiers (Lowest to Highest)

### Tier 1 — Header Only

Extracts only the paper's metadata: title, authors, date. No body text, no sections, no references.

**Use case:** Quick metadata cataloging, building paper indexes.

```python
import scipdf

# Header-only parsing
xml = scipdf.parse_pdf(
    'path/to/paper.pdf',
    fulltext=False,        # header only — no body text
    soup=True,
    grobid_url="http://localhost:8070"
)

# Extract metadata manually
from scipdf.pdf.parse_pdf import parse_authors, parse_date
title = xml.find("title", attrs={"type": "main"}).text.strip()
authors = parse_authors(xml)
pub_date = parse_date(xml)

print(f"Title: {title}")
print(f"Authors: {authors}")
print(f"Date: {pub_date}")
```

**What you get:**
- Title
- Authors (semicolon-separated)
- Publication date
- DOI (if available)

**What you don't get:**
- Abstract, sections, body text, references, figures, formulas

---

### Tier 2 — Full Text (Structured Dict, No Coordinates)

Full document extraction into a structured dictionary. All sections, references, figure captions, and formulas are parsed as separate objects.

**Use case:** Programmatic access to paper content, building search indexes, NLP pipelines.

```python
import scipdf

article = scipdf.parse_pdf_to_dict(
    'path/to/paper.pdf',
    fulltext=True,
    return_coordinates=False,   # skip coordinate data
    grobid_url="http://localhost:8070"
)

# Access structured data
print(article['title'])
print(article['abstract'])

for section in article['sections']:
    print(f"\n## {section['heading']}")
    print(section['text'])
    print(f"  Cites: {section['publication_ref']}")
    print(f"  Figures: {section['figure_ref']}")

for ref in article['references']:
    print(f"[{ref['ref_id']}] {ref['authors']} ({ref['year']}). {ref['title']}. {ref['journal']}")

for fig in article['figures']:
    print(f"{fig['figure_label']} ({fig['figure_type']}): {fig['figure_caption']}")

for formula in article['formulas']:
    print(f"{formula['formula_id']}: {formula['formula_text']}")
```

**Output structure:**
```python
{
    'title': str,
    'authors': str,              # "Author1; Author2; Author3"
    'pub_date': str,
    'abstract': str,
    'doi': str,
    'sections': [
        {
            'heading': str,              # section title
            'text': str,                 # full section text
            'publication_ref': [str],    # cited reference IDs
            'figure_ref': [str],         # referenced figure IDs
            'table_ref': [str],          # referenced table IDs
        },
        ...
    ],
    'references': [
        {
            'ref_id': str,
            'title': str,
            'journal': str,
            'year': str,
            'authors': str,
        },
        ...
    ],
    'figures': [
        {
            'figure_label': str,         # "Figure 1", "Table 2"
            'figure_type': str,          # "figure" or "table"
            'figure_id': str,
            'figure_caption': str,
            'figure_data': str,          # table cell data (tables only)
        },
        ...
    ],
    'formulas': [
        {
            'formula_id': str,
            'formula_text': str,
        },
        ...
    ],
}
```

---

### Tier 3 — Full Text with Coordinates

Same as Tier 2 but with spatial coordinate data for formulas, figures, references, and person names. Coordinates map elements back to their PDF page positions.

**Use case:** Layout analysis, PDF annotation, linking extracted text to visual positions.

```python
import scipdf

article = scipdf.parse_pdf_to_dict(
    'path/to/paper.pdf',
    fulltext=True,
    return_coordinates=True,    # request TEI coordinates
    grobid_url="http://localhost:8070"
)

# Formulas now include coordinates
for formula in article['formulas']:
    print(f"{formula['formula_id']}: {formula['formula_text']}")
    print(f"  Coords: {formula['formula_coordinates']}")  # [x1, y1, x2, y2]
```

**Coordinate types requested from GROBID:**
- `persName` — author name positions
- `figure` — figure/table bounding boxes
- `ref` — citation reference positions
- `formula` — formula bounding boxes
- `biblStruct` — bibliography entry positions

---

### Tier 3.5 — Full Text with Paragraph-Level Granularity

Same content as Tier 2/3, but section text is returned as a list of individual paragraphs instead of a single joined string. Better for paragraph-level processing.

**Use case:** Paragraph-level NLP, sentence segmentation, fine-grained text analysis.

```python
import scipdf

article = scipdf.parse_pdf_to_dict(
    'path/to/paper.pdf',
    fulltext=True,
    as_list=True,               # sections return list of paragraphs
    return_coordinates=True,
    grobid_url="http://localhost:8070"
)

for section in article['sections']:
    print(f"\n## {section['heading']}")
    for i, paragraph in enumerate(section['text']):  # text is now a list
        print(f"  P{i+1}: {paragraph[:100]}...")
```

---

### Tier 4 — Full Text + Figure Image Extraction (pdffigures2)

Adds physical figure image extraction using [pdffigures2](https://github.com/allenai/pdffigures2) (Allen AI). This is a **separate step** that extracts actual figure images from the PDF as image files and saves metadata JSON.

**Requires:** Java installed on the system.

**Use case:** Building figure databases, visual content extraction, training image models on paper figures.

```python
import scipdf

# Step 1: Parse text content (same as Tier 2/3)
article = scipdf.parse_pdf_to_dict(
    'path/to/paper.pdf',
    fulltext=True,
    return_coordinates=True,
    grobid_url="http://localhost:8070"
)

# Step 2: Extract figure images (separate step)
# NOTE: pdf_folder must contain ONLY PDF files
scipdf.parse_figures(
    'path/to/pdf_folder/',      # folder containing the PDF
    output_folder='figures',     # output base directory
    resolution=300,              # DPI (default 300, range 1-600)
)
# Creates:
#   figures/data/   — JSON metadata for each figure
#   figures/figures/ — extracted figure images
```

**Resolution options:**
| DPI | Use case |
|-----|----------|
| 72 | Quick preview, small files |
| 150 | Screen viewing |
| 300 | Print quality (default) |
| 600 | High-fidelity archival |

---

### Tier 5 — Full Pipeline + Text Analysis Features

Combines full text extraction with NLP-based text analysis: readability metrics, part-of-speech tagging, and journal/reference statistics.

**Requires:** spaCy with `en_core_web_sm` model, textstat, pandas, numpy.

**Use case:** Scientometrics, readability analysis, automated paper review, corpus linguistics.

```python
import scipdf
from scipdf.features import compute_readability_stats, compute_text_stats
from scipdf.features.text_utils import compute_journal_features  # NOT exported from scipdf.features
import spacy

nlp = spacy.load("en_core_web_sm")

# Step 1: Parse the paper
article = scipdf.parse_pdf_to_dict(
    'path/to/paper.pdf',
    fulltext=True,
    return_coordinates=True,
    grobid_url="http://localhost:8070"
)

# Step 2: Readability analysis on abstract
readability = compute_readability_stats(article['abstract'])
print(f"Flesch Reading Ease: {readability['flesch_reading_ease']}")
print(f"Flesch-Kincaid Grade: {readability['flesch_kincaid_grade']}")
print(f"SMOG Index: {readability['smog']}")
print(f"Gunning Fog: {readability['gunning_fog']}")
print(f"Dale-Chall: {readability['dale_chall']}")
print(f"Avg sentence length: {readability['avg_sentence_length']}")

# Step 3: POS tagging and linguistic features
doc = nlp(article['abstract'])
text_stats = compute_text_stats(doc)
print(f"Word count: {text_stats['n_word']}")
print(f"Sentence count: {text_stats['n_sents']}")
print(f"Verb count: {text_stats['n_verb']}")
print(f"POS distribution: {text_stats['pos']}")

# Step 4: Journal/reference statistics
journal_features = compute_journal_features(article)
print(f"Total references: {journal_features['n_reference']}")
print(f"Unique journals: {journal_features['n_unique_journals']}")
print(f"Avg reference year: {journal_features['avg_ref_year']}")
print(f"Year range: {journal_features['min_ref_year']}–{journal_features['max_ref_year']}")

# Step 5: Section heading normalization
from scipdf.features.text_utils import merge_section_list
section_headings = [s['heading'] for s in article['sections']]
normalized = merge_section_list(section_headings)
print(f"Normalized sections: {normalized}")
```

**Readability metrics returned (13 total):**
| Metric | Description |
|--------|-------------|
| `flesch_reading_ease` | 0-100 ease score (higher = easier) |
| `flesch_kincaid_grade` | US school grade level |
| `smog` | SMOG readability index |
| `coleman_liau_index` | Coleman-Liau index |
| `automated_readability_index` | ARI score |
| `dale_chall` | Dale-Chall readability |
| `difficult_words` | Count of difficult words |
| `linsear_write` | Linsear Write formula |
| `gunning_fog` | Gunning Fog index |
| `text_standard` | Consensus grade level |
| `n_syllable` | Total syllable count |
| `avg_letter_per_word` | Average letters per word |
| `avg_sentence_length` | Average sentence length |

---

## Summary Table

| Tier | What It Extracts | Requirements | Speed |
|------|-----------------|--------------|-------|
| **1** | Title, authors, date only | Docker + GROBID | Fastest |
| **2** | Full text: sections, refs, figures, formulas (as dict) | Docker + GROBID | Fast |
| **3** | Full text + spatial coordinates | Docker + GROBID | Fast |
| **3.5** | Full text + paragraph-level granularity | Docker + GROBID | Fast |
| **4** | Full text + extracted figure images | Docker + GROBID + Java | Medium |
| **5** | Full text + readability/POS/journal analysis | Docker + GROBID + spaCy | Medium |

---

## Converting Dict Output to Markdown

SciPDF Parser does not produce markdown natively. Here is a helper to convert the structured dict to markdown:

```python
def article_dict_to_markdown(article):
    """Convert scipdf article dict to markdown string."""
    lines = []

    # Title
    lines.append(f"# {article['title']}\n")

    # Authors and date
    if article.get('authors'):
        lines.append(f"**Authors:** {article['authors']}\n")
    if article.get('pub_date'):
        lines.append(f"**Date:** {article['pub_date']}\n")
    if article.get('doi'):
        lines.append(f"**DOI:** {article['doi']}\n")

    # Abstract
    if article.get('abstract'):
        lines.append(f"## Abstract\n")
        lines.append(f"{article['abstract']}\n")

    # Sections
    for section in article.get('sections', []):
        heading = section.get('heading', '')
        text = section.get('text', '')
        if heading:
            lines.append(f"## {heading}\n")
        if isinstance(text, list):
            lines.append('\n\n'.join(text) + '\n')
        elif text:
            lines.append(f"{text}\n")

    # Figures and tables
    if article.get('figures'):
        lines.append(f"## Figures\n")
        for fig in article['figures']:
            label = fig.get('figure_label', '')
            caption = fig.get('figure_caption', '')
            fig_type = fig.get('figure_type', 'figure')
            lines.append(f"**{label}** ({fig_type}): {caption}\n")
            if fig.get('figure_data'):
                lines.append(f"```\n{fig['figure_data']}\n```\n")

    # Formulas
    if article.get('formulas'):
        lines.append(f"## Formulas\n")
        for formula in article['formulas']:
            lines.append(f"- `{formula['formula_text']}`\n")

    # References
    if article.get('references'):
        lines.append(f"## References\n")
        for ref in article['references']:
            authors = ref.get('authors', '')
            year = ref.get('year', '')
            title = ref.get('title', '')
            journal = ref.get('journal', '')
            lines.append(f"- {authors} ({year}). {title}. *{journal}*\n")

    return '\n'.join(lines)
```

---

## All Parameters Reference

### `scipdf.parse_pdf()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_path` | str or bytes | — | Path to PDF, URL, or PDF bytes |
| `fulltext` | bool | `True` | Full text extraction (`True`) or header only (`False`) |
| `soup` | bool | `False` | Return BeautifulSoup (`True`) or raw XML string (`False`) |
| `return_coordinates` | bool | `False` | Include spatial coordinates in output |
| `grobid_url` | str | `"http://localhost:8070"` | GROBID server endpoint |

### `scipdf.parse_pdf_to_dict()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_path` | str | — | Path to PDF or URL |
| `fulltext` | bool | `True` | Full text or header only |
| `soup` | bool | `True` | Convert XML to BeautifulSoup |
| `as_list` | bool | `False` | Section text as paragraph list vs joined string |
| `return_coordinates` | bool | `True` | Include spatial coordinates |
| `grobid_url` | str | `"http://localhost:8070"` | GROBID server endpoint |

### `scipdf.parse_figures()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdf_folder` | str | — | Folder containing **only** PDF files |
| `jar_path` | str | (bundled) | Path to pdffigures2 JAR |
| `resolution` | int | `300` | Output DPI for extracted images |
| `output_folder` | str | `"figures"` | Base output directory |

---

## Input Formats

SciPDF Parser accepts three input types:

```python
# 1. Local file path
article = scipdf.parse_pdf_to_dict('/path/to/paper.pdf')

# 2. URL (must end in .pdf)
article = scipdf.parse_pdf_to_dict('https://arxiv.org/pdf/2512.24601v2.pdf')

# 3. Byte string (via parse_pdf directly)
with open('paper.pdf', 'rb') as f:
    pdf_bytes = f.read()
xml = scipdf.parse_pdf(pdf_bytes, soup=True)
```

---

## Key Differences from Other Parsers

| Feature | SciPDF Parser | Marker / Docling / MarkItDown |
|---------|--------------|-------------------------------|
| **Output** | Structured Python dict | Markdown text |
| **Backend** | GROBID (Docker service) | Local ML models |
| **Requires** | Docker running | Nothing (self-contained) |
| **Sections** | Separate objects with cross-refs | Flat heading + text |
| **References** | Parsed into title/author/year/journal | Raw text |
| **Figures** | Captions + metadata objects | Inline in markdown |
| **Tables** | Detected as figure objects with data | Markdown pipe tables |
| **Formulas** | Separate objects with coordinates | Inline Unicode/LaTeX |
| **Math rendering** | Plain text only | Unicode symbols / LaTeX |
| **Page selection** | Cannot limit to page range | Page range supported |
| **Inline formatting** | None (plain text) | Bold, italic, superscript |
| **Hyperlinks** | None | Cross-references, URLs |

---

## Tested Code & Benchmarks

**Paper:** "Recursive Language Models" (2512.24601v2), full PDF (19 pages — SciPDF cannot limit page range)
**Machine:** Apple M2 Pro, 16GB unified RAM, macOS
**GROBID:** v0.7.3 Docker (x86 image via Rosetta on Apple Silicon)

### Benchmark Results

| Tier | Time | Output (chars) | Lines | Sections | Refs | Figures | Formulas |
|------|------|----------------|-------|----------|------|---------|----------|
| **3** (text + coords) | 58.9s | 94,853 | 452 | 41 | 41 | 22 | 1 |
| **4** (+ figure images) | +0s* | 94,853 | 452 | (same) | (same) | (same) | (same) |
| **5** (+ text analysis) | +3.9s | 99,601 | 536 | (same) | (same) | (same) | (same) |

*Tier 4 figure extraction produced 0 images because Java was not installed. The text output is identical to Tier 3.

### Tested Code — Tier 3 (Full Text + Coordinates)

```python
import scipdf
import json
import time

paper = '/path/to/2512.24601v2.pdf'

t0 = time.time()
article = scipdf.parse_pdf_to_dict(
    paper,
    fulltext=True,
    return_coordinates=True,
    grobid_url='http://localhost:8070',
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")

# Save as JSON
with open('output.json', 'w') as f:
    json.dump(article, f, indent=2)

print(f"Sections: {len(article.get('sections', []))}")
print(f"References: {len(article.get('references', []))}")
print(f"Figures: {len(article.get('figures', []))}")
print(f"Formulas: {len(article.get('formulas', []))}")
```

### Tested Code — Tier 4 (+ Figure Image Extraction)

```python
import scipdf
import shutil
import tempfile

# Step 1: Parse text (same as Tier 3)
article = scipdf.parse_pdf_to_dict(
    '/path/to/paper.pdf',
    fulltext=True,
    return_coordinates=True,
    grobid_url='http://localhost:8070',
)

# Step 2: pdffigures2 needs a folder with ONLY PDFs
tmpdir = tempfile.mkdtemp()
shutil.copy('/path/to/paper.pdf', tmpdir)

try:
    scipdf.parse_figures(
        tmpdir,
        output_folder='figures_output',
        resolution=300,
    )
    # Creates: figures_output/data/ and figures_output/figures/
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
```

### Tested Code — Tier 5 (+ Text Analysis)

```python
import scipdf
from scipdf.features import compute_readability_stats, compute_text_stats
from scipdf.features.text_utils import compute_journal_features  # NOT in scipdf.features.__init__
import spacy

nlp = spacy.load("en_core_web_sm")

# Step 1: Parse the paper
article = scipdf.parse_pdf_to_dict(
    '/path/to/paper.pdf',
    fulltext=True,
    return_coordinates=True,
    grobid_url='http://localhost:8070',
)

# Step 2: Readability (on abstract)
readability = compute_readability_stats(article['abstract'])
# Result: Flesch 31.1, Grade 15-16 (academic/difficult)

# Step 3: POS tagging
doc = nlp(article['abstract'])
text_stats = compute_text_stats(doc)

# Step 4: Journal features
journal_features = compute_journal_features(article)
# Result: 41 refs, avg year 2024

# Step 5: Per-section readability
for section in article.get('sections', []):
    text = section.get('text', '')
    if len(text) > 50:
        section_readability = compute_readability_stats(text)
        print(f"{section['heading']}: Flesch {section_readability['flesch_reading_ease']}")
```

### Sample Output (Tier 5 — Abstract Readability)

| Metric | Value |
|--------|-------|
| flesch_reading_ease | 31.1 |
| flesch_kincaid_grade | 15.1 |
| smog | 17.1 |
| gunning_fog | 19.2 |
| dale_chall | 11.3 |
| text_standard | 15th and 16th grade |
| avg_sentence_length | 30.5 |
| difficult_words | 42 |

---

## Known Issues

### 1. `compute_journal_features` not exported
The `__init__.py` in `scipdf.features` lists `compute_journal_features` in `__all__` but doesn't import it. Use the direct import:
```python
# Wrong — raises ImportError:
from scipdf.features import compute_journal_features

# Correct:
from scipdf.features.text_utils import compute_journal_features
```

### 2. XML parser warning
GROBID returns TEI XML but scipdf parses it with `lxml` in HTML mode, producing a warning:
```
XMLParsedAsHTMLWarning: It looks like you're using an HTML parser to parse an XML document.
```
This is harmless — parsing still works. Suppress with:
```python
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
```

### 3. `serve_grobid.sh` ARM tag doesn't exist
The script tries `grobid/grobid:0.7.3-arm` on ARM machines, but this tag doesn't exist on Docker Hub. On Apple Silicon, run the standard image directly:
```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 -d grobid/grobid:0.7.3
```

### 4. GROBID startup time
After starting the Docker container, GROBID takes ~50 seconds to initialize its ML models before it can serve requests. Check readiness with:
```bash
curl http://localhost:8070/api/isalive
```

### 5. pdffigures2 requires Java
Figure image extraction (Tier 4) silently fails if Java is not installed — it creates empty output directories but extracts nothing. Install with:
```bash
brew install openjdk
```

### 6. FutureWarning in journal features
`compute_journal_features` triggers a pandas `FutureWarning` about `pd.unique()` on a plain list. Harmless but noisy.

---

## Known Limitations

1. **No page range selection** — GROBID processes the entire PDF. You cannot limit to specific pages (unlike Marker/Docling).
2. **No markdown output** — produces structured dicts, not readable markdown. Must convert yourself.
3. **No inline formatting** — all text is plain (no bold, italic, superscript).
4. **No math rendering** — formulas are extracted as plain text, not LaTeX or Unicode math.
5. **Requires Docker** — GROBID must be running as a Docker container (or use cloud service).
6. **pdffigures2 requires Java** — figure image extraction needs a JVM installed.
7. **20-second timeout** on pdffigures2 figure extraction — hardcoded in source.
8. **GROBID v0.7.3** in the bundled script — the README recommends updating to the latest version for improvements.
9. **x86 emulation on Apple Silicon** — GROBID Docker image is x86-only, runs via Rosetta. This adds overhead (58.9s for 19 pages vs native speed).
