# MarkItDown: arXiv Paper Parsing Guide

**Tool:** [MarkItDown](https://github.com/microsoft/markitdown) by Microsoft
**Version:** 0.1.5b1 (beta)
**License:** MIT
**Python:** 3.10+

MarkItDown is a lightweight Python utility that converts files to Markdown. Unlike Marker (which runs deep-learning OCR models locally), MarkItDown extracts embedded text from PDFs using `pdfminer.six` and `pdfplumber` — no neural models, no GPU required. This makes it extremely fast but limited to well-formed digital PDFs (no scanned documents).

---

## Quick Reference

| Tier | Method | Speed | Quality | Cost | Best For |
|------|--------|-------|---------|------|----------|
| 1 | Default CLI | ~seconds | Basic | Free | Quick text extraction |
| 2 | Python API (tuned) | ~seconds | Better | Free | Tables + structured text |
| 3 | + LLM image captions | ~seconds + API | Enhanced | API cost | When image context matters |
| 4 | Azure Document Intelligence | ~10-30s | Highest | Azure cost | Scanned PDFs, formulas, OCR |

---

## Installation

```bash
# Minimal (no PDF support)
pip install markitdown

# With PDF support (required for arXiv papers)
pip install 'markitdown[pdf]'

# With all optional dependencies
pip install 'markitdown[all]'

# Specific extras
pip install 'markitdown[pdf,docx,pptx,xlsx]'
```

### Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `pdf` | `pdfminer.six`, `pdfplumber` | PDF text and table extraction |
| `docx` | `mammoth`, `lxml` | Word documents |
| `xlsx` | `pandas`, `openpyxl` | Modern Excel files |
| `xls` | `pandas`, `xlrd` | Legacy Excel files |
| `pptx` | `python-pptx` | PowerPoint files |
| `outlook` | `olefile` | Outlook .msg emails |
| `audio-transcription` | `pydub`, `SpeechRecognition` | Audio speech-to-text |
| `youtube-transcription` | `youtube-transcript-api` | YouTube transcripts |
| `az-doc-intel` | `azure-ai-documentintelligence`, `azure-identity` | Azure cloud OCR |
| `all` | All of the above | Everything |

---

## Tier 1: Default CLI (Fastest, Basic)

The simplest approach — just pipe a PDF through the CLI.

### Command

```bash
# Basic conversion
markitdown paper.pdf

# Save to file
markitdown paper.pdf -o output.md

# Pipe from stdin
cat paper.pdf | markitdown -x .pdf > output.md
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `-o, --output FILE` | Output file path (default: stdout) |
| `-x, --extension EXT` | File extension hint (needed for stdin) |
| `-m, --mime-type TYPE` | MIME type hint |
| `-c, --charset CHARSET` | Charset hint |
| `--keep-data-uris` | Keep base64-encoded images in output (default: truncated) |
| `-p, --use-plugins` | Enable third-party plugins |
| `--list-plugins` | List installed plugins |
| `-d, --use-docintel` | Use Azure Document Intelligence |
| `-e, --endpoint URL` | Azure endpoint (with `-d`) |

### What You Get

- Embedded PDF text extracted via `pdfminer.six`
- Basic table detection via `pdfplumber` (word position clustering)
- Markdown headings, paragraphs, links preserved
- Normalized whitespace and newlines

### What You Don't Get

- No images extracted
- No OCR (scanned PDFs will produce empty output)
- No LaTeX math recognition
- No figure extraction
- No page boundaries
- No equation rendering

### Expected Performance

- **Speed:** 1-5 seconds for a typical 10-page paper
- **Output:** Plain text with basic table structure
- **Size:** Small (text only, no images)

### Example

```bash
markitdown /path/to/2512.24601v2.pdf -o tier1.md
```

---

## Tier 2: Python API (Better Control)

Using the Python API gives you access to the full `MarkItDown` object and its configuration options.

### Code

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("/path/to/paper.pdf")

# Access results
print(result.title)       # Extracted title (if available)
print(result.markdown)    # Full markdown content

# Save to file
with open("output.md", "w") as f:
    f.write(result.markdown)
```

### With Data URI Preservation

By default, any embedded images are truncated to `data:image/png;base64,...`. To keep full base64-encoded images:

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("/path/to/paper.pdf", keep_data_uris=True)
```

### Stream-Based Conversion

For more control over input handling:

```python
from markitdown import MarkItDown, StreamInfo

md = MarkItDown()

with open("paper.pdf", "rb") as f:
    result = md.convert_stream(f, stream_info=StreamInfo(
        extension=".pdf",
        mimetype="application/pdf"
    ))
```

### URL-Based Conversion

Convert directly from a URL (e.g., arXiv PDF link):

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("https://arxiv.org/pdf/2512.24601v2")
```

### How PDF Extraction Works Internally

MarkItDown's PDF converter uses a dual strategy:

1. **pdfplumber (form/table detection):**
   - Extracts words with X/Y positions from each page
   - Clusters words by Y-position into rows
   - Analyzes X-position distribution to detect column boundaries
   - If structured columns found (≥3 columns), renders as markdown table
   - Rejects false positives (standard paragraph text)

2. **pdfminer.six (text extraction):**
   - Fallback for pages without table structure
   - Extracts text in reading order
   - Preserves paragraph breaks

3. **Strategy selection per document:**
   - Counts pages with detected forms vs plain text
   - If more plain-text pages → uses pdfminer for entire document
   - Otherwise → uses pdfplumber chunks
   - Falls back to pdfminer if pdfplumber fails entirely

### What This Adds Over Tier 1

- Same extraction quality, but with programmatic access
- URL-based conversion (fetch arXiv PDFs directly)
- Data URI control for embedded images
- Better integration into pipelines

---

## Tier 3: LLM-Enhanced (Image Descriptions)

Add an LLM client to generate descriptions for images found in documents. This is most useful for PPTX files with embedded images, but has limited benefit for PDFs since MarkItDown doesn't extract PDF images.

### Code

```python
from openai import OpenAI
from markitdown import MarkItDown

client = OpenAI(api_key="sk-...")

md = MarkItDown(
    llm_client=client,
    llm_model="gpt-4o",
    llm_prompt="Write a detailed description of this image, focusing on any text, charts, or diagrams visible."
)

result = md.convert("/path/to/paper.pdf")
```

### Custom Prompt Examples

For arXiv papers with figures:

```python
# For architecture diagrams
llm_prompt = "Describe this figure from a research paper. Include any axis labels, data points, model names, and key takeaways visible in the image."

# For code screenshots
llm_prompt = "Transcribe all code visible in this image. Preserve formatting and indentation."

# For mathematical figures
llm_prompt = "Describe this mathematical figure, including all equations, variables, and relationships shown."
```

### Supported LLM Providers

MarkItDown uses the OpenAI client interface, so any OpenAI-compatible API works:

```python
# OpenAI
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# Azure OpenAI
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key="...",
    api_version="2024-02-15-preview",
    azure_endpoint="https://your-resource.openai.azure.com"
)

# Any OpenAI-compatible API (Ollama, vLLM, etc.)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

### Limitations for PDFs

- MarkItDown's PDF converter does **not** extract images from PDFs
- LLM captions only work for formats that produce images (PPTX, standalone image files)
- For PDF images, you'd need to extract them separately first
- This tier is more valuable for PPTX, DOCX, and standalone image files

### What This Adds Over Tier 2

- LLM-generated descriptions for images (in supported formats)
- Custom prompt engineering for domain-specific content
- Better for multi-format pipelines (not just PDFs)

---

## Tier 4: Azure Document Intelligence (Highest Quality)

Microsoft's cloud-based document analysis service provides OCR, formula extraction, and high-resolution processing — the closest to Marker's Tier 4/5 quality.

### Setup

1. Create an Azure Document Intelligence resource in the [Azure Portal](https://portal.azure.com)
2. Get the endpoint URL and API key
3. Install the dependency:

```bash
pip install 'markitdown[az-doc-intel]'
```

### CLI Usage

```bash
markitdown paper.pdf -d -e "https://your-resource.cognitiveservices.azure.com/"
```

> Note: The CLI uses `DefaultAzureCredential` for authentication. Set up Azure credentials via environment variables or Azure CLI login.

### Python API Usage

```python
from markitdown import MarkItDown
from azure.identity import DefaultAzureCredential
# Or: from azure.core.credentials import AzureKeyCredential

endpoint = "https://your-resource.cognitiveservices.azure.com/"
credential = DefaultAzureCredential()
# Or: credential = AzureKeyCredential("your-api-key")

md = MarkItDown(
    docintel_endpoint=endpoint,
    docintel_credential=credential,
    docintel_api_version="2024-07-31-preview",  # default
)

result = md.convert("/path/to/paper.pdf")
```

### Features Enabled for PDFs

When using Document Intelligence with PDFs/images, these features are automatically enabled:

| Feature | Description |
|---------|-------------|
| `FORMULAS` | Extracts and renders mathematical formulas |
| `OCR_HIGH_RESOLUTION` | High-resolution OCR for better text accuracy |
| `STYLE_FONT` | Detects font styles (bold, italic, headings) |

### What This Adds Over Tier 2

- **OCR capability** — works on scanned PDFs (not just digital)
- **Formula extraction** — mathematical equations rendered properly
- **High-res OCR** — better text accuracy for complex layouts
- **Font style detection** — proper heading hierarchy from font sizes
- **Cloud processing** — offloads compute to Azure

### Cost

Azure Document Intelligence pricing (as of 2025):
- **Read model:** ~$1.50 per 1,000 pages
- **Layout model:** ~$10.00 per 1,000 pages
- Free tier available (500 pages/month)

### Limitations

- Requires Azure subscription and internet connectivity
- Data is sent to Microsoft's cloud for processing
- Latency depends on document size and network speed
- API version may affect available features

---

## Comparison with Marker

| Feature | MarkItDown | Marker |
|---------|-----------|--------|
| **Approach** | Text extraction (pdfminer/pdfplumber) | Deep-learning OCR (Surya models) |
| **GPU required** | No | No (but much faster with GPU/MPS) |
| **Speed (10-page PDF)** | 1-5 seconds | 2-17 minutes |
| **OCR capability** | Only via Azure | Built-in (Surya) |
| **Math/LaTeX** | Only via Azure | Built-in equation recognition |
| **Image extraction** | No (PDF) | Yes (JPEG/PNG) |
| **Table extraction** | Basic (position clustering) | ML-based (TableRecModel) |
| **Figure content** | Not extracted | OCR'd from figures |
| **Page boundaries** | No | Optional (`paginate_output`) |
| **LLM integration** | Image captions only | Text correction, table fixing, equation cleanup |
| **Local processing** | Yes (except Azure tier) | Yes (all tiers) |
| **Cost** | Free (except Azure/LLM) | Free (except LLM tier) |
| **Best for** | Quick text extraction from clean digital PDFs | High-fidelity conversion with figures/math |

### When to Use MarkItDown Over Marker

- You need **speed** — MarkItDown is 100-1000x faster for text extraction
- The PDF is **clean digital text** (not scanned) and you only need the body text
- You're building a **pipeline** that processes many file types (not just PDFs)
- You want **minimal dependencies** — no PyTorch, no GPU drivers
- You need to convert **URLs directly** (arXiv, web pages, etc.)

### When to Use Marker Over MarkItDown

- You need **figures and images** extracted
- You need **LaTeX math** rendering
- The PDF has **complex tables** that need ML-based recognition
- The PDF is **scanned** (OCR required) and you can't use Azure
- You need content **from inside figures** (code snippets, diagrams)
- You need **page-level structure**

---

## Batch Processing

MarkItDown has no built-in batch mode, but it's easy to script:

### Shell Loop

```bash
for pdf in papers/*.pdf; do
    name=$(basename "$pdf" .pdf)
    markitdown "$pdf" -o "output/${name}.md"
    echo "Converted: $pdf"
done
```

### Python Parallel Processing

```python
from markitdown import MarkItDown
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def convert_paper(pdf_path):
    md = MarkItDown()
    result = md.convert(str(pdf_path))
    out_path = Path("output") / f"{pdf_path.stem}.md"
    out_path.write_text(result.markdown)
    return str(pdf_path)

papers = list(Path("papers").glob("*.pdf"))
with ThreadPoolExecutor(max_workers=8) as executor:
    for done in executor.map(convert_paper, papers):
        print(f"Done: {done}")
```

---

## Plugin System

MarkItDown supports third-party plugins for additional file formats.

### Using Plugins

```bash
# List installed plugins
markitdown --list-plugins

# Use plugins during conversion
markitdown -p file.rtf -o output.md
```

```python
md = MarkItDown(enable_plugins=True)
result = md.convert("file.rtf")
```

### Creating a Plugin

A plugin must:
1. Define `__plugin_interface_version__ = 1`
2. Implement `register_converters(markitdown, **kwargs)`
3. Register as a `markitdown.plugin` entry point

```python
# my_plugin/__init__.py
from markitdown import MarkItDown, DocumentConverter, DocumentConverterResult

class MyConverter(DocumentConverter):
    def accepts(self, file_stream, stream_info, **kwargs):
        return stream_info.extension == ".xyz"

    def convert(self, file_stream, stream_info, **kwargs):
        text = file_stream.read().decode()
        return DocumentConverterResult(markdown=text)

__plugin_interface_version__ = 1

def register_converters(markitdown: MarkItDown, **kwargs):
    markitdown.register_converter(MyConverter())
```

```toml
# pyproject.toml
[project.entry-points."markitdown.plugin"]
my_plugin = "my_plugin"
```

---

## arXiv-Specific Tips

### Direct URL Conversion

```python
from markitdown import MarkItDown

md = MarkItDown()
# Convert directly from arXiv URL
result = md.convert("https://arxiv.org/pdf/2512.24601v2")
```

### Limitations for arXiv Papers

1. **No page selection** — MarkItDown processes the entire PDF; there's no `--page_range` equivalent
2. **No image extraction** — figures, charts, and diagrams are lost
3. **No math rendering** — equations appear as garbled Unicode or are missing entirely
4. **Basic tables** — pdfplumber's position-based clustering works for simple tables but struggles with complex multi-row/multi-column academic tables
5. **Two-column layouts** — pdfminer generally handles two-column text well, but column merging can occur

### Recommended Workflow

For a quick first pass of an arXiv paper:

```bash
# Fast text extraction (seconds)
markitdown paper.pdf -o quick_text.md
```

For higher quality (if Azure is available):

```bash
# Azure Document Intelligence (cloud OCR + formulas)
markitdown paper.pdf -d -e "https://your-resource.cognitiveservices.azure.com/" -o high_quality.md
```

For best results, combine tools:

```bash
# Use MarkItDown for quick text, Marker for figures/math
markitdown paper.pdf -o text_only.md
marker_single paper.pdf --output_dir ./marker_output --force_ocr
# Then merge: text from MarkItDown, images/math from Marker
```

---

## Summary

MarkItDown is the right tool when you need **fast, lightweight text extraction** from clean digital PDFs. It processes papers in seconds rather than minutes, requires no GPU or ML models, and integrates easily into larger pipelines.

For arXiv paper parsing specifically:
- **Text only?** → MarkItDown (Tier 1-2) — fast and accurate
- **Need math/figures?** → Use Marker instead, or MarkItDown Tier 4 (Azure)
- **Scanned PDF?** → MarkItDown Tier 4 (Azure) or Marker with `--force_ocr`
- **Processing hundreds of papers?** → MarkItDown for text extraction at scale, Marker for select papers needing full fidelity
