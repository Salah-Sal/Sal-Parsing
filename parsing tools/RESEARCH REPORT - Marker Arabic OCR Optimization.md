# Research Report: Optimizing marker-pdf OCR for Arabic Dictionary Pages

A deep technical investigation into improving OCR accuracy for **Al-Mu'jam al-Wasit** (المعجم الوسيط) — a 1099-page scanned Arabic dictionary with full tashkeel/diacritics — using [marker-pdf](https://github.com/datalab-to/marker) v1.10.2 and its underlying [surya-ocr](https://github.com/VikParuchuri/surya) engine.

**Current baseline:** ~240+ errors across 10 test pages, ~70-75% word-level accuracy.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Q1: Model Selection — Surya, Cloud, or Hybrid?](#q1-model-selection--surya-cloud-or-hybrid)
- [Q2: Language Hinting — Does It Exist?](#q2-language-hinting--does-it-exist)
- [Q3: DPI and Resolution Settings](#q3-dpi-and-resolution-settings)
- [Q4: Multi-Column RTL Layout Detection](#q4-multi-column-rtl-layout-detection)
- [Q5: Tashkeel / Diacritics Handling](#q5-tashkeel--diacritics-handling)
- [Q6: force_ocr and Scanned PDF Handling](#q6-force_ocr-and-scanned-pdf-handling)
- [Q7: Post-Processing and LLM Correction](#q7-post-processing-and-llm-correction)
- [Q8: Alternative OCR Backends](#q8-alternative-ocr-backends)
- [Q9: Known Arabic Issues in Marker / Surya](#q9-known-arabic-issues-in-marker--surya)
- [Q10: Other Relevant Configuration Options](#q10-other-relevant-configuration-options)
- [Risk Areas: Regex and Text Processing](#risk-areas-regex-and-text-processing)
- [Recommended Configuration Tiers](#recommended-configuration-tiers)
- [Recommended Strategy](#recommended-strategy)

---

## Executive Summary

After a thorough investigation of the marker-pdf codebase (v1.10.2), surya-ocr source code, GitHub issues, and the broader Arabic OCR landscape, the key findings are:

1. **Surya's Arabic support is basic.** Arabic (`ar`) is listed as a supported language, but there is no language-specific hinting mechanism, no Arabic-specific fonts in the recognition pipeline, and no diacritics-aware processing. GitHub issues confirm inconsistent Arabic quality.

2. **No silver bullet in marker's configuration.** Increasing DPI, forcing OCR, and tuning batch sizes will provide marginal improvements but will not bridge the gap from 75% to 95%+ accuracy for diacritically-dense Arabic.

3. **LLM post-correction is the most promising marker-native approach.** The `--use_llm` flag with a custom `block_correction_prompt` specifically instructing Arabic/tashkeel correction can substantially improve output quality.

4. **For production quality, consider alternative pipelines.** QARI-OCR (state-of-the-art for diacritized Arabic, WER 0.160, CER 0.061), Google Cloud Vision API, or a hybrid approach (marker for layout + specialized Arabic OCR for text) are likely to significantly outperform surya on this specific task.

5. **Several regex patterns in marker's processors could damage Arabic text.** These are documented in the [Risk Areas](#risk-areas-regex-and-text-processing) section and may need patching for Arabic-heavy workloads.

---

## Q1: Model Selection — Surya, Cloud, or Hybrid?

### Surya's Architecture

Surya uses a unified foundation model architecture for OCR (`FOUNDATION_MODEL_CHECKPOINT: s3://text_recognition/2025_09_23`). The recognition pipeline:

1. **DetectionPredictor** — detects text line bounding boxes
2. **RecognitionPredictor** — recognizes text within detected boxes using a foundation model
3. **OCRErrorPredictor** — post-hoc error detection model

The `RecognitionPredictor.__call__()` signature takes images and task names — but **no language parameter**. The model is a single multilingual model that handles all supported languages without explicit language routing.

**Source:** `surya/recognition/__init__.py` — the `RecognitionPredictor` class has no `language` or `lang` parameter.

### Cloud Options via Marker

Marker v1.10.2 supports LLM-enhanced processing via:

| Service | Class | Default Model | Arabic Quality |
|---------|-------|---------------|----------------|
| Google Gemini | `marker.services.gemini.GoogleGeminiService` | `gemini-2.0-flash` | Good — strong multilingual, vision capable |
| Claude | `marker.services.claude.ClaudeService` | `claude-3-7-sonnet-20250219` | Very Good — excellent Arabic comprehension |
| OpenAI | `marker.services.openai.OpenAIService` | `gpt-4o-mini` | Good — decent multilingual |
| Ollama | `marker.services.ollama.OllamaService` | `llama3.2-vision` | Poor — limited Arabic diacritics |

### Hybrid Recommendation

The most effective approach for this dictionary:

```
Marker (layout detection + structure) → Custom Arabic OCR (text recognition) → LLM (post-correction)
```

Marker's layout model (LayoutPredictor) and structure builders are language-agnostic and effective for detecting columns, headings, and text blocks. The weak link is surya's text recognition for diacritized Arabic. Replacing just that component would yield the biggest improvement.

---

## Q2: Language Hinting — Does It Exist?

**Short answer: No.**

### What Was Investigated

- **Surya's RecognitionPredictor** (`surya/recognition/__init__.py`): The `__call__` method accepts images and optional `task_names` (e.g., `"ocr_with_boxes"`, `"ocr_without_boxes"`). There is no `language`, `lang`, or `script` parameter.

- **Surya's languages.py** (`surya/recognition/languages.py`): Contains a `CODE_TO_LANGUAGE` dict mapping `"ar"` to `"Arabic"` and a reverse `LANGUAGE_TO_CODE` dict. These are defined but **never referenced** in the actual recognition pipeline — they appear to be metadata only.

- **Marker's OcrBuilder** (`marker/builders/ocr.py`): Passes images to surya's recognition model. No language parameter is forwarded.

- **Marker's LineBuilder** (`marker/builders/line.py`): No language configuration.

### What `ocr_task_name` Does

The closest thing to a mode switch is `ocr_task_name` in OcrBuilder:
- `"ocr_with_boxes"` (default) — recognition within detected bounding boxes
- `"ocr_without_boxes"` — full-page recognition without line-level detection

For Arabic with dense diacritics, `"ocr_without_boxes"` *might* perform differently (the model processes the full image rather than cropped line regions), but this is speculative and untested for Arabic.

### Implication

You cannot tell surya "this is Arabic text" to activate any Arabic-specific processing path. The model must figure out the script from the image alone.

---

## Q3: DPI and Resolution Settings

### Default Values

| Setting | Location | Default | Used For |
|---------|----------|---------|----------|
| `lowres_image_dpi` | `marker/builders/document.py` | 96 | Layout detection, reading order |
| `highres_image_dpi` | `marker/builders/document.py` | 192 | OCR text recognition |
| `IMAGE_DPI` | `surya/settings.py` | 96 | Surya internal (detection, layout) |
| `IMAGE_DPI_HIGHRES` | `surya/settings.py` | 192 | Surya internal (OCR, table rec) |

### Why This Matters for Arabic Diacritics

Arabic diacritical marks (tashkeel) are **tiny glyphs** positioned above or below base letters:

- Fathah (فَتْحَة) — small diagonal stroke above
- Kasrah (كَسْرَة) — small diagonal stroke below
- Dammah (ضَمَّة) — small waw-like mark above
- Shaddah (شَدَّة) — small w-like mark above
- Sukun (سُكُون) — small circle above
- Tanween forms — doubled marks

At 192 DPI, these marks can be as small as 2-4 pixels. This is insufficient for reliable recognition, especially in dictionary-sized fonts (typically 8-10pt).

### Recommended DPI Settings

```json
{
    "highres_image_dpi": 400,
    "lowres_image_dpi": 200
}
```

**Rationale:**
- **400 DPI for OCR**: Diacritical marks need ~6-10 pixels to be distinguishable. For 8pt Arabic text, this requires approximately 300-400 DPI. Going above 400 yields diminishing returns and significantly increases memory/processing time.
- **200 DPI for layout**: Higher layout resolution helps detect the small gaps between dictionary columns and between entries.

**Trade-offs:**
- Processing time increases roughly quadratically with DPI (4x area at 2x DPI)
- Memory usage increases proportionally
- At 400 DPI, a single dictionary page (~B5 size) becomes a ~2600x3700 pixel image
- You will need to reduce batch sizes to compensate for memory

### Surya-Level DPI Override

You can also set surya's DPI directly via environment variables:

```bash
export IMAGE_DPI_HIGHRES=400
export IMAGE_DPI=200
```

This affects surya regardless of marker's settings.

---

## Q4: Multi-Column RTL Layout Detection

### How Marker Handles Columns

Marker's column detection happens in the **LayoutPredictor** (surya's layout model). The model outputs bounding boxes with block type labels. For multi-column layouts, the model should detect separate text blocks for each column.

Key configuration in `marker/builders/layout.py`:
- `force_layout_block`: Can force the entire page to be treated as a single block type (e.g., `"Text"`). This is **counterproductive** for multi-column — do not use it.
- `expand_block_types`: Block types whose bounding boxes get expanded (default includes `"Text"`)
- `max_expand_frac`: Maximum expansion as fraction of page (default `0.05`)

### Reading Order

Reading order is determined by surya's `LayoutPredictor` which assigns order indices to detected blocks. For RTL documents, the reading order should naturally detect right-column-first ordering, but this is **not guaranteed** for Arabic dictionaries.

### Column Merging Issue

The user reported column merging as a specific error pattern. Possible causes:

1. **Layout model doesn't detect the column gap** — the gap between columns in dense dictionaries may be too narrow for the model at 96 DPI.
2. **Block expansion merges columns** — if `max_expand_frac` is too large, adjacent column blocks could merge.
3. **Post-processing reorders blocks** — `SectionProcessor` or `ColumnProcessor` might incorrectly merge blocks.

### Recommended Settings

```json
{
    "lowres_image_dpi": 200,
    "max_expand_frac": 0.02,
    "column_gap_ratio": 0.01
}
```

Additionally, running with `--debug` saves layout visualization images that show exactly how the layout model segments the page. This is essential for diagnosing column issues:

```bash
marker_single dictionary.pdf --debug --output_dir ./debug --page_range "0-2"
```

Check the debug output to see if columns are being detected as separate blocks.

### RTL Rendering

Marker's markdown renderer (`marker/renderers/markdown.py`) outputs standard markdown. RTL text direction must be handled at the rendering level (CSS `direction: rtl;`). Marker does not add RTL markers to its output. If using the HTML output format, you may need to wrap output in `<div dir="rtl">`.

---

## Q5: Tashkeel / Diacritics Handling

### Good News: No Active Diacritics Stripping

After thorough code review, **marker does not strip or normalize Unicode diacritical marks**. There is no call to `unicodedata.normalize('NFD', ...)` followed by category filtering, no explicit tashkeel removal, and no Arabic-specific text cleaning.

The `ftfy` library (used for text fixing in some processors) generally preserves combining characters in Arabic. The `markdownify` library used in the renderer also passes through Unicode combining characters.

### Risk: Regex Patterns

Several regex patterns in marker's processors use character classes that could interact poorly with Arabic diacritics. See [Risk Areas](#risk-areas-regex-and-text-processing) below.

### The Real Problem: Recognition Accuracy

The diacritics issue is not in post-processing — it's in **surya's recognition model**. The model was trained on multilingual data but Arabic diacritics are:

1. **Visually tiny** — easily confused with noise or scan artifacts
2. **Positionally dependent** — same mark looks different above ب vs ت vs ث
3. **Combinatorially complex** — each base letter can carry 0-3 marks
4. **Training data scarce** — fully diacritized Arabic text is relatively rare in training corpora

Surya's recognition model checkpoint (`s3://text_recognition/2025_09_23`) is a general-purpose multilingual model. It has no Arabic-specific training augmentation for diacritized text.

### Font Configuration Gap

Surya's settings define recognition render fonts:

```python
RECOGNITION_RENDER_FONTS = {
    "all": "GoNotoCurrent-Regular.ttf",
    "zh": "GoNotoCJKCore.ttf",
    "ja": "GoNotoCJKCore.ttf",
    "ko": "GoNotoCJKCore.ttf",
}
```

Notice: **no Arabic-specific font**. The `GoNotoCurrent-Regular.ttf` ("all" fallback) handles Arabic but is not optimized for it. CJK scripts get dedicated fonts but Arabic does not. This could affect recognition quality since these fonts are used in the recognition training/inference pipeline for rendering reference text.

---

## Q6: force_ocr and Scanned PDF Handling

### What force_ocr Does

Setting `force_ocr: true` (or `--force_ocr` CLI flag) affects the **LineBuilder** (`marker/builders/line.py`):

- **Without force_ocr**: Marker first tries to extract text from the PDF using `pdftext`. Only pages where extracted text coverage falls below `min_document_ocr_threshold` (default: 0.85, i.e., 85%) trigger OCR.
- **With force_ocr**: All pages are OCR'd regardless of embedded text quality.

**Source:** `marker/builders/line.py` — the `min_document_ocr_threshold` check is bypassed when `force_ocr=True`.

### Critical for Al-Mu'jam al-Wasit

For a **scanned PDF** (no embedded text layer), `force_ocr` is **essential**. Without it:

- If the PDF has no embedded text, marker will extract empty strings from `pdftext`
- The OCR threshold check should trigger OCR automatically, but explicitly setting `force_ocr: true` ensures consistent behavior
- Some scanned PDFs contain a garbage OCR layer from the scanning process — `force_ocr` bypasses this

### Recommended:

```json
{
    "force_ocr": true
}
```

Always use `force_ocr: true` for scanned Arabic dictionaries.

---

## Q7: Post-Processing and LLM Correction

### The Most Promising Marker-Native Approach

Marker's `LLMPageCorrectionProcessor` (`marker/processors/llm/llm_page_correction.py`) is the single most impactful tool for improving Arabic OCR quality within the marker pipeline.

### How It Works

1. Takes a page image + JSON of detected blocks (with normalized bounding boxes)
2. Sends both to a vision LLM
3. The LLM can:
   - **Rewrite block HTML content** (fix OCR errors)
   - **Reorder blocks** (fix reading order)
   - **Change block types** (e.g., `Text` → `SectionHeader`)
   - **Add/remove blocks**

### Custom Block Correction Prompt

The `block_correction_prompt` parameter lets you inject a custom prompt that gets appended to the system prompt sent to the LLM. This is where Arabic-specific instructions go.

```bash
marker_single dictionary.pdf \
    --use_llm \
    --force_ocr \
    --block_correction_prompt "$(cat arabic_correction_prompt.txt)" \
    --output_dir ./output
```

**Recommended `arabic_correction_prompt.txt`:**

```
You are correcting OCR output from a scanned Arabic dictionary (المعجم الوسيط).

CRITICAL RULES:
1. This is fully diacritized (مُشَكَّل) Arabic text. Every word should have complete tashkeel (فتحة، كسرة، ضمة، سكون، شدة، تنوين). If diacritics are missing or wrong, restore them based on the image.

2. Common OCR errors to watch for and fix:
   - Letter confusion: ب↔ت↔ث↔ن↔ي (dots above/below confused)
   - Letter confusion: ح↔خ↔ج (similar shapes)
   - Letter confusion: د↔ذ, ر↔ز, س↔ش, ص↔ض, ط↔ظ, ع↔غ
   - Diacritics errors: فَ↔فِ↔فُ, missing shaddah, wrong tanween
   - Arabic-Indic numerals: ٠١٢٣٤٥٦٧٨٩ (not Western 0123456789)
   - Hamza placement: أ↔إ↔ا↔آ↔ء↔ئ↔ؤ

3. Dictionary-specific formatting:
   - Headwords (entry words) are typically bold and fully diacritized
   - Root letters may appear in parentheses
   - Quranic verses are quoted and must be exact
   - Cross-references use specific abbreviations

4. This is a two-column RTL layout. Ensure right column comes before left column in reading order.

5. Preserve ALL diacritical marks exactly as shown in the image. Do not simplify or remove tashkeel.
```

### LLM Service Recommendation for Arabic

For Arabic dictionary correction, **Claude** (`marker.services.claude.ClaudeService`) or **Gemini** (`marker.services.gemini.GoogleGeminiService`) are recommended:

- Both have strong Arabic language understanding
- Both have vision capabilities to cross-reference OCR output against the page image
- Gemini Flash is cheaper for bulk processing
- Claude may have higher accuracy for diacritics-level corrections

```bash
# Using Gemini (cheaper, good quality)
export GOOGLE_API_KEY="your-key"
marker_single dictionary.pdf --use_llm --force_ocr \
    --block_correction_prompt "..." \
    --config_json arabic_config.json

# Using Claude (potentially higher Arabic accuracy)
export ANTHROPIC_API_KEY="your-key"
marker_single dictionary.pdf --use_llm --force_ocr \
    --llm_service marker.services.claude.ClaudeService \
    --block_correction_prompt "..." \
    --config_json arabic_config.json
```

### Cost Estimate for Full Dictionary

For 1099 pages with LLM correction:
- **Gemini Flash**: ~$0.02-0.05/page = **$22-55 total**
- **Claude Sonnet**: ~$0.05-0.15/page = **$55-165 total**
- **GPT-4o-mini**: ~$0.03-0.08/page = **$33-88 total**

These are rough estimates. Actual costs depend on token count per page (Arabic dictionary pages are text-dense).

### Concurrency

```json
{
    "max_concurrency": 5
}
```

Higher concurrency speeds up LLM processing but may hit rate limits. Start with 3-5.

---

## Q8: Alternative OCR Backends

### QARI-OCR — State of the Art for Arabic Diacritics

**QARI-OCR** is a specialized Arabic OCR system designed specifically for diacritically-rich texts (Quran, classical Arabic, dictionaries). Published benchmarks:

| Metric | QARI-OCR | Google Cloud Vision | Tesseract (Arabic) |
|--------|----------|--------------------|--------------------|
| WER (Word Error Rate) | **0.160** | ~0.25-0.35 | ~0.40-0.60 |
| CER (Character Error Rate) | **0.061** | ~0.10-0.15 | ~0.20-0.35 |

QARI-OCR is specifically trained on diacritized Arabic text and handles tashkeel far better than general-purpose OCR engines.

**Integration approach:** Use QARI-OCR for text recognition, marker for layout/structure.

### Google Cloud Vision API

- Good Arabic support, better than surya for diacritized text
- Can be accessed via marker's override system (but requires custom code)
- GitHub issue #221 discusses using Google Cloud Vision as an alternative backend
- Cost: ~$1.50 per 1000 pages

### Surya Finetuning

Surya includes a finetuning script at `surya/scripts/finetune_ocr.py` that uses HuggingFace's Trainer API. You could finetune surya on diacritized Arabic data:

**Requirements:**
- Training data: pairs of (image crops, ground truth text) for Arabic dictionary pages
- HuggingFace account for model hosting
- GPU with sufficient VRAM (16GB+ recommended)

**Community resources:**
- `Ollsoft-ai/surya-finetuning` — repository with finetuning workflows for surya
- `ketanmore/surya-ocr-arabic-segment` — HuggingFace dataset for Arabic surya finetuning

**Finetuning workflow:**
1. Prepare 500-2000 line-level image/text pairs from the dictionary
2. Ensure all ground truth text has correct tashkeel
3. Run `finetune_ocr.py` with appropriate hyperparameters
4. Point marker to the finetuned model checkpoint

This is the most technically demanding approach but could yield the best results within the marker/surya ecosystem.

### PaddleOCR

PaddleOCR has dedicated Arabic models and generally handles Arabic better than surya. It supports language specification (`lang='ar'`) and has Arabic-specific preprocessing. However, integrating PaddleOCR into marker's pipeline requires custom code.

### Tesseract with Arabic Training Data

Tesseract 5.x has Arabic models but performs poorly on diacritized text without custom training. Not recommended for this use case.

---

## Q9: Known Arabic Issues in Marker / Surya

### GitHub Issues

| Issue | Title | Key Finding |
|-------|-------|-------------|
| **#294** | Fine tuning for Arabic | Users report poor Arabic quality, request finetuning capability. Maintainer suggests surya finetuning. |
| **#28** | Arabic text detection | Early issue about Arabic text detection. Text detection (line-level) works reasonably well; recognition is the bottleneck. |
| **#221** | OCR using Google Cloud Vision | Discussion of using Google Cloud Vision as alternative backend for better non-Latin OCR. |
| **#581** | Arabic text works well for some | Inconsistent Arabic quality — works for some documents, fails for others. Likely correlated with scan quality and diacritization density. |

### Key Takeaway from Issues

Arabic support in surya is **inconsistent**. Simple, undiacritized Arabic (modern newspaper/web text) works reasonably well. Fully diacritized classical Arabic (dictionaries, Quran, poetry) is a known weak point. The maintainer acknowledges this and points to finetuning as the path forward.

### Progressive Degradation Pattern

The user reported that OCR accuracy degrades on later pages. Possible causes:

1. **`drop_repeated_text`** in `OcrBuilder` (`marker/builders/ocr.py`): When `true`, drops text that appears repeatedly across pages. In a dictionary, header/footer text (page numbers, section letters) repeats and could be dropped. Ensure this is `false`:
   ```json
   {
       "drop_repeated_text": false
   }
   ```

2. **Memory pressure**: As pages accumulate, the document model grows. On memory-constrained systems, this could degrade processing quality for later pages.

3. **Common element removal**: `HeaderFooterProcessor` removes text identified as headers/footers (text appearing in similar positions across many pages). In a dictionary, running headers (الباب, الفصل) could be incorrectly removed. Check with `--debug` output.

---

## Q10: Other Relevant Configuration Options

### Full Recommended Config for Arabic Dictionary

```json
{
    "force_ocr": true,
    "highres_image_dpi": 400,
    "lowres_image_dpi": 200,
    "ocr_task_name": "ocr_with_boxes",
    "drop_repeated_text": false,
    "max_expand_frac": 0.02,
    "detection_line_min_confidence": 0.6,
    "min_document_ocr_threshold": 0.5,
    "layout_coverage_threshold": 0.15,
    "recognition_batch_size": 16,
    "detection_batch_size": 4,
    "layout_batch_size": 4,
    "paginate_output": true,
    "extract_images": false
}
```

### Parameter Explanations

| Parameter | Value | Why |
|-----------|-------|-----|
| `force_ocr` | `true` | Scanned PDF — must OCR all pages |
| `highres_image_dpi` | `400` | Arabic diacritics need high resolution |
| `lowres_image_dpi` | `200` | Better column gap detection |
| `ocr_task_name` | `"ocr_with_boxes"` | Default; use detected line boxes. Try `"ocr_without_boxes"` if line detection is poor |
| `drop_repeated_text` | `false` | Prevent dictionary headers from being dropped |
| `max_expand_frac` | `0.02` | Prevent column merging from box expansion |
| `detection_line_min_confidence` | `0.6` | Lower threshold to catch more Arabic text lines (default 0.8 may miss low-confidence diacritized lines) |
| `min_document_ocr_threshold` | `0.5` | Not critical with force_ocr, but lower threshold as safety net |
| `layout_coverage_threshold` | `0.15` | Lower threshold to ensure small text blocks are captured |
| `recognition_batch_size` | `16` | Reduced from default to handle higher DPI images |
| `detection_batch_size` | `4` | Reduced for higher DPI |
| `layout_batch_size` | `4` | Reduced for higher DPI |
| `paginate_output` | `true` | Adds page markers — useful for a 1099-page dictionary |
| `extract_images` | `false` | Dictionary pages are text-only; skip image extraction |

### Environment Variables

```bash
# Force MPS (Apple Silicon) or CPU
export TORCH_DEVICE=mps  # or cpu

# Override surya's DPI directly
export IMAGE_DPI_HIGHRES=400
export IMAGE_DPI=200

# Increase logging for debugging
export LOGLEVEL=DEBUG
```

---

## Risk Areas: Regex and Text Processing

Several regex patterns in marker's processors could interfere with Arabic text. These are worth monitoring and potentially patching:

### 1. Footnote Processor (`marker/processors/footnote.py`)

```python
re.match(r"^[0-9\W]+", span.text)
```

**Risk:** `\W` (non-word character) in Python regex **matches Arabic diacritical marks** (combining characters like U+064E FATHAH, U+0650 KASRAH, etc.). A span starting with a diacritical mark (e.g., from a line break mid-word) would be incorrectly matched as a footnote marker.

### 2. Span Superscript Logic (`marker/schema/text/span.py`)

```python
re.sub(r"^([0-9\W]+)(.*)", r"<sup>\1</sup>\2", text)
```

**Risk:** Same `\W` issue — Arabic diacritics at the start of a span could be wrapped in `<sup>` tags, corrupting the text.

### 3. Latin-Centric Character Classes

Various processors use `[a-zA-Z]` or `\w` patterns:
- `\w` in Python matches Unicode word characters, which includes Arabic letters — this is generally safe
- `[a-zA-Z]` explicitly excludes Arabic — patterns that check "is this text?" using `[a-zA-Z]` would fail on Arabic-only text

### 4. Whitespace Normalization

The markdown renderer (`marker/renderers/markdown.py`) calls `process_text()` which normalizes whitespace. Arabic text uses different whitespace conventions (no spaces within words due to cursive script, but ZWNJ/ZWJ may be used). This normalization is generally safe but worth verifying.

### Recommended Patches for Arabic Workloads

If you encounter issues from the regex patterns above, modify:

**`marker/processors/footnote.py`:**
```python
# Change from:
re.match(r"^[0-9\W]+", span.text)
# To (exclude Arabic Unicode range):
re.match(r"^[0-9\s\p{P}]+", span.text)  # requires regex module
# Or simpler:
re.match(r"^[0-9\s.,;:!?()]+", span.text)  # explicit punctuation
```

**`marker/schema/text/span.py`:**
```python
# Change from:
re.sub(r"^([0-9\W]+)(.*)", r"<sup>\1</sup>\2", text)
# To:
re.sub(r"^([0-9\s.,;:!?]+)(.*)", r"<sup>\1</sup>\2", text)
```

---

## Recommended Configuration Tiers

### Tier A: Quick Test (No LLM, Local Only)

```bash
marker_single dictionary.pdf \
    --force_ocr \
    --output_dir ./output_tierA \
    --page_range "0-4" \
    --config_json arabic_tierA.json
```

`arabic_tierA.json`:
```json
{
    "force_ocr": true,
    "highres_image_dpi": 400,
    "lowres_image_dpi": 200,
    "drop_repeated_text": false,
    "max_expand_frac": 0.02,
    "detection_line_min_confidence": 0.6,
    "recognition_batch_size": 16,
    "detection_batch_size": 4,
    "layout_batch_size": 4,
    "extract_images": false,
    "paginate_output": true
}
```

**Expected improvement:** 5-15% over defaults (from higher DPI and better layout settings).

### Tier B: LLM-Enhanced (Best Within Marker)

```bash
export GOOGLE_API_KEY="your-key"

marker_single dictionary.pdf \
    --force_ocr \
    --use_llm \
    --block_correction_prompt "This is a fully diacritized Arabic dictionary (المعجم الوسيط). Preserve ALL tashkeel marks. Fix letter confusion (ب↔ت↔ث↔ن, ح↔خ↔ج, etc). Use Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩). Right column precedes left column. Headwords are bold with complete diacritics." \
    --output_dir ./output_tierB \
    --page_range "0-4" \
    --config_json arabic_tierB.json
```

`arabic_tierB.json`:
```json
{
    "force_ocr": true,
    "use_llm": true,
    "highres_image_dpi": 400,
    "lowres_image_dpi": 200,
    "drop_repeated_text": false,
    "max_expand_frac": 0.02,
    "max_concurrency": 5,
    "detection_line_min_confidence": 0.6,
    "recognition_batch_size": 16,
    "detection_batch_size": 4,
    "layout_batch_size": 4,
    "extract_images": false,
    "paginate_output": true
}
```

**Expected improvement:** 20-40% over defaults (LLM correction is the big lever).

### Tier C: Hybrid Pipeline (Best Overall Quality)

Use marker for layout detection, then replace OCR with a specialized Arabic engine:

1. Run marker with `--output_format json` to get block coordinates
2. For each text block, crop the high-res page image at that coordinate
3. Send cropped images to QARI-OCR or Google Cloud Vision
4. Reconstruct the document with corrected text

This requires custom scripting but is likely to achieve 90%+ accuracy on diacritized text.

---

## Recommended Strategy

### Phase 1: Baseline with Optimized Marker (1-2 hours)

1. Run Tier A config on 5 test pages
2. Run Tier B config on the same 5 pages
3. Compare error rates against your current baseline
4. Check `--debug` output for layout/column issues

### Phase 2: Evaluate LLM Correction Quality (2-4 hours)

1. If Tier B shows significant improvement, run on 20-50 pages
2. Categorize remaining errors:
   - If mostly diacritics errors → surya recognition is the bottleneck → move to Phase 3
   - If mostly layout/ordering errors → tune layout settings further
   - If mostly LLM hallucinations → refine the block_correction_prompt

### Phase 3: Alternative OCR Engine (if needed)

1. Test Google Cloud Vision API on 5 sample page images
2. Test QARI-OCR if available
3. Compare diacritics accuracy vs surya + LLM correction
4. If significantly better, build the hybrid pipeline (Tier C)

### Phase 4: Production Pipeline

1. Choose the best approach from Phases 1-3
2. Process all 1099 pages in batches (50-100 pages at a time)
3. Implement quality spot-checks every ~100 pages
4. Budget: ~$0-165 depending on approach (free for local-only, API costs for LLM/Cloud Vision)

### Critical First Steps

```bash
# 1. Debug layout detection
marker_single dictionary.pdf --debug --force_ocr --page_range "0-4" \
    --output_dir ./debug_output \
    --config_json arabic_tierA.json

# 2. Test with LLM correction
marker_single dictionary.pdf --use_llm --force_ocr --page_range "0-4" \
    --output_dir ./llm_output \
    --block_correction_prompt "Fully diacritized Arabic dictionary. Preserve all tashkeel." \
    --config_json arabic_tierB.json

# 3. Compare results
diff ./debug_output ./llm_output
```

---

*Report generated February 2026. Based on marker-pdf v1.10.2 and surya-ocr v0.17.x codebase analysis.*
