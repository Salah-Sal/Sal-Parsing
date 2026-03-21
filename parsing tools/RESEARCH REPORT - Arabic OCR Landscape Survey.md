# Arabic OCR Landscape Survey: Tools, Repos & Projects

**Date:** February 2026
**Scope:** Open-source Arabic OCR engines, diacritization models, post-OCR correction tools, and NLP infrastructure on GitHub and HuggingFace
**Purpose:** Identify the most promising tools for OCR of classical Arabic text with full tashkeel (diacritical marks) for the Al-Mu'jam al-Wasit dictionary project

---

## Executive Summary

The Arabic OCR landscape has undergone a significant transformation in 2024-2026, driven by Vision-Language Models (VLMs). The field has moved from traditional CNN/LSTM pipelines to fine-tuned multimodal transformers that treat OCR as a vision-language task. Key findings:

- **QARI-OCR** is the current state-of-the-art for diacritized Arabic text (WER 0.160, CER 0.061)
- **Arabic-Nougat** offers direct page-to-Markdown output, closely matching our marker-based workflow
- **CATT** is the best open-source Arabic diacritizer, outperforming GPT-4-turbo
- **Manazir-OCR** provides a ready-made framework for comparing 7+ Arabic OCR backends
- The ecosystem is highly active, with most major projects updated in 2025-2026

This report catalogs **33+ tools/repos/projects** across 5 tiers of relevance.

---

## TIER 1: High-Priority Arabic-Specialized OCR Engines & Models

### 1. QARI-OCR -- State-of-the-Art for Arabic with Diacritics

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/NAMAA-ORG/qari-ocr-paper-2025 |
| **HuggingFace** | `NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct` (latest) |
| **Stars** | 7 (paper repo) |
| **Architecture** | Fine-tuned Qwen2-VL-2B-Instruct vision-language model |
| **Paper** | arXiv:2506.02295 (2025) |
| **License** | Open weights |

**Performance on diacritized text:**
- WER: 0.160
- CER: 0.061
- BLEU: 0.737
- 71% lower WER than Tesseract
- 328% better BLEU than EasyOCR
- 94.1% overall character accuracy

**Key strengths:**
- Best-in-class tashkeel recognition (fathah, kasrah, dammah, sukun, shadda, tanwin)
- Trained on 12 diverse Arabic fonts (14px-100px)
- Iterative fine-tuning on specialized synthetic datasets
- Available in multiple versions (v0.1, v0.2.2.1, v0.3)

**Relevance: HIGHEST** -- Directly addresses our diacritized Arabic dictionary use case. The model was specifically designed and benchmarked for tashkeel-heavy Arabic text.

---

### 2. DIMI-Arabic-OCR -- Fine-tuned VLM for Arabic OCR

| Field | Detail |
|-------|--------|
| **HuggingFace (v1)** | `AhmedZaky1/DIMI-Arabic-OCR` |
| **HuggingFace (v2)** | `AhmedZaky1/DIMI-Arabic-OCR-V2` |
| **Architecture** | Fine-tuned Qwen2.5-VL-7B-Instruct with LoRA adapters |
| **Base** | `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` |

**Key strengths:**
- V2 delivers 30% reduction in WER on diacritics-heavy text
- 7B parameter model (larger than QARI's 2B -- potentially more accurate for complex cases)
- Enhanced diacritics handling with balanced training data representation
- Supports both diacritized and undiacritized MSA text

**Relevance: HIGH** -- Strong alternative to QARI, especially for complex diacritics. The larger model size may yield better accuracy on dense dictionary pages.

---

### 3. Arabic-Nougat -- Page-to-Markdown for Arabic Books

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/MohamedAliRashad/arabic-nougat |
| **Stars** | 50 |
| **HuggingFace** | `MohamedRashad/arabic-large-nougat`, `arabic-base-nougat`, `arabic-small-nougat` |
| **Architecture** | Fine-tuned Meta Nougat (Vision Transformer encoder-decoder) |
| **Paper** | arXiv:2411.17835 (2024) |

**Key strengths:**
- End-to-end page-to-Markdown conversion (directly comparable to marker's output!)
- Trained on `arabic-img2md` dataset: 13.7k Arabic book page/Markdown pairs
- Released 1.1B token Arabic corpus extracted from 8,500+ books
- Custom Aranizer-PBE-86k tokenizer optimized for Arabic
- Uses torch.bfloat16 + Flash Attention 2 for efficient inference
- `arabic-large-nougat` delivers lowest CER and highest Markdown Structure Accuracy

**Relevance: HIGH** -- The closest equivalent to our marker workflow but specialized for Arabic. Could serve as a direct replacement or complement.

---

### 4. Manazir-OCR -- Arabic-First Multi-Model OCR Framework

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/h9-tec/Manazir-OCR |
| **Stars** | 40 |
| **Updated** | February 2026 |
| **Language** | Python |

**Integrated backends:**
- QARI-OCR
- DIMI-Arabic-OCR
- OCR-RL2
- TrOCR
- Qwen2/VL
- PaddleOCR Arabic
- Surya
- Optional API backends (Gemini, Claude, etc.)

**Key strengths:**
- Pluggable architecture -- switch between OCR backends via configuration
- CLI and Streamlit UIs included
- Outputs HTML and Markdown
- Arabic-first design philosophy
- By the same author as `Awesome_Arabic_NLP` and `Arabic_NLP_resources`

**Relevance: HIGH** -- Could serve as an orchestration layer to quickly benchmark multiple backends on our dictionary pages without building custom integration for each.

---

### 5. Qalam -- Multimodal LLM for Arabic OCR & Handwriting Recognition

| Field | Detail |
|-------|--------|
| **Paper** | https://arxiv.org/abs/2407.13559 (ACL ArabicNLP 2024) |
| **Architecture** | SwinV2 encoder + RoBERTa decoder |
| **Training data** | 4.5M+ Arabic manuscript images + 60k synthetic pairs |

**Performance:**
- WER: 0.80% (handwriting recognition)
- WER: 1.18% (printed OCR)

**Key strengths:**
- Exceptional diacritics handling
- High-resolution input support (addresses common limitation in OCR systems)
- Specifically designed for Arabic script characteristics

**Caveat:** Code/model public availability unclear from search results. The paper was published at ACL 2024.

**Relevance: HIGH** -- Extremely impressive WER numbers, especially for printed text (1.18%). Worth investigating model availability.

---

### 6. KITAB-Bench -- Arabic OCR & Document Understanding Benchmark

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/mbzuai-oryx/KITAB-Bench |
| **Stars** | 61 |
| **Published** | ACL 2025 |
| **Topics** | arabic, benchmark, layout-detection, ocr, pdf-to-text, table-detection, vlms, vqa |

**Description:** Comprehensive multi-domain benchmark for Arabic OCR and document understanding. From MBZUAI (Mohamed Bin Zayed University of AI).

**Relevance: HIGH** -- Essential for benchmarking any solution we adopt. Use this to compare candidate tools on standardized Arabic test data.

---

## TIER 2: General OCR Engines with Arabic Support

### 7. PaddleOCR

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/PaddlePaddle/PaddleOCR |
| **Stars** | 70,580 |
| **Arabic** | Yes (100+ languages) |
| **Strengths** | Detection + recognition pipeline, PP-OCRv4, very well-maintained |
| **Relevance** | MEDIUM -- general-purpose, not Arabic-specialized |

### 8. EasyOCR

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/JaidedAI/EasyOCR |
| **Stars** | 28,939 |
| **Arabic** | Yes (80+ languages) |
| **Strengths** | Dead-simple API, CRAFT detector + CRNN recognizer |
| **Relevance** | MEDIUM -- easy to try, but QARI significantly outperforms it on Arabic |

### 9. docTR (Mindee)

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/mindee/doctr |
| **Stars** | 5,854 |
| **Arabic** | Yes (multilingual) |
| **Strengths** | PyTorch/TensorFlow, detection + recognition, well-documented API |
| **Relevance** | MEDIUM -- solid framework but not Arabic-specialized |

### 10. Surya OCR (Our Current Engine via marker)

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/datalab-to/surya |
| **Stars** | 19,252 |
| **Arabic** | Yes (90+ languages, `"ar": "Arabic"` in language list) |
| **Strengths** | Layout analysis, reading order, table recognition |
| **Weaknesses** | No diacritics-specific training, no language hinting in marker pipeline |
| **Relevance** | BASELINE -- this is what we're currently using |

### 11. DeepSeek-OCR / DeepSeek-OCR-2

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/deepseek-ai/DeepSeek-OCR |
| **Stars** | 22,445 (v1) / 2,204 (v2) |
| **Arabic** | Yes (100+ languages including Arabic script) |
| **v1** | Released Oct 2025, context optical compression |
| **v2** | Released Jan 2026, visual causal flow |
| **Strengths** | Extreme token efficiency, open weights, supported in vLLM |
| **Relevance** | MEDIUM -- worth testing, strong general performance |

### 12. dots.ocr (RedNote HI Lab)

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/rednote-hilab/dots.ocr |
| **Stars** | 7,429 |
| **Architecture** | 1.7B VLM |
| **Arabic** | Yes (100+ languages, +7.4pt on XDocParse benchmark) |
| **Released** | July 2025 |
| **Strengths** | Unified architecture, layout + text extraction, fully open source |
| **Relevance** | MEDIUM-HIGH -- very recent, strong multilingual benchmarks |

### 13. GOT-OCR 2.0 (General OCR Theory)

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/Ucas-HaoranWei/GOT-OCR2.0 |
| **Stars** | 8,081 |
| **Architecture** | Unified end-to-end OCR model |
| **Relevance** | MEDIUM -- worth benchmarking on Arabic |

### 14. Kraken OCR

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/mittagessen/kraken |
| **Stars** | 945 |
| **Strengths** | Optimized for historical & non-Latin scripts |
| **Pre-trained** | `04-01-2021_ArabPersGeneralized.mlmodel` for Arabic/Persian |
| **Used by** | KITAB project, eScriptorium platform |
| **Relevance** | MEDIUM -- strong for historical manuscripts, line-based engine |

### 15. Calamari OCR

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/Calamari-OCR/calamari |
| **Stars** | 1,185 |
| **Strengths** | Line-based ATR, OCRopy-based, supports fine-tuning |
| **Relevance** | LOW-MEDIUM -- limited pre-trained Arabic models |

### 16. Tesseract (with Arabic Fine-Tuning Projects)

Notable fine-tuning projects:
- `OmarSamirz/Fine-Tuning-an-Arabic-OCR-Model-using-Tesseract-5.0` (6 stars)
- `ClearCypher/enhancing-tesseract-arabic-text-recognition` (24 stars)
- `mohammadmansour200/Tesseract-Arabic-Model` (5 stars)
- `agorararmard/Synthetic-Training-Data-Generator-for-Tesseract-Arabic` (5 stars)

**Relevance: LOW** -- Known to underperform on diacritized Arabic. QARI reports 71% lower WER.

---

## TIER 3: Arabic Diacritization (Tashkeel) Tools

These tools add or correct diacritical marks and are critical for post-OCR correction pipelines.

### 17. CATT -- Character-based Arabic Tashkeel Transformer (SOTA Diacritizer)

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/abjadai/catt |
| **Stars** | 59 |
| **Paper** | arXiv:2407.03236 (2024) |
| **Architecture** | Fine-tuned character-based BERT + Noisy-Student training |
| **Organization** | Abjad AI |

**Performance:**
- Outperforms GPT-4-turbo by 9.36% relative DER on CATT dataset
- 30.83% and 35.21% better relative DER than all 11 evaluated models (WikiNews, CATT benchmarks)
- Evaluated against both commercial and open-source diacritizers

**Extensions:** CATT-Whisper (multimodal diacritic restoration, 2025)

**Relevance: HIGH** -- Best open-source diacritizer. Essential component of any post-OCR correction pipeline for our dictionary project.

### 18. Sadeed -- Small Language Model for Diacritization

| Field | Detail |
|-------|--------|
| **Paper** | https://arxiv.org/abs/2504.21635 (2025) |
| **Architecture** | Fine-tuned Kuwain 1.5B (decoder-only) |
| **Training data** | Tashkeela corpus (75M words, classical Arabic) + Arabic Treebank (ATB-3) |
| **Benchmark** | Introduces SadeedDiac-25 (Classical + MSA Arabic) |
| **From** | Misraj AI |

**Key strengths:**
- Specifically targets classical Arabic (our use case!)
- Lightweight (1.5B parameters)
- New benchmark designed for fair evaluation across text genres
- Competitive with proprietary LLMs despite modest resources

**Relevance: HIGH** -- Classical Arabic focus aligns perfectly with Al-Mu'jam al-Wasit.

### 19. Shakkala -- Deep Learning Arabic Vocalization

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/Barqawiz/Shakkala |
| **Stars** | 347 |
| **ONNX ports** | `nipponjo/arabic-vocalization` (14 stars), `nipponjo/arabic_vocalizer` (11 stars) |
| **Relevance** | MEDIUM -- established tool, ONNX deployment available |

### 20. Mishkal -- Rule-Based Arabic Text Vocalization

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/linuxscout/mishkal |
| **Stars** | 302 |
| **Approach** | Rule-based morphological analysis |
| **Relevance** | MEDIUM -- useful as a validation baseline vs. neural approaches |

### 21. libtashkeel -- Multi-Platform Diacritization Library

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/mush42/libtashkeel |
| **Stars** | 45 |
| **Bindings** | Rust, Python, C++, WASM |
| **Relevance** | MEDIUM -- useful for cross-platform integration |

### 22. Meshakkelaty.ai -- Kaggle 1st Place Diacritizer

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/Omar-Al-Sharif/Meshakkelaty.ai |
| **Stars** | 18 |
| **Approach** | Neural + statistical engine |
| **Relevance** | MEDIUM |

### 23. Arabic_Diacritization -- PyTorch Multi-Architecture

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/almodhfer/Arabic_Diacritization |
| **Stars** | 36 |
| **Approach** | Multiple deep learning architectures in PyTorch |
| **Relevance** | MEDIUM -- good reference implementation |

---

## TIER 4: Post-OCR Correction Tools

### 24. CAMeL Lab arafix_ocr -- N-gram Post-OCR Correction for Arabic

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/CAMeL-Lab/arafix_ocr |
| **Stars** | 9 |
| **From** | NYU Abu Dhabi CAMeL Lab |
| **Method** | N-gram based post-correction using SRILM toolkit |
| **Approach** | Text-only (no image knowledge needed) |

**Description:** Improves output of generic Arabic OCR systems by passing encoded text to the SRILM disambig function, which determines token corrections based on preceding context (n-gram language model).

**Relevance: HIGH** -- The only dedicated Arabic post-OCR correction tool. Directly applicable to our pipeline.

### 25. analiticcl -- Fuzzy Matching for OCR Correction

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/proycon/analiticcl |
| **Stars** | 37 |
| **Language** | Rust |
| **Approach** | Approximate string matching, spelling correction, normalization |
| **Relevance** | MEDIUM -- language-agnostic, could work for Arabic with custom lexicon |

### 26. LLM-Based Post-OCR Correction Approaches

| Repo | Stars | Approach |
|------|-------|----------|
| [jarobyte91/post_ocr_correction](https://github.com/jarobyte91/post_ocr_correction) | 39 | Char seq2seq ensembles |
| [Shef-AIRE/llms_post-ocr_correction](https://github.com/Shef-AIRE/llms_post-ocr_correction) | 15 | LLMs for historical newspapers |
| [jzhang512/post-ocr-correction](https://github.com/jzhang512/post-ocr-correction) | 3 | GPT-based prompting |

**Relevance: MEDIUM** -- LLM approaches are promising for our use case, especially combined with marker's existing LLM correction processor.

---

## TIER 5: Arabic NLP Infrastructure & Datasets

### 27. CAMeL Tools -- Comprehensive Arabic NLP Suite

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/CAMeL-Lab/camel_tools |
| **Stars** | 522 |
| **From** | NYU Abu Dhabi CAMeL Lab |
| **Features** | Morphological analysis, disambiguation, diacritization, NER, tokenization, dialect ID |

**Relevance: HIGH** -- Essential NLP infrastructure for validation and post-processing in any Arabic OCR pipeline.

### 28. PyArabic -- Arabic Text Processing Library

| Field | Detail |
|-------|--------|
| **GitHub** | https://github.com/linuxscout/pyarabic |
| **Stars** | 476 |
| **Features** | Arabic text utilities, character manipulation, tashkeel handling, normalization |

**Relevance: HIGH** -- Core utility library for Arabic text processing.

### 29. linuxscout's Arabic NLP Ecosystem

This developer (Taha Zerrouki) maintains the most comprehensive open-source Arabic NLP ecosystem:

| Repo | Stars | Description |
|------|-------|-------------|
| [pyarabic](https://github.com/linuxscout/pyarabic) | 476 | Arabic text processing library |
| [mishkal](https://github.com/linuxscout/mishkal) | 302 | Arabic text vocalization |
| [arramooz](https://github.com/linuxscout/arramooz) | 147 | Arabic morphological dictionary |
| [tashaphyne](https://github.com/linuxscout/tashaphyne) | 103 | Arabic light stemmer |
| [qutrub](https://github.com/linuxscout/qutrub) | 98 | Arabic verb conjugator |
| [arabicnlptoolslist](https://github.com/linuxscout/arabicnlptoolslist) | 90 | Inventory of Arabic NLP tools |
| [adawat](https://github.com/linuxscout/adawat) | 53 | Arabic text tools |
| [yaraspell](https://github.com/linuxscout/yaraspell) | 46 | Arabic spell checker |
| [ayaspell](https://github.com/linuxscout/ayaspell) | 45 | Arabic Hunspell dictionary |
| [qalsadi](https://github.com/linuxscout/qalsadi) | 42 | Arabic morphological analyzer |
| [fareh](https://github.com/linuxscout/fareh) | 41 | Arabic grammar rules database |
| [yarob](https://github.com/linuxscout/yarob) | 40 | Arabic inflection |
| [ghalatawi](https://github.com/linuxscout/ghalatawi) | 30 | Arabic autocorrect |
| [tashkeela2](https://github.com/linuxscout/tashkeela2) | 14 | Arabic vocalized text corpus |
| [aghlat](https://github.com/linuxscout/aghlat) | 8 | Arabic misspelling corpus |

### 30. Arabic NLP Resource Collections & Awesome Lists

| Repo | Stars | Description |
|------|-------|-------------|
| [ARBML/ARBML](https://github.com/ARBML/ARBML) | 422 | Arabic NLP and CV projects with implementations |
| [bakrianoo/aravec](https://github.com/bakrianoo/aravec) | 416 | Pre-trained Arabic word embeddings |
| [h9-tect/Arabic_NLP_resources](https://github.com/h9-tect/Arabic_NLP_resources) | 222 | Arabic NLP resources catalog |
| [ARBML/masader](https://github.com/ARBML/masader) | 193 | Catalog of 500+ Arabic datasets |
| [h9-tect/Awesome_Arabic_NLP](https://github.com/h9-tec/Awesome_Arabic_NLP) | 136 | Curated awesome list for Arabic NLP |
| [Curated-Awesome-Lists/awesome-arabic-nlp](https://github.com/Curated-Awesome-Lists/awesome-arabic-nlp) | 52 | Community-curated Arabic NLP resources |
| [NNLP-IL/Arabic-Resources](https://github.com/NNLP-IL/Arabic-Resources) | 43 | Comprehensive Arabic NLP resource list |

### 31. Arabic OCR Datasets (HuggingFace)

| Dataset | Description |
|---------|-------------|
| `mssqpi/Arabic-OCR-Dataset` | 2M+ labeled Arabic text images for OCR training |
| `riotu-lab/SARD` | 743k synthetic Arabic docs, 662M words, 5 fonts |
| `Nourhann/Arabic-Diacritized-TTS` | Diacritized Arabic text for TTS training |
| `linuxscout/tashkeela2` | Arabic vocalized text corpus |
| `Misraj/Sadeed_Tashkeela` | Dataset for Sadeed diacritization model |

### 32. Arabic Morphological Analyzers

| Repo | Stars | Description |
|------|-------|-------------|
| [Qutuf/Qutuf](https://github.com/Qutuf/Qutuf) | 132 | Morphological analyzer & POS tagger (Expert System) |
| [alsaydi/sarf](https://github.com/alsaydi/sarf) | 46 | Arabic morphology system |
| [otakar-smrz/elixir-fm](https://github.com/otakar-smrz/elixir-fm) | 45 | Functional Arabic morphology (Haskell) |
| [qalsadi](https://github.com/linuxscout/qalsadi) | 42 | Arabic morphological analyzer (Python) |

### 33. Other Notable Arabic OCR Projects

| Repo | Stars | Description |
|------|-------|-------------|
| [HusseinYoussef/Arabic-OCR](https://github.com/HusseinYoussef/Arabic-OCR) | 317 | CNN-based Arabic OCR for typed text |
| [Pythonation/Mistral-Arabic-OCR-test](https://github.com/Pythonation/Mistral-Arabic-OCR-test) | 107 | Mistral AI for Arabic PDF/image OCR |
| [maidaly/Arabic_OCR](https://github.com/maidaly/Arabic_OCR) | 62 | Arabic OCR application |
| [AHR-OCR2024/Arabic-Handwriting-Recognition](https://github.com/AHR-OCR2024/Arabic-Handwriting-Recognition) | 38 | E2E Arabic handwriting OCR with exam grading |
| [Hedrax/Invizo-OCR](https://github.com/Hedrax/Invizo-OCR) | 9 | Arabic OCR for template-based documents |
| [yayaiu6/Text-Vision-Advanced-Arabic-OCR-Model-Using-TrOCR](https://github.com/yayaiu6/Text-Vision-Advanced-Arabic-OCR-Model-Using-TrOCR) | 6 | TrOCR fine-tuned for Arabic |
| [OussamaBenSlama/Alef-OCR-Image2Html](https://github.com/OussamaBenSlama/Alef-OCR-Image2Html) | 4 | Arabic documents to semantic HTML |
| [mustakshif/arabic-ocr-benchmark-tool](https://github.com/mustakshif/arabic-ocr-benchmark-tool) | 0 | Gemini vs Mistral OCR comparison tool |

---

## Recommended Evaluation Strategy

### Phase 1: Quick Benchmarks
Test these 4 candidates on 10 sample dictionary pages with ground truth:
1. **QARI-OCR v0.3** -- Most likely best for diacritized text
2. **DIMI-Arabic-OCR V2** -- Larger 7B model, potentially more accurate
3. **Arabic-Nougat (large)** -- Direct page-to-Markdown output
4. **dots.ocr** -- Very recent, strong multilingual performance

### Phase 2: Pipeline Assembly
Build a correction pipeline combining:
1. Best OCR engine (from Phase 1) for raw text extraction
2. **CATT** or **Sadeed** for diacritization correction/validation
3. **CAMeL arafix_ocr** for n-gram based post-correction
4. **pyarabic** + **camel_tools** for text normalization & validation

### Phase 3: Integration
Three possible integration paths:
- **A)** Replace surya with the best-performing engine in our marker workflow
- **B)** Use **Manazir-OCR** as an orchestration layer to quickly compare engines
- **C)** Build a custom pipeline: OCR engine -> diacritizer -> post-corrector -> markdown formatter

---

## Key Takeaways

1. **QARI-OCR is the current SOTA** for diacritized Arabic OCR (WER 0.160, CER 0.061)
2. **Arabic-Nougat** is uniquely suited for page-to-Markdown conversion (matches our use case)
3. **CATT** is the best open-source diacritizer, outperforming GPT-4-turbo
4. **Manazir-OCR** provides a ready-made framework for comparing 7+ Arabic OCR backends
5. **CAMeL arafix_ocr** is the only dedicated Arabic post-OCR correction tool
6. **Sadeed** specifically targets classical Arabic diacritization (our domain)
7. The field is highly active (most repos updated 2025-2026) with VLM-based approaches dominating
8. **linuxscout** (Taha Zerrouki) maintains the most comprehensive Arabic NLP open-source ecosystem
9. **KITAB-Bench** (ACL 2025) provides standardized benchmarking for Arabic OCR
10. Key HuggingFace datasets (`SARD`, `Arabic-OCR-Dataset`) offer millions of training samples for fine-tuning
