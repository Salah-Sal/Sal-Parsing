# Sal-Parsing

Comparison of PDF-to-Markdown parsing tools for scientific research papers (arXiv-style).

## Tools Evaluated

| Tool | Tiers Tested | Guide |
|------|-------------|-------|
| [Marker](https://github.com/VikParuchuri/marker) | 1, 4 | [Guide](parsing%20tools/PARSING_GUIDE-%20Marker.md) |
| [MarkItDown](https://github.com/microsoft/markitdown) | 1, 2 | [Guide](parsing%20tools/PARSING_GUIDE%20-%20markitdown.md) |
| [Docling](https://github.com/DS4SD/docling) | 1–5 | [Guide](parsing%20tools/PARSING_GUIDE%20-%20Docling.md) |
| [SciPDF](https://github.com/titipata/scipdf_parser) | 3–5 | [Guide](parsing%20tools/PARSING_GUIDE%20-%20Scipdf.md) |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 1 | [Guide](parsing%20tools/PARSING_GUIDE%20-%20PaddleOCR.md) |
| [dots.ocr](https://dots.ocr.com) | — | [Guide](parsing%20tools/PARSING_GUIDE%20-%20dots.ocr.md) |
| [OCRFlux](https://github.com/ocrflux/ocrflux) | — | [Guide](parsing%20tools/PARSING_GUIDE%20-%20OCRFlux.md) |
| [DeepSeek-OCR](https://github.com/deepseek-ai) | — | [Guide](parsing%20tools/PARSING_GUIDE%20-%20DeepSeek-OCR.md) |
| [Dolphin](https://github.com/bytedance/Dolphin) | — | [Guide](parsing%20tools/PARSING_GUIDE%20-%20Dolphin.md) |

## Structure

```
parsing tools/          # Setup guides with tier configs and benchmarks
  comparison_reports/   # Side-by-side quality comparisons
output/                 # Parsed markdown/JSON samples per tool and tier
papers/                 # Source PDFs (gitignored)
```

## Comparison Reports

- [Marker Tier 1 vs Tier 4](parsing%20tools/comparison_reports/marker_tier1_vs_tier4.md)
- [MarkItDown vs Marker](parsing%20tools/comparison_reports/markitdown_tier1_vs_marker_tier1.md)
- [Docling vs Marker vs MarkItDown](parsing%20tools/comparison_reports/docling_tier1_vs_marker_tier1_vs_markitdown_tier1.md)
- [Marker Tier 1 vs Docling Tier 4](parsing%20tools/comparison_reports/marker_tier1_vs_docling_tier4.md)
- [SciPDF Tier 3 vs Marker](parsing%20tools/comparison_reports/scipdf_tier3_vs_marker_tier1.md)
- [SciPDF Tier 5 vs Marker](parsing%20tools/comparison_reports/scipdf_tier5_vs_marker_tier1.md)
