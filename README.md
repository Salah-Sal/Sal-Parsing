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
| [MinerU2.5](https://github.com/opendatalab/MinerU) | benchmarked (M2 Pro) | [Guide](parsing%20tools/PARSING_GUIDE%20-%20MinerU.md) |
| [Chandra](https://github.com/datalab-to/chandra) | benchmarked (M2 Pro) | [Guide](parsing%20tools/PARSING_GUIDE%20-%20Chandra.md) |
| [QARI-OCR](https://github.com/NAMAA-ORG/qari-ocr-paper-2025) | benchmarked (Arabic) | [Survey](parsing%20tools/RESEARCH%20REPORT%20-%20Arabic%20OCR%20Landscape%20Survey.md) |

## Structure

```
parsing tools/          # Setup guides with tier configs and benchmarks
  comparison_reports/   # Side-by-side quality comparisons
benchmark/              # Tier configs and benchmark scripts
output/                 # Parsed markdown/JSON per tool and tier
papers/                 # Source PDFs (gitignored)
```

## Benchmark document

Every parse under `output/` is of the same source document, so the tools and
tiers are directly comparable:

> Zhang, Alex L., Tim Kraska, and Omar Khattab. *Recursive Language Models*.
> arXiv:[2512.24601v2](https://arxiv.org/abs/2512.24601). Licensed
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

It was chosen for having the failure modes that separate these tools: a
multi-column layout, inline math, footnotes, and a multi-section comparison
table with grouped column headers.

## Comparison Reports

- [Marker Tier 1 vs Tier 4](parsing%20tools/comparison_reports/marker_tier1_vs_tier4.md)
- [MarkItDown vs Marker](parsing%20tools/comparison_reports/markitdown_tier1_vs_marker_tier1.md)
- [Docling vs Marker vs MarkItDown](parsing%20tools/comparison_reports/docling_tier1_vs_marker_tier1_vs_markitdown_tier1.md)
- [Marker Tier 1 vs Docling Tier 4](parsing%20tools/comparison_reports/marker_tier1_vs_docling_tier4.md)
- [SciPDF Tier 3 vs Marker](parsing%20tools/comparison_reports/scipdf_tier3_vs_marker_tier1.md)
- [SciPDF Tier 5 vs Marker](parsing%20tools/comparison_reports/scipdf_tier5_vs_marker_tier1.md)
- [New Parsers Benchmark — MinerU2.5 vs Marker vs QARI on Apple M2 Pro / 16 GB](parsing%20tools/comparison_reports/new_parsers_benchmark_M2Pro.md)

## License and attribution

Original work in this repository (guides, comparison reports, research reports,
benchmark scripts, notebooks) is MIT licensed: see [`LICENSE`](LICENSE).
Third-party material and the reasoning behind which documents are and are not
republished here are documented in [`NOTICE.md`](NOTICE.md).
