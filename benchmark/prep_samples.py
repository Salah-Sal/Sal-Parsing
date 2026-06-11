"""Prepare shared evaluation samples for the new-tool benchmark.

Picks the most *representative* pages instead of guessing:
  - science PDF  -> pages richest in math indicators (=, ^, _, \\, sum/int glyphs)
  - arabic PDF   -> pages richest in Arabic-script codepoints (U+0600..U+06FF)

For each chosen page it writes a 1-page PDF and a 300-DPI PNG (PNG is what the
VLM OCR tools consume). Outputs land in samples/<slug>/page_<n>.{pdf,png}.
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path
import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
DPI = 300

ARABIC = re.compile(r"[؀-ۿ]")
MATH = re.compile(r"[=^_\\∑∫√≤≥≈→⊗⟨⟩∈±×·]|\\(frac|sum|int|alpha|beta|theta|rho)")

def score(text: str, kind: str) -> int:
    if kind == "arabic":
        return len(ARABIC.findall(text))
    return len(MATH.findall(text))

def pick(pdf_path: Path, kind: str, n: int) -> list[int]:
    doc = fitz.open(pdf_path)
    scores = []
    for i, page in enumerate(doc):
        scores.append((score(page.get_text(), kind), i))
    doc.close()
    scores.sort(reverse=True)
    top = sorted(i for _, i in scores[:n])
    print(f"[{pdf_path.name}] {kind}: top pages (0-idx) = {top} "
          f"(scores {[s for s,_ in scores[:n]]})")
    return top

def export(pdf_path: Path, pages: list[int], slug: str):
    out = SAMPLES / slug
    out.mkdir(parents=True, exist_ok=True)
    src = fitz.open(pdf_path)
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    for p in pages:
        # 1-page PDF
        one = fitz.open()
        one.insert_pdf(src, from_page=p, to_page=p)
        one.save(out / f"page_{p+1:03d}.pdf")
        one.close()
        # PNG render
        pix = src[p].get_pixmap(matrix=mat)
        pix.save(out / f"page_{p+1:03d}.png")
        print(f"  wrote {slug}/page_{p+1:03d}.{{pdf,png}}  ({pix.width}x{pix.height})")
    src.close()

if __name__ == "__main__":
    # Point ARABIC_PDF at any Arabic-script PDF to generate the Arabic samples.
    arabic_src = os.environ.get("ARABIC_PDF", str(ROOT / "samples" / "arabic_source.pdf"))
    jobs = [
        (ROOT / "samples/2512.24601v2.pdf", "math", "science", 3),
        (Path(arabic_src), "arabic", "arabic", 2),
    ]
    for pdf, kind, slug, n in jobs:
        if not pdf.exists():
            print(f"SKIP missing {pdf}", file=sys.stderr); continue
        pages = pick(pdf, kind, n)
        export(pdf, pages, slug)
    print("done.")
