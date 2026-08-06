# Notice: third-party material

This repository benchmarks PDF-to-Markdown conversion tools. Doing that honestly
means publishing the converted output, so any document used as a conversion
sample has to be one that permits redistribution.

## The sample document

All parsed output under `output/` comes from a single source document:

> Zhang, Alex L., Tim Kraska, and Omar Khattab. *Recursive Language Models*.
> arXiv:2512.24601v2. <https://arxiv.org/abs/2512.24601>

The authors distribute it under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
which permits redistribution and derivative works, including format conversions
such as the Markdown here, provided the work is attributed.

The converted files are derivatives produced by the tools named in each output
path. Any conversion error in them is an artifact of the tool being tested and
is not attributable to the authors of the paper. That is, after all, the point
of the benchmark.

## Documents deliberately not included

Earlier revisions of this repository published converted output from documents
whose licenses do not permit redistribution. Those files were removed on
2026-08-05. Benchmark findings that involved them are reported in the comparison
reports without reproducing the source text:

- An arXiv thesis under the arXiv `nonexclusive-distrib/1.0` license, which
  grants distribution rights to arXiv rather than to third parties.
- A *Computational Linguistics* article under CC BY-NC-ND 4.0. The NoDerivatives
  term covers format conversion, so a converted copy cannot be redistributed
  even though the original is openly readable.
- A journal article in Arabic carrying no license statement, presumed all
  rights reserved.
- A blog post reproduced from its author's website, all rights reserved.

Short quotations elsewhere in this repository, such as the individual words and
table cells cited in the comparison reports as evidence of specific conversion
errors, are quotations for analysis rather than redistribution of the works they
come from.

## This repository's own work

The setup guides, comparison reports, research reports, benchmark scripts, and
notebooks are original work, licensed under the MIT License. See `LICENSE`.
