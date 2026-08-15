"""Manual pre-processing step, run once per company before ingest.py.

Look up the page range of the section(s) you want to keep in the report's
own table of contents (Letter to Shareholders through Outlook is usually one
contiguous block), then:

    .venv/bin/python pipeline/trim_pdf.py \\
        data/raw_full/BMW_Report_2025.pdf 20 100 \\
        data/raw/BMW_management_report_2025.pdf

Run it again with a different page range/output name for a second,
non-contiguous section of the same report (e.g. a Risk Report that sits
apart from the main Lagebericht block) -- ingest.py picks up every file
under data/raw/, so multiple trimmed files per company are fine.

The output file is what ingest.py should be pointed at; keep the full,
untrimmed source PDFs out of data/raw/ so they don't also get ingested.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pdf_utils import extract_section

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_pdf", type=Path, help="Full, untrimmed source report")
    parser.add_argument("start_page", type=int, help="First page to keep, 1-indexed, as printed in the report's own TOC")
    parser.add_argument("end_page", type=int, help="Last page to keep, inclusive")
    parser.add_argument("output_pdf", type=Path, help="Where to write the trimmed section PDF")
    args = parser.parse_args()

    extract_section(args.input_pdf, args.start_page, args.end_page, args.output_pdf)
