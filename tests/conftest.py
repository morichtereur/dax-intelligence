import sys
from pathlib import Path

# tests run from the repo root (pytest's default rootdir insertion covers
# the top-level package, but pipeline/*.py import each other flatly, e.g.
# "from pdf_utils import ...", the same way ingest.py is meant to be run
# as a script -- so pipeline/ itself needs to be on sys.path too)
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

import pytest
from reportlab.pdfgen import canvas


@pytest.fixture
def make_pdf(tmp_path):
    """Factory fixture: make_pdf(pages) -> Path to a PDF where pages is a
    list of strings, one per page, written as the page's extractable text.
    """
    def _make(pages: list[str], name: str = "test.pdf") -> Path:
        pdf_path = tmp_path / name
        c = canvas.Canvas(str(pdf_path))
        for text in pages:
            for i, line in enumerate(text.split("\n")):
                c.drawString(72, 780 - i * 15, line)
            c.showPage()
        c.save()
        return pdf_path
    return _make
