from pathlib import Path

from pdf_utils import extract_section, extract_text_by_page, get_original_page_offset


def test_get_original_page_offset_defaults_to_1_for_untrimmed_pdf(make_pdf):
    pdf = make_pdf(["page one", "page two"])
    assert get_original_page_offset(pdf) == 1


def test_extract_section_slices_the_requested_page_range(make_pdf, tmp_path):
    full = make_pdf([f"PAGE_MARKER_{n}" for n in range(1, 31)], name="full.pdf")
    trimmed = tmp_path / "trimmed.pdf"

    extract_section(full, start_page=10, end_page=20, output_path=trimmed)

    pages = extract_text_by_page(trimmed)
    assert len(pages) == 11  # 10..20 inclusive
    assert "PAGE_MARKER_10" in pages[0]["text"]
    assert "PAGE_MARKER_20" in pages[-1]["text"]
    # nothing from outside the requested range should have leaked in
    assert "PAGE_MARKER_9" not in pages[0]["text"]
    assert "PAGE_MARKER_21" not in pages[-1]["text"]


def test_extract_section_stamps_recoverable_offset(make_pdf, tmp_path):
    full = make_pdf([f"PAGE_MARKER_{n}" for n in range(1, 31)], name="full.pdf")
    trimmed = tmp_path / "trimmed.pdf"

    extract_section(full, start_page=10, end_page=20, output_path=trimmed)

    assert get_original_page_offset(trimmed) == 10


def test_extract_section_rejects_out_of_range_pages(make_pdf, tmp_path):
    full = make_pdf(["only page"], name="full.pdf")
    trimmed = tmp_path / "trimmed.pdf"

    import pytest
    with pytest.raises(ValueError):
        extract_section(full, start_page=1, end_page=5, output_path=trimmed)
