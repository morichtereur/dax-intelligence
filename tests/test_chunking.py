from chunking import chunk_pages, detect_section


def test_detect_section_matches_known_markers():
    assert detect_section("See our Risk Report for details.") == "risk_report"
    assert detect_section("Nothing relevant here.") == "general"


def test_detect_section_picks_first_marker_in_priority_order():
    # documents current behavior (first match in SECTION_MARKERS order wins)
    # so a future change to the priority list is a deliberate edit, not a
    # silent regression
    text = "This mentions both management report and outlook topics."
    assert detect_section(text) == "management_report"


def test_chunk_pages_tracks_real_page_ranges_not_linear_estimates():
    # deliberately uneven word count per page: page 2 has almost no text,
    # which is exactly what breaks a chunk-index/total-chunks estimate
    pages = [
        {"page": 1, "text": " ".join(f"w{i}" for i in range(60))},
        {"page": 2, "text": "sparse"},
        {"page": 3, "text": " ".join(f"w{i}" for i in range(60))},
    ]

    chunks = chunk_pages(pages, chunk_size=20, overlap=5)

    # every chunk's page range must fall within the real 1-3 range
    assert all(1 <= c["start_page"] <= c["end_page"] <= 3 for c in chunks)

    # the sparse page-2 word should show up attributed to page 2 somewhere,
    # not smeared across a wide estimated range
    sparse_chunks = [c for c in chunks if "sparse" in c["text"]]
    assert sparse_chunks
    for c in sparse_chunks:
        assert c["start_page"] <= 2 <= c["end_page"]


def test_chunk_pages_covers_all_words_with_overlap():
    pages = [{"page": 1, "text": " ".join(f"w{i}" for i in range(100))}]
    chunks = chunk_pages(pages, chunk_size=30, overlap=10)

    # every word should appear in at least one chunk
    all_words = {f"w{i}" for i in range(100)}
    covered = set()
    for c in chunks:
        covered.update(c["text"].split())
    assert all_words <= covered
