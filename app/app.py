import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from app.retriever import retrieve, get_companies, get_years, collection, MIN_RELEVANCE, LOW_CONFIDENCE
from app.llm import ask, verify_citations, highlight_citations, page_label, estimate_cost
from app.branding import COMPANIES, company_meta, logo_data_uri, author_photo_uri, report_pdf_url
from app.export import build_memo_pdf

st.set_page_config(
    page_title="DAX Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system: a research-desk "ledger" — greenbar accounting-paper tones,
# a stamped letterhead, and an audit trail that shows its work rather than
# just asserting an answer. IBM Plex Serif carries the letterhead voice,
# Plex Sans the prose, Plex Mono every number and citation (tabular figures
# throughout, the way a statement lines up its amounts).
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #171B16;
    --paper: #EFEFE4;
    --paper-dim: #E5E5D6;
    --paper-card: #F7F7EF;
    --rule: #C7CBB6;
    --green: #1E6B45;
    --green-dim: #DCE8DC;
    --red: #A0392D;
    --red-dim: #F1DDD7;
    --amber: #92661A;
    --amber-dim: #F2E6CC;
    --navy: #15222B;
    --navy-2: #1E313D;
    --paper-on-navy: #E7E9E1;
}

html, body, [data-testid="stAppViewContainer"], .main {
    background: var(--paper) !important;
    color: var(--ink);
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.4rem; max-width: 980px; }

h1, h2, h3 { font-family: 'IBM Plex Serif', serif; color: var(--ink); letter-spacing: -0.01em; }

/* ---------- Paper grain ---------- */
body::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 999;
    opacity: 0.05;
    mix-blend-mode: multiply;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ---------- Letterhead ---------- */
.letterhead {
    position: relative;
    background: var(--navy);
    color: var(--paper-on-navy);
    margin: -0.4rem -0.2rem 0 -0.2rem;
    padding: 1.9rem 2.1rem 1.6rem 2.1rem;
    border-radius: 3px;
}
.letterhead-mark {
    font-family: 'IBM Plex Serif', serif;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: -0.01em;
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
}
.letterhead-sub {
    margin-top: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9AA79C;
}

/* ---------- Index roll ---------- */
.roll-wrap {
    background: var(--paper-card);
    border: 1px solid var(--rule);
    border-top: none;
    padding: 1rem 1.2rem 0.8rem 1.2rem;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.9rem 0.4rem;
}
.roll-tile {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    gap: 0.35rem;
    height: 44px;
    padding-top: 0.2rem;
    border-top: 2px solid transparent;
    transition: border-color 0.25s ease;
}
.roll-tile img {
    max-height: 24px;
    max-width: 84px;
    object-fit: contain;
    filter: grayscale(1) opacity(0.42);
    transition: filter 0.25s ease;
}
.roll-tile .tick {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.04em;
    color: #9a9d8d;
    transition: color 0.25s ease;
}
.roll-tile.considered img { filter: grayscale(0.55) opacity(0.75); }
.roll-tile.considered .tick { color: #6b7060; }
.roll-tile.cited { border-top-color: var(--green); }
.roll-tile.cited img { filter: grayscale(0) opacity(1); }
.roll-tile.cited .tick { color: var(--green); font-weight: 600; }

/* ---------- Ledger stat lines (dot leaders, like a statement footer) ---------- */
.ledger-stats {
    background: var(--paper-card);
    border: 1px solid var(--rule);
    border-top: none;
    padding: 0.95rem 1.4rem 1.05rem 1.4rem;
}
.ledger-line {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    padding: 0.3rem 0;
}
.ledger-line .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #545843;
    white-space: nowrap;
}
.ledger-line .fill {
    flex: 1;
    border-bottom: 1px dotted var(--rule);
    margin-bottom: 0.3rem;
}
.ledger-line .value {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1rem;
    color: var(--ink);
    white-space: nowrap;
}

/* ---------- Byline ---------- */
.byline {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 2.5rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--rule);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #6b7060;
}
.byline img {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    object-fit: cover;
    object-position: 50% 20%;
    filter: grayscale(1) contrast(1.05);
    border: 1px solid var(--rule);
    flex-shrink: 0;
}

hr, div[data-testid="stDivider"] { border-color: var(--rule) !important; }

/* ---------- Query field ---------- */
[data-testid="stTextInput"] input {
    background: var(--paper-card) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 2px !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Serif', serif !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 0.9rem !important;
}
[data-testid="stTextInput"] input:focus {
    box-shadow: 0 0 0 1px var(--green) !important;
    border-color: var(--green) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #8a8d7c; font-style: italic; }

/* ---------- Buttons (example chips) ---------- */
.stButton button {
    background: var(--paper-card) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 2px !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    text-align: left !important;
    padding: 0.5rem 0.7rem !important;
}
.stButton button:hover { border-color: var(--green) !important; color: var(--green) !important; }

[data-testid="stDownloadButton"] button {
    background: var(--paper-card) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 2px !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 0.45rem 0.9rem !important;
}
[data-testid="stDownloadButton"] button:hover { border-color: var(--green) !important; color: var(--green) !important; background: var(--green-dim) !important; }

/* ---------- Expander (example queries + audit rows) ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--rule) !important;
    border-radius: 2px !important;
    background: var(--paper-card) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ---------- Sidebar control rail ---------- */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid #0d1519;
}
[data-testid="stSidebar"] * { color: var(--paper-on-navy) !important; }
[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9AA79C !important;
    border-bottom: 1px solid var(--navy-2);
    padding-bottom: 0.5rem;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: var(--navy-2) !important;
    border-color: #33474f !important;
}
[data-testid="stSidebar"] label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; }
[data-testid="stSidebar"] [role="slider"] { background: var(--green) !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[style*="background-color: rgb(255"] { background: var(--green) !important; }

/* ---------- Answer memo ---------- */
.memo-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 2px solid var(--ink);
    padding-bottom: 0.35rem;
    margin: 1.6rem 0 0.9rem 0;
}
.memo-head .title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #545843;
}
.memo-body { font-size: 1rem; line-height: 1.65; }
.memo-body p { margin-bottom: 0.9rem; }
.memo-body h4 {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    margin: 1.3rem 0 0.5rem 0;
    color: var(--ink);
}
.cite-ok {
    color: var(--green);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88em;
    background: var(--green-dim);
    padding: 0.03rem 0.28rem;
    border-radius: 2px;
}
.cite-flag {
    color: var(--red);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88em;
    background: var(--red-dim);
    padding: 0.03rem 0.28rem;
    border-radius: 2px;
}

.stamp {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    transform: rotate(-1deg);
}
.stamp-green { background: var(--green-dim); color: var(--green); border: 1px solid var(--green); }
.stamp-red { background: var(--red-dim); color: var(--red); border: 1px solid var(--red); }
.stamp-amber { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber); }

.ledger-banner {
    border: 1px solid var(--amber);
    background: var(--amber-dim);
    padding: 0.6rem 0.8rem;
    font-size: 0.88rem;
    margin-bottom: 0.6rem;
    display: flex;
    gap: 0.6rem;
    align-items: baseline;
}
.ledger-empty {
    border: 1px dashed var(--rule);
    padding: 1.4rem;
    text-align: left;
}
.ledger-empty p { margin-top: 0.6rem; color: #4a4d3e; }

.verify-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: #545843;
    margin-top: 0.4rem;
}

/* ---------- Audit trail ledger ---------- */
.audit-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #545843;
    margin: 1.6rem 0 0.5rem 0;
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.35rem;
}
.audit-row {
    display: grid;
    grid-template-columns: 22px 1fr 106px 84px 58px 74px;
    align-items: center;
    gap: 0.6rem;
    padding: 0.45rem 0.2rem;
    border-bottom: 1px solid var(--rule);
    font-size: 0.84rem;
}
.audit-row img { height: 14px; max-width: 20px; object-fit: contain; filter: grayscale(1) opacity(0.7); }
.audit-row .co { font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.audit-row .sec { color: #6b7060; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.audit-row .pg { font-family: 'IBM Plex Mono', monospace; color: #545843; white-space: nowrap; }
.audit-row .pg a { color: #545843; text-decoration: none; border-bottom: 1px dotted #545843; }
.audit-row .pg a:hover { color: var(--green); border-bottom-color: var(--green); }
.audit-row .sc { font-family: 'IBM Plex Mono', monospace; text-align: right; color: var(--ink); }
.audit-row .sc.kw { color: #7a8f6b; font-style: italic; font-size: 0.72rem; }
.via-both { color: var(--green); font-weight: 700; }
.audit-row .flag-ok { color: var(--green); font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; text-align: right; white-space: nowrap; }
.audit-row .flag-dim { color: #a3a68f; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; text-align: right; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def _companies() -> list[str]:
    return get_companies()


@st.cache_data
def _chunk_count() -> int:
    return collection.count()


@st.cache_data
def _years() -> list[str]:
    return get_years()


present_companies = set(_companies())
present_years = sorted(_years())
latest_year = present_years[-1] if present_years else "2025"
year_label = f"FY{present_years[0]}–FY{present_years[-1]}" if len(present_years) > 1 else f"FY{latest_year}"
year_phrase = f"fiscal years {present_years[0]}–{present_years[-1]}" if len(present_years) > 1 else f"fiscal year {latest_year}"

# ---------------------------------------------------------------------------
# Letterhead
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="letterhead">
    <div class="letterhead-mark">DAX Intelligence</div>
    <div class="letterhead-sub">{len(present_companies)} of 40 constituents · {year_phrase} · retrieval-grounded research memo</div>
</div>
""", unsafe_allow_html=True)

roll_slot = st.empty()

# ---------------------------------------------------------------------------
# Control rail
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Scope")
    companies = _companies()
    company_filter = st.selectbox(
        "Company",
        options=["All companies"] + companies,
        help="Restrict retrieval to a single constituent, or query the full index",
        label_visibility="collapsed",
    )
    compare_mode = st.toggle("Compare across companies", value=False)

    year_filter = latest_year
    if len(present_years) > 1:
        year_options = [f"Latest only (FY{latest_year})"] + [f"FY{y} only" for y in present_years[:-1]] + ["All years (year-over-year)"]
        year_choice = st.selectbox(
            "Reporting period",
            options=year_options,
            help="Restrict retrieval to one fiscal year, or open both so the memo can speak to what changed",
            label_visibility="collapsed",
        )
        if year_choice.startswith("Latest"):
            year_filter = latest_year
        elif year_choice.startswith("All years"):
            year_filter = None
        else:
            year_filter = year_choice.split()[0].removeprefix("FY")

    n_results = st.slider("Excerpts retrieved", min_value=4, max_value=16, value=8)

    st.markdown("### Index composition")
    for c in COMPANIES:
        if c["key"] in present_companies:
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;'
                f'display:flex;justify-content:space-between;padding:0.2rem 0;color:#c7cbb9;">'
                f'<span>{c["name"]}</span><span style="color:#6f8f7c;">{c["ticker"]}</span></div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Stat strip — ledger-style dot leaders, like the footer of a statement
# ---------------------------------------------------------------------------
total_chunks = _chunk_count()
st.markdown(f"""
<div class="ledger-stats">
    <div class="ledger-line"><span class="label">Constituents on file</span><span class="fill"></span><span class="value">{len(present_companies)}</span></div>
    <div class="ledger-line"><span class="label">Indexed excerpts</span><span class="fill"></span><span class="value">{total_chunks:,}</span></div>
    <div class="ledger-line"><span class="label">Reporting year</span><span class="fill"></span><span class="value">{year_label}</span></div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
query = st.text_input(
    "Ask the record",
    placeholder="Ask the record — e.g. how do Siemens and SAP frame their AI investment strategy?",
    label_visibility="collapsed",
)

examples = [
    "Which companies disclosed structural cost reduction programs?",
    "How is VW addressing the EV transition financially?",
    "What do CFOs say about macroeconomic risks in 2025?",
    "Compare how BMW and Mercedes frame EV investment costs",
    "Which companies mentioned GBS or shared services transformation?",
    "What are the key strategic priorities across DAX industrials?",
]

with st.expander("Sample questions on file"):
    cols = st.columns(2)
    for i, eq in enumerate(examples):
        with cols[i % 2]:
            if st.button(eq, key=eq, use_container_width=True):
                query = eq

# ---------------------------------------------------------------------------
# Retrieval + synthesis + verification
# ---------------------------------------------------------------------------
considered_companies: set[str] = set()
cited_companies: set[str] = set()

if query:
    selected_company = None if company_filter == "All companies" else company_filter

    with st.spinner("Searching the record…"):
        chunks = retrieve(query, n_results=n_results, company_filter=selected_company, year_filter=year_filter)

    considered_companies = {c["company"] for c in chunks}

    if not chunks:
        st.markdown(f"""
        <div class="ledger-empty">
            <span class="stamp stamp-red">NOT ON FILE</span>
            <p>No excerpt across the {len(present_companies)} loaded reports clears the {MIN_RELEVANCE:.2f}
            relevance bar for this question — nothing was sent to the model. Rephrase it, widen the company
            scope, or confirm the topic is actually covered in FY2025 annual reporting.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        scored = [c["score"] for c in chunks if c["score"] is not None]
        best_score = max(scored) if scored else MIN_RELEVANCE
        with st.spinner("Drafting memo…"):
            result = ask(query, chunks, compare_mode=compare_mode)
        answer, usage = result["text"], result["usage"]

        citations = verify_citations(answer, chunks)
        cited_companies = {c["company"] for c in citations if c["verified"]}
        unverified = [c for c in citations if not c["verified"]]

        st.markdown('<div class="memo-head"><span class="title">Analyst note</span></div>', unsafe_allow_html=True)

        if best_score < LOW_CONFIDENCE:
            st.markdown(f"""
            <div class="ledger-banner">
                <span class="stamp stamp-amber">WEAK MATCH</span>
                <span>Best retrieved excerpt scored {best_score:.2f} similarity, below the {LOW_CONFIDENCE:.2f}
                confidence bar. This memo may rest on tangential evidence — check the audit trail below
                before relying on it.</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f'<div class="memo-body">{highlight_citations(answer, chunks)}</div>', unsafe_allow_html=True)

        if citations:
            verified_n = len(citations) - len(unverified)
            if unverified:
                bad = ", ".join(f"{c['company']} {c['year']+' ' if c['year'] else ''}p.{c['page']}" for c in unverified)
                st.markdown(f"""
                <div class="verify-line">
                    <span class="stamp stamp-red">{len(unverified)} UNVERIFIED</span>
                    &nbsp;{verified_n}/{len(citations)} citations matched retrieved excerpts exactly.
                    Flagged: {bad} — not found among retrieved pages, treat as unconfirmed.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verify-line">
                    <span class="stamp stamp-green">VERIFIED</span>
                    &nbsp;{verified_n}/{len(citations)} citations matched retrieved excerpts exactly.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="verify-line">
                <span class="stamp stamp-red">NO CITATIONS</span>
                &nbsp;The memo made no verifiable (Company, p.N) citation — treat any claim above as unsupported.
            </div>
            """, unsafe_allow_html=True)

        if usage:
            cost = estimate_cost(usage)
            st.markdown(
                f'<div class="verify-line">{usage["input_tokens"]:,} input / '
                f'{usage["output_tokens"]:,} output tokens · ~${cost:.3f} this query</div>',
                unsafe_allow_html=True,
            )

        pdf_bytes = build_memo_pdf(
            query=query, answer=answer, chunks=chunks, citations=citations,
            best_score=best_score, low_confidence=best_score < LOW_CONFIDENCE,
            low_confidence_threshold=LOW_CONFIDENCE,
        )
        st.download_button(
            "Export this memo as PDF",
            data=pdf_bytes,
            file_name=f"dax-intelligence-memo-{datetime.now().strftime('%Y%m%d-%H%M')}.pdf",
            mime="application/pdf",
        )

        # -------------------------------------------------------------
        # Audit trail — ground truth from retrieval, independent of
        # anything the model claims above.
        # -------------------------------------------------------------
        st.markdown(f'<div class="audit-title">Audit trail — {len(chunks)} excerpts retrieved</div>', unsafe_allow_html=True)

        cited_pages = {(c["company"], c["year"], c["page"]) for c in citations if c["verified"]}
        years_present = {str(c["year"]) for c in chunks if c["year"]}

        rows = []
        for c in chunks:
            meta = company_meta(c["company"])
            logo_uri = logo_data_uri(meta["logo"]) if meta.get("logo") else ""
            start, end = c["approx_page"], c.get("end_page", c["approx_page"])
            is_cited = any(
                cc == c["company"]
                and (cy is None or cy == str(c["year"] or ""))
                and start - 1 <= cp <= end + 1
                for cc, cy, cp in cited_pages
            )
            flag = '<span class="flag-ok">CITED</span>' if is_cited else '<span class="flag-dim">retrieved</span>'
            img_tag = f'<img src="{logo_uri}" />' if logo_uri else ""
            if c["score"] is not None:
                score_html = f'<span class="sc">{c["score"]:.3f}</span>'
            else:
                score_html = '<span class="sc kw">keyword</span>'
            via_mark = ' <span class="via-both" title="Found by both semantic and keyword search">·</span>' if c.get("via") == "both" else ""

            pdf_url = report_pdf_url(c["company"], str(c["year"] or "2025"), c["approx_page"])
            year_tag = f"FY{str(c['year'])[-2:]} · " if len(years_present) > 1 and c["year"] else ""
            page_text = f"{year_tag}{page_label(c)}"
            if pdf_url:
                page_html = f'<a href="{pdf_url}" target="_blank" title="Open the source report at this page">{page_text}</a>'
            else:
                page_html = page_text

            rows.append(f"""
            <div class="audit-row">
                {img_tag}
                <span class="co">{meta['name']}</span>
                <span class="sec">{(c['section'] or '').replace('_',' ').title()}</span>
                <span class="pg">{page_html}{via_mark}</span>
                {score_html}
                {flag}
            </div>
            """)
        st.markdown("".join(rows), unsafe_allow_html=True)

        with st.expander("Read the retrieved excerpts in full"):
            for c in chunks:
                meta = company_meta(c["company"])
                st.markdown(f"**{meta['name']} — {(c['section'] or '').replace('_',' ').title()} — {page_label(c)}**")
                st.caption(c["text"][:600] + ("…" if len(c["text"]) > 600 else ""))
                st.markdown("---")

# ---------------------------------------------------------------------------
# Fill the index-roll slot (reserved above the query) now that we know which
# constituents were considered/cited by this run, if any.
# ---------------------------------------------------------------------------
tiles = []
for c in COMPANIES:
    if c["key"] not in present_companies:
        continue
    cls = "roll-tile"
    if c["key"] in cited_companies:
        cls += " cited"
    elif c["key"] in considered_companies:
        cls += " considered"
    logo_uri = logo_data_uri(c["logo"])
    tiles.append(f'<div class="{cls}"><img src="{logo_uri}" title="{c["name"]}"/><span class="tick">{c["ticker"]}</span></div>')
roll_slot.markdown(f'<div class="roll-wrap">{"".join(tiles)}</div>', unsafe_allow_html=True)

_photo = author_photo_uri()
_photo_tag = f'<img src="{_photo}"/>' if _photo else ""
st.markdown(
    f'<div class="byline">{_photo_tag}<span>ChromaDB retrieval · LLM synthesis · '
    'every claim checked against its source excerpt · built by Moritz Richter</span></div>',
    unsafe_allow_html=True,
)
