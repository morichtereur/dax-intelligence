import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from app.retriever import relevant_companies, get_companies
from app.llm import ask, estimate_cost

st.set_page_config(
    page_title="DAX Intelligence",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { max-width: 900px; }
    .stTextInput > div > div > input { font-size: 16px; }
    .answer-box { 
        background: #1a1a2e; 
        border-left: 4px solid #e31937;
        padding: 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .source-tag {
        background: #2d2d2d;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 4px;
    }
    .metric-card {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 DAX Intelligence")
st.caption("AI-powered analysis of DAX 40 annual reports — built for finance & strategy professionals")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Query Settings")
    companies = get_companies()

    company_filter = st.selectbox(
        "Company scope",
        options=["All companies"] + companies,
        help="Filter to a single company or query across all"
    )

    compare_mode = st.toggle(
        "Compare mode",
        value=False,
        help="Structures the answer as a cross-company comparison"
    )

    n_results = st.slider(
        "Retrieval depth",
        min_value=4,
        max_value=16,
        value=8,
        help="How many chunk hits to scan when ranking which companies' "
             "reports are relevant — the full PDFs for those companies (not "
             "just the matched chunks) are what Claude actually reads"
    )

    max_companies = st.slider(
        "Max companies per query",
        min_value=1,
        max_value=8,
        value=4,
        help="Caps cost/context: each company attaches its full trimmed "
             "report PDF, not just a snippet"
    )

    st.divider()
    st.markdown("### 📁 Loaded Reports")
    for c in companies:
        st.markdown(f"• {c} (FY2025)")

    st.divider()
    st.markdown(
        "Built with ChromaDB + Claude Sonnet · "
        "[GitHub](https://github.com) · "
        "by Moritz Richter"
    )

# Stats row
col1, col2, col3 = st.columns(3)
from app.retriever import collection
total_chunks = collection.count()
with col1:
    st.metric("Companies", len(companies))
with col2:
    st.metric("Total Chunks", f"{total_chunks:,}")
with col3:
    st.metric("Reports", f"FY2025")

st.divider()

# Query input
query = st.text_input(
    "Ask a question about DAX 40 annual reports",
    placeholder="e.g. How do Siemens and SAP frame their AI investment strategy?",
    label_visibility="collapsed"
)

# Example queries
examples = [
    "Which companies disclosed structural cost reduction programs?",
    "How is VW addressing the EV transition financially?",
    "What do CFOs say about macroeconomic risks in 2025?",
    "Compare how BMW and Mercedes frame EV investment costs",
    "Which companies mentioned GBS or shared services transformation?",
    "What are the key strategic priorities across DAX industrials?",
]

with st.expander("💡 Example queries"):
    cols = st.columns(2)
    for i, eq in enumerate(examples):
        with cols[i % 2]:
            if st.button(eq, key=eq, use_container_width=True):
                query = eq

# Answer
if query:
    selected_company = None if company_filter == "All companies" else company_filter

    with st.spinner("Finding relevant reports..."):
        matched_companies = relevant_companies(
            query, n_results=n_results, company_filter=selected_company,
            max_companies=max_companies,
        )

    if not matched_companies:
        st.warning("No matching reports found for this query.")
    else:
        with st.spinner(f"Reading {len(matched_companies)} report(s) and synthesising an answer..."):
            result = ask(query, matched_companies, compare_mode=compare_mode)

        st.markdown("### Answer")
        st.markdown(result["text"])

        st.markdown(
            "**Sources:** " + " · ".join(f"`{c}`" for c in matched_companies)
        )

        usage = result["usage"]
        if usage:
            cost = estimate_cost(usage)
            st.caption(
                f"~{usage['input_tokens']:,} input / {usage['output_tokens']:,} output "
                f"tokens · ~${cost:.3f} this query"
            )

        citations = result["citations"]
        with st.expander(f"📄 Citations ({len(citations)})"):
            if not citations:
                st.caption("No page-level citations were returned for this answer.")
            for cite in citations:
                page_range = (f"p. {cite['start_page']}" if cite['start_page'] == cite['end_page']
                              else f"pp. {cite['start_page']}–{cite['end_page']}")
                st.markdown(f"**{cite['document_title']}** — {page_range}")
                st.caption(f"“{cite['cited_text'][:300]}”")
                st.divider()