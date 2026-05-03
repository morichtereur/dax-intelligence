import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from app.retriever import retrieve, get_companies
from app.llm import ask

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
        "Context depth",
        min_value=4,
        max_value=16,
        value=8,
        help="More chunks = broader context, slower response"
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

    with st.spinner("Searching reports..."):
        chunks = retrieve(query, n_results=n_results, company_filter=selected_company)

    with st.spinner("Synthesising answer..."):
        answer = ask(query, chunks, compare_mode=compare_mode)

    st.markdown("### Answer")
    st.markdown(answer)

    # Source companies used
    source_companies = sorted(set(c["company"] for c in chunks))
    st.markdown(
        "**Sources:** " + " · ".join(f"`{c}`" for c in source_companies)
    )

    with st.expander(f"📄 Source chunks ({len(chunks)})"):
        for chunk in chunks:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f"**{chunk['company']} {chunk['year']}** — "
                    f"{chunk['section'].replace('_', ' ').title()} — "
                    f"p.~{chunk['approx_page']}"
                )
                st.caption(chunk['text'][:400] + "...")
            with col_b:
                st.metric("Relevance", f"{chunk['score']:.2f}")
            st.divider()