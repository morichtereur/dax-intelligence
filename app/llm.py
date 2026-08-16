import html as _html
import os
import re
import anthropic
from dotenv import load_dotenv

from app.branding import COMPANIES

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-5"

# List price, $ per 1M tokens — keyed by model id so both ask()'s generation
# calls and the eval harness's judge calls (a cheaper model) price correctly
# from the same table.
PRICING_PER_MILLION = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


def estimate_cost(usage: dict | None, model: str = MODEL) -> float:
    """USD estimate for one API call from its usage dict, at the given
    model's list price (defaults to MODEL, the generation model)."""
    if not usage:
        return 0.0
    price = PRICING_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
    return (usage["input_tokens"] / 1_000_000 * price["input"]
            + usage["output_tokens"] / 1_000_000 * price["output"])


SYSTEM_PROMPT = """You are a senior equity research analyst producing an internal research memo built strictly from excerpts of DAX 40 annual reports.

Ground rules, in order of priority:
1. Use only the information in the excerpts below. Do not draw on outside knowledge of these companies, their industries, or general macroeconomic context, even where you are confident it is correct.
2. Every factual claim carries an inline citation in the exact format (Company, p.N), placed immediately after the sentence it supports, where Company matches one of the company names given in the excerpts and N is a single page number. Some excerpts are labeled with a page range (e.g. pp.141-143) because the underlying chunk spans multiple pages — pick the single page within that range the specific claim actually comes from rather than citing the whole range. If the excerpts include more than one fiscal year for that company, cite the year too, in the format (Company YYYY, p.N) — pull YYYY from the excerpt's own bracket label, never assume it.
3. Never cite a company, year or page number that is not present in the supplied excerpts. If you are not certain an excerpt supports a claim, drop the claim rather than guess the citation.
4. If the excerpts do not cover the question — fully or partially — say so plainly as the first line (e.g. "The retrieved excerpts do not cover this.") and stop. Do not speculate or fill the gap with general knowledge.
5. Where reports disagree or frame the same topic differently, say so explicitly rather than smoothing it into one narrative. When the excerpts span two fiscal years for the same company, call out what changed year-over-year rather than only describing the latest year.
6. Write like a research memo: precise, no hedging filler ("it's worth noting that…", "it's important to remember...").

Format: short paragraphs, one idea each. Use bullet points only for direct side-by-side comparisons."""

# Matches "(Company, p.45)" / "(Company p.~45)" / "(Company 2025, p.45)" /
# "(Company, p.45-47)" — the year group is captured (not just tolerated) so
# a citation can be checked against the specific fiscal year it names, which
# matters once excerpts span more than one year for the same company and
# "p.68" alone is ambiguous. A trailing "-NN" page-range suffix is matched
# but not captured: excerpt labels now show multi-page ranges (page_label()
# below) for chunks spanning more than one page, and the model sometimes
# echoes that shape back in its own citation — the start page alone is
# enough for verification against the chunk's real [start, end] range.
CITATION_RE = re.compile(
    r"\(([A-Z][\w&.\-]*(?:\s+[A-Z&][\w&.\-]*){0,3}),?\s*(?:(\d{4})\s*,?\s*)?p\.\s*~?(\d+)(?:[-–]\d+)?\)"
)


def page_label(c: dict) -> str:
    start, end = c["approx_page"], c.get("end_page", c["approx_page"])
    return f"p.{start}" if start == end else f"pp.{start}-{end}"


def ask(query: str, chunks: list[dict], compare_mode: bool = False) -> dict:
    """Returns {"text": answer prose, "usage": {input_tokens, output_tokens}}."""
    context = ""
    for c in chunks:
        context += f"\n---\n[{c['company']} {c['year']} | {c['section']} | {page_label(c)}]\n{c['text']}\n"

    if compare_mode:
        user_prompt = f"""Compare across companies based on the following excerpts.

Question: {query}

Excerpts:
{context}

Highlight similarities, differences, and any notable outliers. Every claim still needs its (Company, p.N) citation."""
    else:
        user_prompt = f"""Answer the following question based solely on the excerpts below.

Question: {query}

Excerpts:
{context}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return {"text": response.content[0].text, "usage": usage}


def _pages_by_company(chunks: list[dict]) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """company -> year -> list of (start_page, end_page) ranges actually
    retrieved. Keyed by year (not just company) so a citation naming a year
    can be checked against that exact fiscal year rather than any year the
    company happens to have on file. Ranges come from pipeline/chunking.py's
    real word-level page tracking, not an estimate, so verification checks
    genuine containment rather than a fixed +/-N fudge factor."""
    pages: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for c in chunks:
        try:
            start = int(c["approx_page"])
            end = int(c.get("end_page", start))
        except (TypeError, ValueError):
            continue
        year = str(c.get("year") or "")
        pages.setdefault(c["company"], {}).setdefault(year, []).append((start, end))
    return pages


def _norm(s: str) -> str:
    """Strips everything but letters/digits so "Munich Re", "MunichRe" and
    "Munich-Re" all compare equal — the model tends to write the natural
    company name even when the context literally shows the metadata key."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


# The model sometimes cites a company by its real name ("Volkswagen") rather
# than the metadata key it was literally shown ("VW") — those two share no
# substring, so fuzzy matching alone misses it. This closes the gap using
# the same name/ticker table the UI already maintains.
_ALIAS_TO_KEY = {}
for _c in COMPANIES:
    _ALIAS_TO_KEY[_norm(_c["key"])] = _c["key"]
    _ALIAS_TO_KEY[_norm(_c["name"])] = _c["key"]
    _ALIAS_TO_KEY[_norm(_c["ticker"])] = _c["key"]


def _resolve(raw_company: str, year: str | None, page: int, pages_by_company: dict[str, dict[str, list[tuple[int, int]]]]):
    """Match a citation's company (and, if given, fiscal year) against
    retrieved chunk metadata, confirming the page falls within a real
    retrieved chunk's page range (+/-1, for a citation landing just past a
    chunk boundary) — a genuine containment check now that ranges are exact,
    not the +/-3 point-estimate fudge factor the old linear-interpolation
    pager needed. With no year named, any year retrieved for that company
    counts — the common case where only one year is in scope and naming it
    is optional."""
    raw_norm = _norm(raw_company)
    aliased = _ALIAS_TO_KEY.get(raw_norm)
    if aliased in pages_by_company:
        resolved = aliased
    else:
        resolved = next(
            (k for k in pages_by_company
             if _norm(k) == raw_norm
             or _norm(k) in raw_norm
             or raw_norm in _norm(k)),
            None
        )
    if resolved is None:
        return None, False

    years_pages = pages_by_company[resolved]
    ranges: list[tuple[int, int]] = years_pages.get(year, []) if year else [r for rs in years_pages.values() for r in rs]
    verified = any(start - 1 <= page <= end + 1 for start, end in ranges)
    return resolved, verified


def verify_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Cross-check every (Company[, YYYY], p.N) citation the model wrote
    against the chunks it was actually given. This is a ground-truth check,
    not a second model opinion — a citation is only "verified" if that
    company/year/page combination was literally in the retrieved context."""
    pages_by_company = _pages_by_company(chunks)
    seen = set()
    citations = []
    for match in CITATION_RE.finditer(answer):
        raw_company, year, page = match.group(1).strip(), match.group(2), int(match.group(3))
        key = (raw_company.lower(), year, page)
        if key in seen:
            continue
        seen.add(key)
        resolved, verified = _resolve(raw_company, year, page, pages_by_company)
        citations.append({
            "company": resolved or raw_company,
            "year": year,
            "page": page,
            "verified": verified,
        })
    return citations


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def highlight_citations(answer: str, chunks: list[dict]) -> str:
    """Return the answer as fully-formed HTML — headings, bold and citation
    verification spans all resolved — for inline rendering next to the
    prose that made each claim, not just in a summary line below it.

    Streamlit's st.markdown treats a string starting with a block tag like
    <div> as a raw CommonMark "HTML block": once inside it, no Markdown
    syntax is processed at all, so passing the LLM's raw "## Heading" /
    "**bold**" straight through a <div> wrapper renders the literal hashes
    and asterisks instead of formatting them. Converting Markdown to HTML
    ourselves here sidesteps that rather than fighting the parser."""
    pages_by_company = _pages_by_company(chunks)

    def _sub(m: re.Match) -> str:
        raw_company, year, page = m.group(1).strip(), m.group(2), int(m.group(3))
        _, verified = _resolve(raw_company, year, page, pages_by_company)
        cls = "cite-ok" if verified else "cite-flag"
        return f'<span class="{cls}">{m.group(0)}</span>'

    blocks = []
    for para in answer.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        escaped = _html.escape(para, quote=False).replace("\n", "<br/>")
        escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
        escaped = CITATION_RE.sub(_sub, escaped)
        if para.startswith("#"):
            blocks.append(f"<h4>{escaped.lstrip('#').strip()}</h4>")
        else:
            blocks.append(f"<p>{escaped}</p>")
    return "".join(blocks)
