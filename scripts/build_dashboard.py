"""Build dashboard.html — a set of recorded runs against the real index.

The app itself cannot be deployed publicly. It needs a 373 MB Chroma index and
3.6 GB of source PDFs, neither of which belongs in git, and a live endpoint
would put an API key behind an unmetered public text box. So this records the
runs instead: every answer, citation, verification verdict and retrieved
excerpt on this page came out of the real pipeline against the real corpus, and
the page says so rather than implying a live system.

What it is meant to show is the guardrail, not the prose. Four behaviours:

  * a normal answer, with every citation checked back against the excerpts the
    model was actually handed;
  * a question the excerpts do not actually support, which the model declines
    rather than answering unsourced — the interesting one;
  * a query the corpus only half covers, which trips the low-confidence band;
  * a query it does not cover at all, which never reaches the model.

Recorded runs read as claims unless you can check them, so the page is built to
be walked rather than read: every citation in the prose is a control that jumps
to the excerpt it was checked against, every retrieval carries its score
against the two gates, and the audit trail can be filtered to the lane each
excerpt arrived on.

Run:  make dashboard              (COSTS REAL API CALLS — one per query)
      make dashboard-rerender     (free — re-renders the recorded runs)
"""

from __future__ import annotations

import base64
import html
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm import MODEL, ask, estimate_cost, verify_citations  # noqa: E402
from app.retriever import (  # noqa: E402
    LOW_CONFIDENCE,
    MIN_RELEVANCE,
    collection,
    get_companies,
    get_years,
    retrieve,
)

# Five drawn from eval/gold_queries.json and the README's own examples, plus
# two chosen for what they fail at. Tesla is not in the corpus but shares its
# vocabulary with three carmakers that are, which is exactly the case the
# confidence band exists for; sourdough shares nothing, and is the case the
# hard floor exists for.
QUERIES = [
    {
        "q": "How do Siemens and SAP frame their AI investment strategy?",
        "compare": True,
        "note": "The comparative case: one question, two filings, every claim carrying its own page.",
    },
    {
        "q": "Compare how BMW and Mercedes-Benz frame EV investment costs",
        "compare": True,
        "note": "Two companies describing the same transition in different language.",
    },
    {
        "q": "What does Allianz's CFO say about macroeconomic risk?",
        "compare": False,
        "note": "The corpus holds Allianz's risk chapter but no CFO commentary on macro risk. "
        "Watch what the model does with that.",
    },
    {
        "q": "Which companies disclosed structural cost reduction programs?",
        "compare": True,
        "note": "Open across the whole corpus — the retriever decides who is in the answer.",
    },
    {
        "q": "What does Tesla say about its battery supply chain?",
        "compare": False,
        "note": "Tesla is not in this corpus at all. The question still retrieves, because "
        "carmakers talk alike — which is the failure the confidence band exists for.",
    },
    {
        "q": "How do I bake sourdough bread at home?",
        "compare": False,
        "note": "Nothing clears the relevance floor, so the model is never called at all.",
    },
]

# The index keys each report by a compact company key. These are the names
# those companies publish under, and are presentation only — the key is what
# the retrieval actually matched on.
DISPLAY_NAMES = {
    "Mercedes": "Mercedes-Benz",
    "Merck": "Merck KGaA",
    "MunichRe": "Munich Re",
    "SiemensEnergy": "Siemens Energy",
    "VW": "Volkswagen",
}

CITATION_RE = re.compile(r"\(([A-Za-z][\w .&-]*?)(?:,?\s*(20\d{2}))?,\s*p\.\s*(\d+)\)")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def run_one(spec: dict) -> dict:
    """One query, end to end, recording what the app would have shown."""
    query = spec["q"]
    print(f"  {query}")
    chunks = retrieve(query, n_results=8)

    scored = [c["score"] for c in chunks if c["score"] is not None]
    best = max(scored) if scored else None

    record = {
        "run_date": date.today().isoformat(),
        "query": query,
        "note": spec["note"],
        "compare": spec["compare"],
        "best_score": best,
        "chunks": [
            {
                "company": c["company"],
                "year": c["year"],
                "section": c["section"],
                "page": c["approx_page"],
                "end_page": c["end_page"],
                "score": c["score"],
                "via": c["via"],
                "text": c["text"][:420].rsplit(" ", 1)[0] + "…",
            }
            for c in chunks
        ],
    }

    if not chunks:
        record["outcome"] = "gated"
        print("    -> gated, no model call")
        return record

    result = ask(query, chunks, compare_mode=spec["compare"])
    citations = verify_citations(result["text"], chunks)

    record["answer"] = result["text"]
    record["citations"] = citations
    record["usage"] = result["usage"]
    record["cost"] = estimate_cost(result["usage"], MODEL)
    # A model that was handed excerpts and wrote no citation has refused, and
    # that is a different event from a weak retrieval — the prompt forbids an
    # unsourced answer, so this is the guardrail firing rather than failing.
    record["low_confidence"] = best < LOW_CONFIDENCE
    record["outcome"] = "declined" if not citations else "answered"

    verified = sum(1 for c in citations if c["verified"])
    flag = " (low confidence)" if record["low_confidence"] else ""
    if record["outcome"] == "declined":
        print(f"    -> declined, no source in the excerpts{flag}")
    else:
        print(f"    -> answered, {verified}/{len(citations)} verified{flag}")
    return record


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def link_citations(record: dict) -> None:
    """Resolve every citation to the excerpt it was checked against.

    verify_citations records the verdict but not which excerpt produced it,
    and the verdict is the only interesting thing on the page — so re-run the
    same company/page match here and keep the index. The page turns that into
    a control: click a citation, land on the excerpt that justifies it."""
    for cit in record.get("citations", []):
        target, tightest = None, None
        for i, c in enumerate(record["chunks"]):
            if _norm(c["company"]) != _norm(cit["company"]):
                continue
            if cit.get("year") and c["year"] != cit["year"]:
                continue
            start = c["page"]
            end = c["end_page"] or start
            # The same +/-1 tolerance verify_citations allows, for a claim
            # that sits a line either side of a chunk's page boundary.
            if start - 1 <= cit["page"] <= end + 1:
                span = end - start
                if tightest is None or span < tightest:
                    target, tightest = i, span
        cit["chunk"] = target


def render_answer(text: str, citations: list[dict]) -> str:
    """The model writes markdown. Render the subset it actually uses —
    headings, bullets, bold — and turn every citation into a control carrying
    its verdict, which is the treatment app.py gives it, so this page shows
    what the app shows."""
    index: dict[tuple[str, str | None, int], int] = {}
    for i, c in enumerate(citations):
        index[(_norm(c["company"]), c.get("year"), c["page"])] = i
        index.setdefault((_norm(c["company"]), None, c["page"]), i)

    def cite(m: re.Match) -> str:
        company, year, page = _norm(m.group(1)), m.group(2), int(m.group(3))
        i = index.get((company, year, page), index.get((company, None, page)))
        if i is None:  # a company name the resolver normalised differently
            i = next(
                (
                    j
                    for (comp, _, pg), j in index.items()
                    if pg == page and (comp in company or company in comp)
                ),
                None,
            )
        if i is None:
            return f'<span class="cite">{html.escape(m.group(0))}</span>'
        c = citations[i]
        cls = "cite ok" if c["verified"] else "cite bad"
        label = "verified against a retrieved excerpt" if c["verified"] else "no matching excerpt"
        return (
            f'<button type="button" class="{cls}" data-cite="{i}" '
            f'title="{label} — click to see the excerpt">{html.escape(m.group(0))}</button>'
        )

    def inline(s: str) -> str:
        s = html.escape(s)
        s = BOLD_RE.sub(r"<strong>\1</strong>", s)
        return CITATION_RE.sub(cite, s)

    out: list[str] = []
    para: list[str] = []
    items: list[str] = []

    def flush_para() -> None:
        if para:
            out.append("<p>" + "<br>".join(inline(l) for l in para) + "</p>")
            para.clear()

    def flush_list() -> None:
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_para()
            flush_list()
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_para()
            flush_list()
            # The model's own h1 is the answer's title, which sits under the
            # query -- so everything shifts down two levels.
            level = min(len(heading.group(1)) + 2, 6)
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            continue

        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet:
            flush_para()
            items.append(bullet.group(1))
            continue

        flush_list()
        para.append(stripped)

    flush_para()
    flush_list()
    return "".join(out)


def coverage(records: list[dict]) -> list[dict]:
    """Which of the fifteen reports these runs actually pulled from, and how
    often. Coverage of the corpus is a claim; this is the measurement."""
    years = get_years()
    counts: dict[tuple[str, str], int] = {}
    for r in records:
        for c in r["chunks"]:
            counts[(c["company"], c["year"])] = counts.get((c["company"], c["year"]), 0) + 1
    return [
        {
            "key": company,
            "name": DISPLAY_NAMES.get(company, company),
            "years": [counts.get((company, y), 0) for y in years],
        }
        for company in get_companies()
    ]


def font_face(name: str, file: str, weight: int) -> str:
    data = base64.b64encode((ROOT / "assets" / "fonts" / file).read_bytes()).decode()
    return (
        f"@font-face{{font-family:'{name}';font-style:normal;font-weight:{weight};"
        f"font-display:swap;src:url(data:font/woff2;base64,{data}) format('woff2')}}"
    )


RUNS = ROOT / "assets" / "dashboard_runs.json"


def build(from_cache: bool = False) -> None:
    """Re-rendering the page should not cost money. The runs are recorded once
    into assets/dashboard_runs.json and committed, so anyone can rebuild this
    page from the repository without the index, the filings or an API key."""
    if from_cache:
        records = json.loads(RUNS.read_text(encoding="utf-8"))
        print(f"Re-rendering {len(records)} cached runs — no API calls.")
    else:
        print(f"Recording {len(QUERIES)} runs against {collection.count()} chunks…")
        records = [run_one(spec) for spec in QUERIES]
        RUNS.write_text(
            json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )

    total_cost = sum(r.get("cost") or 0 for r in records)
    answered = [r for r in records if r["outcome"] == "answered"]
    all_cites = [c for r in records for c in r.get("citations", [])]
    verified = sum(1 for c in all_cites if c["verified"])

    for r in records:
        link_citations(r)
        r["answer_html"] = (
            render_answer(r["answer"], r["citations"]) if r["outcome"] != "gated" else ""
        )
        for c in r["chunks"]:
            c["name"] = DISPLAY_NAMES.get(c["company"], c["company"])

    unresolved = [c for c in all_cites if c["verified"] and c.get("chunk") is None]
    if unresolved:
        print(f"  warning: {len(unresolved)} verified citations resolved to no excerpt")

    years = get_years()
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    meta = json.dumps(
        {
            "minRelevance": MIN_RELEVANCE,
            "lowConfidence": LOW_CONFIDENCE,
            "years": years,
            "coverage": coverage(records),
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")

    fonts = "".join(
        [
            font_face("Archivo", "archivo-400.woff2", 400),
            font_face("Archivo", "archivo-700.woff2", 700),
            font_face("Source Serif 4", "source-serif-400.woff2", 400),
            font_face("IBM Plex Mono", "plex-mono-400.woff2", 400),
        ]
    )

    tokens = {
        "__FONTS__": fonts,
        "__PAYLOAD__": payload,
        "__META__": meta,
        "__NREPORTS__": str(len(get_companies())),
        "__YEARS__": "–".join([years[0], years[-1]]),
        "__CHUNKS__": f"{collection.count():,}",
        "__MODEL__": MODEL,
        "__MINRELEVANCE__": f"{MIN_RELEVANCE:.2f}",
        "__LOWCONFIDENCE__": f"{LOW_CONFIDENCE:.2f}",
        "__RUNDATE__": records[0].get("run_date", date.today().isoformat()),
        "__NQUERIES__": str(len(records)),
        "__NANSWERED__": str(len(answered)),
        "__NCITES__": str(len(all_cites)),
        "__NVERIFIED__": str(verified),
        "__TOTALCOST__": f"{total_cost:.3f}",
    }

    html_out = TEMPLATE
    for token, value in tokens.items():
        html_out = html_out.replace(token, value)

    out = ROOT / "dashboard.html"
    out.write_text(html_out, encoding="utf-8")
    size = out.stat().st_size / 1024
    print(f"\nWrote {out.name} ({size:.0f} KB)")
    print(f"  {verified}/{len(all_cites)} citations verified across {len(answered)} answers")
    print(f"  total generation cost USD {total_cost:.3f}")


# The page is one file: fonts, data and behaviour all inline, so it can be
# opened from disk, served from a static host, or archived, with nothing to
# fetch. Placeholders are __TOKENS__ rather than format fields, so the CSS and
# JS below keep their own braces.
TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answers with the citation check attached — DAX Intelligence</title>
<meta name="description" content="Six recorded runs of a retrieval-augmented research assistant over
fifteen DAX annual reports. Every citation is checked against the excerpts the model was handed,
and every retrieval carries its score against the two gates.">
<meta property="og:title" content="Answers with the citation check attached">
<meta property="og:description" content="Six recorded RAG runs over fifteen DAX annual reports —
every citation checked against the excerpts the model was actually given.">
<meta property="og:type" content="article">
<style>
__FONTS__
:root {
  color-scheme: light;
  --sans:'Archivo',system-ui,-apple-system,sans-serif;
  --serif:'Source Serif 4',Georgia,'Times New Roman',serif;
  --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;

  --paper:#ffffff; --surface:#f4f6f1; --surface-2:#eceee7;
  --ink:#121a17; --ink-2:#59635e; --ink-3:#848b81;
  --rule:#dee2da; --rule-strong:#848b81;
  --accent:#146b54; --accent-soft:rgba(20,107,84,.10);
  --band:#113a2e; --band-text:#eef0e9; --band-text-2:rgba(238,240,233,.72);
  --band-line:rgba(238,240,233,.28); --band-accent:#8fd0b9;
  --warn:#8a5a12; --warn-soft:rgba(138,90,18,.10);
  --bad:#9a2f2f; --bad-soft:rgba(154,47,47,.10);
  --lit:rgba(20,107,84,.14);
}
@media (prefers-color-scheme:dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper:#101312; --surface:#181c1a; --surface-2:#1f2422;
    --ink:#eceeef; --ink-2:#aab3ad; --ink-3:#7f8a84;
    --rule:#272d2a; --rule-strong:#4a534e;
    --accent:#5cc6a2; --accent-soft:rgba(92,198,162,.12);
    --band:#0c2a21; --band-text:#eef0e9; --band-text-2:rgba(238,240,233,.72);
    --band-line:rgba(238,240,233,.22); --band-accent:#8fd0b9;
    --warn:#d9a441; --warn-soft:rgba(217,164,65,.12);
    --bad:#e0736b; --bad-soft:rgba(224,115,107,.12);
    --lit:rgba(92,198,162,.14);
  }
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-size:17px;line-height:1.6;-webkit-font-smoothing:antialiased}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.inner{max-width:64rem;margin:0 auto;padding:0 1.5rem}
.mono{font-family:var(--mono);font-size:.85em}
.label{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 .75rem}

/* --- masthead band ------------------------------------------------------ */
.band{background:var(--band);color:var(--band-text);padding:4rem 0 0}
.band .label{color:var(--band-text-2)}
.band h1{font-family:var(--serif);font-weight:400;font-size:clamp(2.1rem,5.2vw,3.6rem);
  line-height:1.04;letter-spacing:-.03em;margin:0 0 1.25rem;max-width:20ch}
.band .standfirst{color:var(--band-text-2);margin:0 0 2.5rem;max-width:42rem;font-size:1.0625rem}
.band .prov{font-family:var(--mono);font-size:.72rem;letter-spacing:.04em;color:var(--band-text-2);
  margin:0;padding:1rem 0 2.5rem;border-top:1px solid var(--band-line)}
.band .prov a{color:var(--band-accent)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));
  border-top:1px solid var(--band-line)}
.fact{padding:1rem 1.25rem 1.1rem 0}
.fact .k{display:block;font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--band-text-2);margin-bottom:.35rem}
.fact .v{font-family:var(--sans);font-weight:700;font-size:1.7rem;letter-spacing:-.02em;
  line-height:1}
.fact .v small{font-size:.9rem;font-weight:400;color:var(--band-text-2)}

/* --- sections ----------------------------------------------------------- */
main{padding:3.5rem 0 0}
section{margin:0 0 4rem}
h2{font-family:var(--serif);font-weight:400;font-size:1.9rem;line-height:1.15;
  letter-spacing:-.02em;margin:0 0 .6rem}
.lede{color:var(--ink-2);margin:0 0 1.75rem;max-width:42rem}
.plate{background:var(--surface);border:1px solid var(--rule);padding:1.5rem}
.plate p{margin:.7rem 0 0} .plate p:first-of-type{margin-top:0}
.plate strong{font-family:var(--sans);font-weight:700;font-size:.95em;letter-spacing:-.01em}

/* --- pipeline diagram --------------------------------------------------- */
.exhibit{border:1px solid var(--rule);background:var(--surface)}
.exhibit .cap{font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);padding:.7rem 1rem;border-bottom:1px solid var(--rule)}
.exhibit .body{padding:1.25rem 1rem;overflow-x:auto}
.diagram{display:block;min-width:52rem;width:100%;height:auto;color:var(--ink)}
.diagram .box{fill:var(--paper);stroke:var(--rule-strong);stroke-width:1}
.diagram .box.term{fill:var(--surface-2);stroke:var(--rule)}
.diagram .box.acc{stroke:var(--accent)}
.diagram .ttl{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;
  fill:var(--ink-3)}
.diagram .txt{font-family:var(--serif);font-size:13px;fill:var(--ink)}
.diagram .num{font-family:var(--mono);font-size:11px;fill:var(--accent)}
.diagram .num.mute{fill:var(--ink-3)}
.diagram .flow{stroke:var(--rule-strong);stroke-width:1;fill:none}
.diagram .flow.dash{stroke-dasharray:3 3}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:0;
  border:1px solid var(--rule);border-bottom:0;margin-top:1.5rem}
.legend div{padding:1rem 1.15rem;border-bottom:1px solid var(--rule)}
.legend div+div{border-left:1px solid var(--rule)}
.legend .k{font-family:var(--mono);font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;
  display:block;margin-bottom:.4rem}
.legend p{margin:0;font-size:.92rem;color:var(--ink-2)}
.k-ok{color:var(--accent)} .k-warn{color:var(--warn)} .k-gate{color:var(--ink-3)}

/* --- runs: rail + panel ------------------------------------------------- */
.runs{display:grid;grid-template-columns:minmax(14rem,18rem) 1fr;gap:2.5rem;align-items:start}
.rail{position:sticky;top:1.5rem;display:flex;flex-direction:column;
  border-top:1px solid var(--rule)}
.tab{text-align:left;background:none;border:0;border-bottom:1px solid var(--rule);
  color:var(--ink);font-family:var(--serif);font-size:.95rem;line-height:1.35;
  padding:.85rem 0 .85rem .9rem;cursor:pointer;position:relative;display:block;width:100%}
.tab:before{content:"";position:absolute;left:0;top:-1px;bottom:-1px;width:2px;background:transparent}
.tab:hover{background:var(--surface)}
.tab[aria-selected="true"]{background:var(--surface)}
.tab[aria-selected="true"]:before{background:var(--accent)}
.tab .n{font-family:var(--mono);font-size:.68rem;color:var(--ink-3);display:block;
  margin-bottom:.3rem}
.tab .badge{display:block;font-family:var(--mono);font-size:.64rem;letter-spacing:.08em;
  text-transform:uppercase;margin-top:.45rem}

.q{font-family:var(--serif);font-weight:400;font-size:1.55rem;line-height:1.2;
  letter-spacing:-.02em;margin:0 0 .5rem}
.note{color:var(--ink-2);font-size:.95rem;margin:0 0 1.25rem;max-width:58ch}
.verdict{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:0 0 1.5rem}
.stamp{font-family:var(--mono);font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;
  padding:.35rem .6rem}
.s-ok{background:var(--accent-soft);color:var(--accent)}
.s-warn{background:var(--warn-soft);color:var(--warn)}
.s-gate{background:var(--surface-2);color:var(--ink-3)}
.meta{font-family:var(--mono);font-size:.7rem;color:var(--ink-3);letter-spacing:.04em}
.banner{border-left:2px solid var(--warn);background:var(--warn-soft);padding:.9rem 1rem;
  margin:0 0 1.5rem;font-size:.95rem}
.banner.gate{border-left-color:var(--rule-strong);background:var(--surface);color:var(--ink-2)}
.banner strong{font-family:var(--sans);font-weight:700;font-size:.92em}

/* --- the answer --------------------------------------------------------- */
.answer{max-width:62ch}
.answer p{margin:0 0 .95rem}
.answer h3,.answer h4,.answer h5,.answer h6{font-family:var(--serif);font-weight:400;
  line-height:1.2;letter-spacing:-.02em;margin:1.9rem 0 .6rem}
.answer h3{font-size:1.35rem} .answer h4{font-size:1.1rem}
.answer h5,.answer h6{font-size:.95rem;color:var(--ink-2)}
.answer>:first-child{margin-top:0}
.answer ul{margin:0 0 .95rem;padding-left:1.15rem}
.answer li{margin:0 0 .45rem}
.answer strong{font-family:var(--sans);font-weight:700;font-size:.94em;letter-spacing:-.01em}
button.cite,span.cite{font-family:var(--mono);font-size:.78rem;white-space:nowrap}
button.cite{background:none;border:0;border-bottom:1px solid currentColor;padding:0 .05em;
  cursor:pointer;color:inherit;line-height:inherit}
button.cite.ok{color:var(--accent)}
button.cite.bad{color:var(--bad);border-bottom-style:dotted}
button.cite:hover,button.cite.lit{background:var(--lit)}
button.cite.lit{box-shadow:0 0 0 2px var(--lit)}

/* --- audit trail -------------------------------------------------------- */
.sub{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin:2.5rem 0 .25rem;padding-top:1.25rem;border-top:1px solid var(--rule)}
.sub-note{color:var(--ink-2);font-size:.92rem;margin:0 0 1.25rem}
.strip{display:block;width:100%;height:auto;color:var(--ink)}
.strip .lane{cursor:pointer}
.strip .lane rect.hit{fill:transparent}
.strip .lane:hover rect.hit,.strip .lane.lit rect.hit{fill:var(--lit)}
.strip .lbl{font-family:var(--mono);font-size:10px;fill:var(--ink-2)}
.strip .val{font-family:var(--mono);font-size:10px;fill:var(--ink-3)}
.strip .bar.hi{fill:var(--accent)}
.strip .bar.lo{fill:var(--warn)}
.strip .none{fill:none;stroke:var(--rule-strong);stroke-width:1;stroke-dasharray:2 2}
.strip .gate{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 3}
.strip .axis{stroke:var(--rule);stroke-width:1}
.strip .gatelbl{font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;fill:var(--ink-3)}

.filters{display:flex;flex-wrap:wrap;gap:.4rem;margin:1.5rem 0 1rem}
.route{font-family:var(--mono);font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;
  background:none;border:1px solid var(--rule);color:var(--ink-2);padding:.35rem .6rem;
  cursor:pointer}
.route:hover{border-color:var(--rule-strong)}
.route[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);
  background:var(--accent-soft)}
.trail{list-style:none;margin:0;padding:0;border-top:1px solid var(--rule)}
.ex{border-bottom:1px solid var(--rule);padding:.9rem .8rem;cursor:pointer;
  transition:background 120ms linear}
.ex:hover{background:var(--surface)}
.ex.lit{background:var(--lit);box-shadow:inset 2px 0 0 var(--accent)}
.ex.off{display:none}
.ex-head{display:flex;flex-wrap:wrap;gap:.5rem;align-items:baseline;margin-bottom:.45rem}
.ex-src{font-family:var(--mono);font-size:.76rem;color:var(--ink);letter-spacing:.02em}
.tag{font-family:var(--mono);font-size:.64rem;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ink-3);border:1px solid var(--rule);padding:.12rem .4rem}
.tag.cited{color:var(--accent);border-color:var(--accent)}
.tag.score{color:var(--ink-2)}
.ex-text{margin:0;font-size:.92rem;line-height:1.5;color:var(--ink-2)}
.usage{font-family:var(--mono);font-size:.7rem;color:var(--ink-3);margin:1.25rem 0 0;
  letter-spacing:.04em}

/* --- coverage ----------------------------------------------------------- */
.cov{width:100%;border-collapse:collapse;font-size:.85rem}
.cov th{font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3);font-weight:400;text-align:right;padding:.5rem .6rem;
  border-bottom:1px solid var(--rule-strong)}
.cov th:first-child{text-align:left}
.cov td{padding:.45rem .6rem;border-bottom:1px solid var(--rule);text-align:right;
  font-family:var(--mono);font-size:.8rem}
.cov td:first-child{text-align:left;font-family:var(--serif);font-size:.95rem}
.cov td.hit{color:var(--accent)}
.cov td.miss{color:var(--ink-3)}
.cov tr.dim td:first-child{color:var(--ink-3)}

footer{border-top:1px solid var(--rule);margin-top:1rem;padding:1.5rem 0 3.5rem;
  font-size:.88rem;color:var(--ink-3)}
footer p{margin:0 0 .5rem;max-width:70ch}

@media (max-width:60rem){
  .runs{grid-template-columns:1fr;gap:1.75rem}
  .rail{position:static}
  .diagram{min-width:44rem}
}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *{transition:none!important}
}
</style>
</head>
<body>

<header class="band">
  <div class="inner">
    <p class="label">DAX 40 · __NREPORTS__ reports · FY__YEARS__ · __CHUNKS__ chunks</p>
    <h1>Answers with the citation check attached</h1>
    <p class="standfirst">One question across fifteen annual reports. The interesting part is not
    the prose — it is that every citation in it was parsed back out and checked against the
    excerpts the model was actually handed, before you were allowed to see it.</p>
    <div class="facts">
      <div class="fact"><span class="k">Recorded runs</span><span class="v">__NQUERIES__</span></div>
      <div class="fact"><span class="k">Answered</span><span class="v">__NANSWERED__<small> of __NQUERIES__</small></span></div>
      <div class="fact"><span class="k">Citations verified</span><span class="v">__NVERIFIED__<small>/__NCITES__</small></span></div>
      <div class="fact"><span class="k">Relevance floor</span><span class="v">__MINRELEVANCE__</span></div>
      <div class="fact"><span class="k">Confidence bar</span><span class="v">__LOWCONFIDENCE__</span></div>
    </div>
    <p class="prov">Recorded __RUNDATE__ against the real index · generation by __MODEL__ ·
    USD __TOTALCOST__ once · <a href="https://github.com/morichtereur/dax-intelligence">source</a></p>
  </div>
</header>

<main class="inner">

  <section>
    <div class="plate">
      <p class="label">What this page is</p>
      <p><strong>Recorded runs, not a live system.</strong> Every answer, citation, verdict and
      excerpt below came out of the real pipeline against the real index on __RUNDATE__, using
      <span class="mono">__MODEL__</span>. Nothing is illustrative and nothing was edited.</p>
      <p>It is recorded because the app cannot honestly be deployed: it needs a 373 MB vector index
      and 3.6 GB of source filings that do not belong in a repository, and a public endpoint would
      put an API key behind an unmetered text box. Recording it costs
      <span class="mono">USD __TOTALCOST__</span> once and nothing thereafter.</p>
    </div>
  </section>

  <section id="how">
    <h2>Where the guardrail sits</h2>
    <p class="lede">Two gates and one check. The gates decide whether a question reaches the model
    at all; the check decides whether its answer reaches you as written. Both are ground truth
    against the retrieved excerpts, not a second model's opinion.</p>

    <figure class="exhibit" style="margin:0">
      <figcaption class="cap">Exhibit — one query through the pipeline</figcaption>
      <div class="body">
        <svg class="diagram" viewBox="0 0 1000 392" role="img"
             aria-label="Pipeline diagram: a query is searched by dense retrieval and BM25 in
             parallel, the two result sets are unioned and re-ranked by a cross-encoder to eight
             excerpts, then gated on the best cosine score before the model is called, and every
             citation the model writes is re-parsed and matched back against those eight excerpts.">
          <rect class="box" x="1" y="52" width="118" height="46"/>
          <text class="ttl" x="14" y="72">Query</text>
          <text class="txt" x="14" y="89">one question</text>

          <path class="flow" d="M119 75 H148 M148 46 V104 M148 46 H164 M148 104 H164"/>
          <path class="flow" d="M162 42 l8 4 -8 4 z M162 100 l8 4 -8 4 z" fill="currentColor" stroke="none"/>

          <rect class="box" x="170" y="22" width="220" height="48"/>
          <text class="ttl" x="184" y="42">Dense — Chroma</text>
          <text class="txt" x="184" y="59">cosine over __CHUNKS__ chunks</text>

          <rect class="box" x="170" y="80" width="220" height="48"/>
          <text class="ttl" x="184" y="100">Keyword</text>
          <text class="txt" x="184" y="117">BM25 over the same chunks</text>

          <path class="flow" d="M390 46 H420 M390 104 H420 M420 46 V104 M420 75 H448"/>
          <path class="flow" d="M446 71 l8 4 -8 4 z" fill="currentColor" stroke="none"/>

          <rect class="box" x="454" y="52" width="230" height="46"/>
          <text class="ttl" x="468" y="72">Union, then re-rank</text>
          <text class="txt" x="468" y="89">cross-encoder over both lists</text>

          <path class="flow" d="M684 75 H712"/>
          <path class="flow" d="M710 71 l8 4 -8 4 z" fill="currentColor" stroke="none"/>

          <rect class="box acc" x="718" y="52" width="280" height="46"/>
          <text class="ttl" x="732" y="72">Top 8 excerpts</text>
          <text class="txt" x="732" y="89">each keeping its real page range</text>

          <path class="flow" d="M858 98 V138 H40 V180"/>
          <path class="flow" d="M36 178 l4 8 4 -8 z" fill="currentColor" stroke="none"/>

          <rect class="box acc" x="1" y="188" width="150" height="60"/>
          <text class="ttl" x="14" y="208">Gate</text>
          <text class="txt" x="14" y="226">best cosine</text>
          <text class="txt" x="14" y="242">score of the 8</text>

          <path class="flow dash" d="M151 218 H186 M186 205 V300 M186 205 H212 M186 250 H212 M186 300 H212"/>
          <path class="flow" d="M210 201 l8 4 -8 4 z M210 246 l8 4 -8 4 z M210 296 l8 4 -8 4 z" fill="currentColor" stroke="none"/>

          <rect class="box term" x="220" y="185" width="250" height="40"/>
          <text class="num mute" x="232" y="201">&lt; __MINRELEVANCE__</text>
          <text class="txt" x="232" y="217">never reaches the model</text>

          <rect class="box" x="220" y="230" width="250" height="40"/>
          <text class="num" x="232" y="246">__MINRELEVANCE__ – __LOWCONFIDENCE__</text>
          <text class="txt" x="232" y="262">answered, flagged low confidence</text>

          <rect class="box" x="220" y="280" width="250" height="40"/>
          <text class="num" x="232" y="296">≥ __LOWCONFIDENCE__</text>
          <text class="txt" x="232" y="312">answered</text>

          <path class="flow" d="M470 250 H494 M470 300 H494 M494 250 V300 M494 275 H516"/>
          <path class="flow" d="M514 271 l8 4 -8 4 z" fill="currentColor" stroke="none"/>

          <rect class="box" x="522" y="252" width="208" height="46"/>
          <text class="ttl" x="534" y="272">Model</text>
          <text class="txt" x="534" y="289">cites (Company, Year, p.N)</text>

          <path class="flow" d="M730 275 H758"/>
          <path class="flow" d="M756 271 l8 4 -8 4 z" fill="currentColor" stroke="none"/>

          <rect class="box acc" x="764" y="245" width="235" height="60"/>
          <text class="ttl" x="776" y="265">Citation check</text>
          <text class="txt" x="776" y="283">every citation re-parsed and</text>
          <text class="txt" x="776" y="299">matched to those 8 excerpts</text>

          <path class="flow" d="M881 305 V336"/>
          <path class="flow" d="M877 334 l4 8 4 -8 z" fill="currentColor" stroke="none"/>
          <text class="ttl" x="776" y="358" style="fill:var(--accent)">Verified</text>
          <text class="ttl" x="852" y="358">or</text>
          <text class="ttl" x="880" y="358" style="fill:var(--bad)">flagged</text>
          <text class="txt" x="776" y="378">shown per citation, in the prose</text>
        </svg>
      </div>
    </figure>

    <div class="legend">
      <div><span class="k k-ok">Answered</span><p>The excerpts supported the question. Every
        citation in the answer resolved to one of them.</p></div>
      <div><span class="k k-warn">Declined</span><p>Excerpts came back, but none supported the
        question. The model wrote no citation rather than an unsourced claim.</p></div>
      <div><span class="k k-gate">Gated</span><p>Nothing cleared the relevance floor, so the
        question never reached the model — and cost nothing.</p></div>
    </div>
  </section>

  <section id="runs">
    <h2>Six runs, two of which fail on purpose</h2>
    <p class="lede">Pick a run. Inside it, every citation is a control: click one and the page
    jumps to the excerpt it was checked against. Click an excerpt to see which claims rest on it.</p>
    <div class="runs">
      <div class="rail" id="tabs" role="tablist" aria-label="Recorded runs"></div>
      <div class="panel" id="panel"></div>
    </div>
  </section>

  <section id="coverage">
    <h2>Which reports these runs actually opened</h2>
    <p class="lede">The index holds __NREPORTS__ companies across two fiscal years. Six questions
    do not touch all of them — this is what the retriever reached for, counted in excerpts.</p>
    <div id="cov"></div>
  </section>

  <footer>
    <p>Generated by <span class="mono">scripts/build_dashboard.py</span> —
    <a href="https://github.com/morichtereur/dax-intelligence">source</a>. The runs are cached in
    the repository, so this page rebuilds without the index, the filings or an API key.</p>
    <p>Excerpts are quoted from published annual reports for the purpose of showing retrieval;
    each one names its company, year and page.</p>
  </footer>
</main>

<script id="data" type="application/json">__PAYLOAD__</script>
<script id="meta" type="application/json">__META__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const META = JSON.parse(document.getElementById('meta').textContent);
const tabs = document.getElementById('tabs');
const panel = document.getElementById('panel');
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ROUTE = {embedding:'dense', keyword:'keyword', both:'both'};

const badge = r =>
  r.outcome === 'gated' ? ['k-gate', 'never reached the model']
  : r.outcome === 'declined' ? ['k-warn', 'declined — no source']
  : ['k-ok', r.citations.filter(c => c.verified).length + '/' + r.citations.length + ' citations verified'];

/* --- the rail ----------------------------------------------------------- */
DATA.forEach((r, i) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'tab';
  b.id = 'tab-' + i;
  b.setAttribute('role', 'tab');
  b.setAttribute('aria-controls', 'panel');
  b.setAttribute('aria-selected', 'false');
  b.tabIndex = -1;
  const [cls, label] = badge(r);
  b.innerHTML = '<span class="n">Run ' + String(i + 1).padStart(2, '0') + '</span>'
    + esc(r.query)
    + '<span class="badge ' + cls + '">' + label + '</span>';
  b.onclick = () => show(i);
  tabs.appendChild(b);
});

tabs.addEventListener('keydown', e => {
  const keys = {ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1};
  let next = null;
  if (keys[e.key]) next = (current + keys[e.key] + DATA.length) % DATA.length;
  else if (e.key === 'Home') next = 0;
  else if (e.key === 'End') next = DATA.length - 1;
  if (next === null) return;
  e.preventDefault();
  show(next);
  tabs.children[next].focus();
});

/* --- the retrieval strip ------------------------------------------------ */
function strip(r) {
  const W = 620, gutter = 132, right = 30, lane = 22, head = 30, foot = 34;
  const SCALE = 0.7;
  const x = v => gutter + (v / SCALE) * (W - gutter - right);
  const H = head + r.chunks.length * lane + foot;
  const bottom = head + r.chunks.length * lane;
  let s = '';

  [[META.minRelevance, 'floor'], [META.lowConfidence, 'confidence']].forEach(([v, name]) => {
    s += '<line class="gate" x1="' + x(v) + '" y1="' + (head - 10) + '" x2="' + x(v) + '" y2="' + bottom + '"/>'
      + '<text class="gatelbl" x="' + x(v) + '" y="' + (head - 16) + '" text-anchor="middle">'
      + name.toUpperCase() + ' ' + v.toFixed(2) + '</text>';
  });

  r.chunks.forEach((c, i) => {
    const y = head + i * lane;
    const mid = y + lane / 2;
    const cited = r.citations ? r.citations.filter(z => z.chunk === i).length : 0;
    s += '<g class="lane" data-chunk="' + i + '">'
      + '<title>' + esc(c.name + ' ' + c.year + ', p.' + c.page)
      + (c.score === null ? ' — BM25 hit, no cosine score' : ' — cosine ' + c.score.toFixed(3))
      + (cited ? '. Cited ' + cited + '× in the answer.' : '. Not cited.') + '</title>'
      + '<rect class="hit" x="0" y="' + y + '" width="' + W + '" height="' + lane + '"/>'
      + '<text class="lbl" x="' + (gutter - 10) + '" y="' + (mid + 3.5) + '" text-anchor="end">'
      + esc(c.name + ' ' + c.year) + '</text>';
    if (c.score === null) {
      s += '<rect class="none" x="' + gutter + '" y="' + (mid - 5) + '" width="10" height="10"/>'
        + '<text class="val" x="' + (gutter + 18) + '" y="' + (mid + 3.5) + '">BM25 hit · no cosine score'
        + (cited ? ' · cited' : '') + '</text>';
    } else {
      const w = Math.max(1, x(c.score) - gutter);
      s += '<rect class="bar ' + (c.score >= META.lowConfidence ? 'hi' : 'lo') + '" x="' + gutter
        + '" y="' + (mid - 5) + '" width="' + w + '" height="10"/>'
        + '<text class="val" x="' + (gutter + w + 6) + '" y="' + (mid + 3.5) + '">'
        + c.score.toFixed(3) + (cited ? ' · cited' : '') + '</text>';
    }
    s += '</g>';
  });

  s += '<line class="axis" x1="' + gutter + '" y1="' + (bottom + 8) + '" x2="' + x(SCALE) + '" y2="' + (bottom + 8) + '"/>';
  [0, 0.2, 0.4, 0.6].forEach(v => {
    s += '<line class="axis" x1="' + x(v) + '" y1="' + (bottom + 8) + '" x2="' + x(v) + '" y2="' + (bottom + 12) + '"/>'
      + '<text class="val" x="' + x(v) + '" y="' + (bottom + 26) + '" text-anchor="middle">' + v.toFixed(1) + '</text>';
  });
  s += '<text class="val" x="' + (gutter - 26) + '" y="' + (bottom + 26) + '" text-anchor="end">cosine similarity</text>';

  return '<svg class="strip" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="'
    + 'Retrieval scores for each of the ' + r.chunks.length + ' excerpts, against the '
    + META.minRelevance.toFixed(2) + ' relevance floor and the ' + META.lowConfidence.toFixed(2)
    + ' confidence bar.">' + s + '</svg>';
}

/* --- the audit trail ---------------------------------------------------- */
function trail(r) {
  const counts = {all: r.chunks.length, embedding: 0, keyword: 0, both: 0};
  r.chunks.forEach(c => counts[c.via]++);
  const chips = [['all', 'all excerpts']].concat(
    Object.keys(ROUTE).filter(k => counts[k]).map(k => [k, ROUTE[k]])
  ).map(([k, label], i) =>
    '<button type="button" class="route" data-route="' + k + '" aria-pressed="'
    + (i === 0 ? 'true' : 'false') + '">' + label + ' · ' + counts[k] + '</button>'
  ).join('');

  const items = r.chunks.map((c, i) => {
    const cited = r.citations ? r.citations.filter(z => z.chunk === i).length : 0;
    const pages = 'p.' + c.page + (c.end_page && c.end_page !== c.page ? '–' + c.end_page : '');
    return '<li class="ex" id="ex-' + i + '" data-chunk="' + i + '" data-via="' + c.via + '">'
      + '<div class="ex-head">'
      + '<span class="ex-src">' + esc(c.name + ' ' + c.year) + ' · ' + pages + '</span>'
      + '<span class="tag">' + esc((c.section || 'unsectioned').replace(/_/g, ' ')) + '</span>'
      + '<span class="tag score">' + ROUTE[c.via] + (c.score === null ? '' : ' ' + c.score.toFixed(3)) + '</span>'
      + (cited ? '<span class="tag cited">cited ' + cited + '×</span>' : '')
      + '</div><p class="ex-text">' + esc(c.text) + '</p></li>';
  }).join('');

  const citedChunks = new Set((r.citations || []).map(c => c.chunk).filter(c => c !== null));
  return '<p class="sub">Audit trail — the excerpts the model was given</p>'
    + '<p class="sub-note">' + r.chunks.length + ' excerpts retrieved, ' + citedChunks.size
    + ' of them cited. Every bar below is one excerpt; the two dashed lines are the gates.</p>'
    + strip(r)
    + '<div class="filters">' + chips + '</div>'
    + '<ul class="trail">' + items + '</ul>';
}

/* --- a run -------------------------------------------------------------- */
let current = -1;

function show(i) {
  if (i === current) return;
  current = i;
  const r = DATA[i];
  [...tabs.children].forEach((t, j) => {
    t.setAttribute('aria-selected', j === i ? 'true' : 'false');
    t.tabIndex = j === i ? 0 : -1;
  });
  history.replaceState(null, '', '#run-' + (i + 1));
  panel.dataset.chunk = '';

  let head = '<h3 class="q">' + esc(r.query) + '</h3><p class="note">' + esc(r.note) + '</p>';

  if (r.outcome === 'gated') {
    panel.innerHTML = head
      + '<div class="verdict"><span class="stamp s-gate">Gated — no model call</span>'
      + '<span class="meta">0 tokens · USD 0.0000</span></div>'
      + '<div class="banner gate"><strong>No excerpt cleared the relevance floor of '
      + META.minRelevance.toFixed(2) + '.</strong> The question never reached the model, so there '
      + 'is no answer to check — which is the correct outcome, and cheaper than a fluent one '
      + 'about nothing.</div>'
      + '<p class="sub">Audit trail — the excerpts the model was given</p>'
      + '<p class="sub-note">None. The retriever returned nothing above the floor, and the '
      + 'pipeline stopped there.</p>';
    return;
  }

  const verified = r.citations.filter(c => c.verified).length;
  const declined = r.outcome === 'declined';
  head += '<div class="verdict">'
    + (declined
        ? '<span class="stamp s-warn">Declined — no citation written</span>'
        : '<span class="stamp ' + (verified === r.citations.length ? 's-ok' : 's-warn') + '">'
          + (verified === r.citations.length ? 'Verified' : 'Partial') + ' — ' + verified + '/'
          + r.citations.length + ' citations matched retrieved excerpts</span>')
    + '<span class="meta">best score ' + r.best_score.toFixed(3) + ' · '
    + r.usage.input_tokens.toLocaleString() + ' in · ' + r.usage.output_tokens.toLocaleString()
    + ' out · USD ' + r.cost.toFixed(4) + '</span></div>';

  if (r.low_confidence) {
    head += '<div class="banner"><strong>Low confidence.</strong> The best excerpt scored '
      + r.best_score.toFixed(3) + ', below the ' + META.lowConfidence.toFixed(2) + ' bar. The '
      + 'answer is shown with this warning rather than presented as solid ground — read the '
      + 'audit trail before using it.</div>';
  }
  if (declined) {
    head += '<div class="banner"><strong>The model declined.</strong> It was handed '
      + r.chunks.length + ' excerpts and wrote none of them into a citation, because none of them '
      + 'answers the question. The prompt forbids an unsourced claim, so a refusal is the '
      + 'guardrail working rather than failing.</div>';
  }

  panel.innerHTML = head + '<div class="answer">' + r.answer_html + '</div>' + trail(r);
}

/* --- linking citations to excerpts -------------------------------------- */
function light(chunk) {
  const on = panel.dataset.chunk !== String(chunk);
  panel.querySelectorAll('.ex.lit, .lane.lit, button.cite.lit')
       .forEach(el => el.classList.remove('lit'));
  panel.dataset.chunk = on ? String(chunk) : '';
  if (!on) return;
  const row = panel.querySelector('.ex[data-chunk="' + chunk + '"]');
  const lane = panel.querySelector('.lane[data-chunk="' + chunk + '"]');
  if (row) {
    row.classList.remove('off');
    row.classList.add('lit');
    row.scrollIntoView({block: 'center', behavior: reduced ? 'auto' : 'smooth'});
  }
  if (lane) lane.classList.add('lit');
  const r = DATA[current];
  (r.citations || []).forEach((c, i) => {
    if (c.chunk !== chunk) return;
    panel.querySelectorAll('button.cite[data-cite="' + i + '"]').forEach(b => b.classList.add('lit'));
  });
}

panel.addEventListener('click', e => {
  const chip = e.target.closest('.route');
  if (chip) {
    const route = chip.dataset.route;
    panel.querySelectorAll('.route').forEach(c =>
      c.setAttribute('aria-pressed', c === chip ? 'true' : 'false'));
    panel.querySelectorAll('.ex').forEach(row =>
      row.classList.toggle('off', route !== 'all' && row.dataset.via !== route));
    return;
  }
  const cite = e.target.closest('button.cite');
  if (cite) {
    const c = DATA[current].citations[+cite.dataset.cite];
    if (c && c.chunk !== null && c.chunk !== undefined) light(c.chunk);
    return;
  }
  const row = e.target.closest('.ex, .lane');
  if (row) light(+row.dataset.chunk);
});

/* --- coverage ----------------------------------------------------------- */
(function coverage() {
  const years = META.years;
  const rows = META.coverage.map(c => {
    const total = c.years.reduce((a, b) => a + b, 0);
    return '<tr' + (total ? '' : ' class="dim"') + '><td>' + esc(c.name) + '</td>'
      + c.years.map(n => '<td class="' + (n ? 'hit' : 'miss') + '">' + (n || '·') + '</td>').join('')
      + '<td class="' + (total ? 'hit' : 'miss') + '">' + (total || '·') + '</td></tr>';
  }).join('');
  document.getElementById('cov').innerHTML =
    '<table class="cov"><thead><tr><th>Report</th>'
    + years.map(y => '<th>FY' + y + '</th>').join('')
    + '<th>Excerpts</th></tr></thead><tbody>' + rows + '</tbody></table>';
})();

const fromHash = parseInt((location.hash.match(/^#run-(\\d+)$/) || [])[1], 10);
show(fromHash >= 1 && fromHash <= DATA.length ? fromHash - 1 : 0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build(from_cache="--from-cache" in sys.argv)
