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


def render_answer(text: str, citations: list[dict]) -> str:
    """The model writes markdown. Render the subset it actually uses —
    headings, bullets, bold — and wrap every citation in its verdict, which is
    the treatment app.py gives it, so this page shows what the app shows."""
    verdict = {(c["company"].lower(), c["page"]): c["verified"] for c in citations}

    def cite(m: re.Match) -> str:
        company, page = m.group(1).strip(), int(m.group(3))
        ok = verdict.get((company.lower(), page))
        cls = "cite ok" if ok else "cite bad" if ok is False else "cite"
        return f'<span class="{cls}">{html.escape(m.group(0))}</span>'

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
            # The model's own h1 is the answer's title, which sits under this
            # page's h1 -- so everything shifts down two levels.
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
        r["answer_html"] = (
            render_answer(r["answer"], r["citations"]) if r["outcome"] != "gated" else ""
        )

    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")

    fonts = "".join(
        [
            font_face("Archivo", "archivo-400.woff2", 400),
            font_face("Archivo", "archivo-700.woff2", 700),
            font_face("Source Serif 4", "source-serif-400.woff2", 400),
            font_face("IBM Plex Mono", "plex-mono-400.woff2", 400),
        ]
    )

    html_out = TEMPLATE.format(
        fonts=fonts,
        payload=payload,
        n_reports=len(get_companies()),
        years="–".join([get_years()[0], get_years()[-1]]),
        chunks=f"{collection.count():,}",
        model=MODEL,
        min_relevance=f"{MIN_RELEVANCE:.2f}",
        low_confidence=f"{LOW_CONFIDENCE:.2f}",
        run_date=records[0].get("run_date", date.today().isoformat()),
        n_queries=len(records),
        n_answered=len(answered),
        n_cites=len(all_cites),
        n_verified=verified,
        total_cost=f"{total_cost:.3f}",
    )

    out = ROOT / "dashboard.html"
    out.write_text(html_out, encoding="utf-8")
    size = out.stat().st_size / 1024
    print(f"\nWrote {out.name} ({size:.0f} KB)")
    print(f"  {verified}/{len(all_cites)} citations verified across {len(answered)} answers")
    print(f"  total generation cost USD {total_cost:.3f}")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Answers with the citation check attached</title>
<style>
{fonts}
:root {{
  --sans:'Archivo',system-ui,-apple-system,sans-serif;
  --serif:'Source Serif 4',Georgia,serif;
  --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
  --paper:#fbfbfa; --panel:#f2f2f0; --ink:#16181a; --ink-2:#4c5054; --ink-3:#7b8085;
  --rule:#d8d8d4; --rule-strong:#b4b5b0; --accent:#146b54; --accent-soft:rgba(20,107,84,.10);
  --warn:#8a5a12; --warn-soft:rgba(138,90,18,.10); --bad:#9a2f2f;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#131517; --panel:#1b1e21; --ink:#eceeef; --ink-2:#b0b5b9; --ink-3:#838a8f;
    --rule:#2c3033; --rule-strong:#454b50; --accent:#4bbf9a; --accent-soft:rgba(75,191,154,.12);
    --warn:#d9a441; --warn-soft:rgba(217,164,65,.12); --bad:#e0736b;
  }}
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-size:17px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:60rem;margin:0 auto;padding:3rem 1.5rem 5rem}}
.eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 1rem}}
h1{{font-family:var(--sans);font-weight:700;font-size:clamp(1.9rem,4.2vw,2.9rem);line-height:1.08;
  letter-spacing:-.02em;margin:0 0 1rem}}
.standfirst{{color:var(--ink-2);margin:0 0 2rem;max-width:46rem}}
.callout{{border:1px solid var(--rule-strong);padding:1.25rem 1.5rem;margin:0 0 2.5rem}}
.callout .label{{font-family:var(--mono);font-size:.72rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 .6rem}}
.callout p{{margin:.6rem 0 0}} .callout p:first-of-type{{margin-top:0}}
.callout strong{{font-family:var(--sans);font-weight:700}}
.facts{{display:flex;flex-wrap:wrap;gap:0;border:1px solid var(--rule);margin:0 0 2.5rem}}
.fact{{flex:1 1 8rem;padding:.85rem 1rem;border-right:1px solid var(--rule)}}
.fact:last-child{{border-right:0}}
.fact .k{{font-family:var(--mono);font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);display:block;margin-bottom:.25rem}}
.fact .v{{font-family:var(--mono);font-size:1.05rem;color:var(--ink)}}
.tabs{{display:flex;flex-direction:column;gap:.4rem;margin:0 0 2rem}}
.tab{{text-align:left;background:none;border:1px solid var(--rule);color:var(--ink);
  font-family:var(--serif);font-size:.95rem;padding:.7rem .9rem;cursor:pointer;
  display:flex;gap:.75rem;align-items:baseline}}
.tab:hover{{border-color:var(--rule-strong)}}
.tab[aria-selected="true"]{{border-color:var(--accent);background:var(--accent-soft)}}
.tab .n{{font-family:var(--mono);font-size:.72rem;color:var(--ink-3);flex:none}}
.tab .badge{{margin-left:auto;font-family:var(--mono);font-size:.66rem;letter-spacing:.06em;
  text-transform:uppercase;flex:none;padding-left:.75rem}}
.b-ok{{color:var(--accent)}} .b-warn{{color:var(--warn)}} .b-gate{{color:var(--ink-3)}}
.panel{{border-top:1px solid var(--rule-strong);padding-top:1.5rem}}
.q{{font-family:var(--sans);font-weight:700;font-size:1.3rem;line-height:1.25;margin:0 0 .5rem}}
.note{{color:var(--ink-3);font-size:.92rem;margin:0 0 1.25rem}}
.stamp{{display:inline-block;font-family:var(--mono);font-size:.72rem;letter-spacing:.06em;
  padding:.3rem .6rem;margin:0 0 1rem}}
.s-ok{{background:var(--accent-soft);color:var(--accent)}}
.s-warn{{background:var(--warn-soft);color:var(--warn)}}
.banner{{border-left:3px solid var(--warn);background:var(--warn-soft);padding:.85rem 1rem;
  margin:0 0 1.25rem;font-size:.94rem}}
.banner.gate{{border-left-color:var(--ink-3);background:var(--panel);color:var(--ink-2)}}
.answer p{{margin:0 0 .9rem}}
.answer h3,.answer h4,.answer h5,.answer h6{{font-family:var(--sans);line-height:1.25;
  letter-spacing:-.01em;margin:1.7rem 0 .6rem}}
.answer h3{{font-size:1.2rem}} .answer h4{{font-size:1.02rem}}
.answer h5,.answer h6{{font-size:.92rem;color:var(--ink-2)}}
.answer>:first-child{{margin-top:0}}
.answer ul{{margin:0 0 .9rem;padding-left:1.15rem}}
.answer li{{margin:0 0 .4rem}}
.cite{{font-family:var(--mono);font-size:.82rem;white-space:nowrap}}
.cite.ok{{color:var(--accent)}}
.cite.bad{{color:var(--bad);text-decoration:underline wavy}}
.sub{{font-family:var(--mono);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin:2rem 0 .75rem;padding-top:1rem;border-top:1px solid var(--rule)}}
table{{width:100%;border-collapse:collapse;font-size:.86rem}}
th{{font-family:var(--mono);font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);text-align:left;font-weight:400;padding:.4rem .5rem;
  border-bottom:1px solid var(--rule-strong)}}
td{{padding:.55rem .5rem;border-bottom:1px solid var(--rule);vertical-align:top}}
td.num{{font-family:var(--mono);text-align:right;white-space:nowrap}}
td.src{{font-family:var(--mono);font-size:.78rem;white-space:nowrap}}
td.ex{{color:var(--ink-2);font-size:.85rem;line-height:1.45}}
.scroll{{overflow-x:auto}}
.usage{{font-family:var(--mono);font-size:.75rem;color:var(--ink-3);margin-top:1rem}}
footer{{margin-top:3.5rem;padding-top:1.25rem;border-top:1px solid var(--rule);
  font-size:.85rem;color:var(--ink-3)}}
footer a{{color:var(--accent)}}
</style>
</head>
<body>
<div class="wrap">
  <p class="eyebrow">DAX 40 · {n_reports} reports · FY{years} · {chunks} chunks</p>
  <h1>Answers with the citation check attached</h1>
  <p class="standfirst">One question across fifteen annual reports. The interesting part is not
  the prose — it is that every citation in it was parsed back out and checked against the
  excerpts the model was actually handed, before you were allowed to see it.</p>

  <div class="callout">
    <p class="label">What this page is</p>
    <p><strong>Recorded runs, not a live system.</strong> Every answer, citation, verdict and
    excerpt below came out of the real pipeline against the real index on {run_date}, using
    <span class="cite">{model}</span>. Nothing is illustrative and nothing was edited.</p>
    <p>It is recorded because the app cannot honestly be deployed: it needs a 373 MB vector index
    and 3.6 GB of source filings that do not belong in a repository, and a public endpoint would
    put an API key behind an unmetered text box. Recording it costs
    <span class="cite">USD {total_cost}</span> once and nothing thereafter.</p>
  </div>

  <div class="facts">
    <div class="fact"><span class="k">Queries</span><span class="v">{n_queries}</span></div>
    <div class="fact"><span class="k">Answered</span><span class="v">{n_answered}</span></div>
    <div class="fact"><span class="k">Citations</span><span class="v">{n_verified}/{n_cites}</span></div>
    <div class="fact"><span class="k">Relevance floor</span><span class="v">{min_relevance}</span></div>
    <div class="fact"><span class="k">Confidence bar</span><span class="v">{low_confidence}</span></div>
  </div>

  <div class="tabs" id="tabs" role="tablist"></div>
  <div class="panel" id="panel"></div>

  <footer>
    Generated by <span class="cite">scripts/build_dashboard.py</span> —
    <a href="https://github.com/morichtereur/dax-intelligence">source</a>.
    Excerpts are quoted from published annual reports for the purpose of showing retrieval;
    each one names its company, year and page.
  </footer>
</div>

<script id="data" type="application/json">{payload}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const BAR = '{low_confidence}';
const tabs = document.getElementById('tabs');
const panel = document.getElementById('panel');

const badge = r =>
  r.outcome === 'gated' ? ['b-gate', 'never reached the model']
  : r.outcome === 'declined' ? ['b-warn', 'declined, no source']
  : ['b-ok', (r.citations.filter(c => c.verified).length) + '/' + r.citations.length + ' verified'];

DATA.forEach((r, i) => {{
  const b = document.createElement('button');
  b.className = 'tab';
  b.setAttribute('role', 'tab');
  b.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
  const [cls, label] = badge(r);
  b.innerHTML = '<span class="n">' + String(i + 1).padStart(2, '0') + '</span>'
    + '<span>' + r.query + '</span>'
    + '<span class="badge ' + cls + '">' + label + '</span>';
  b.onclick = () => show(i);
  tabs.appendChild(b);
}});

function esc(s) {{
  return String(s).replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
}}

function show(i) {{
  const r = DATA[i];
  [...tabs.children].forEach((t, j) => t.setAttribute('aria-selected', j === i ? 'true' : 'false'));

  let head = '<p class="q">' + esc(r.query) + '</p><p class="note">' + esc(r.note) + '</p>';

  if (r.outcome === 'gated') {{
    head += '<div class="banner gate"><strong>No excerpt cleared the relevance floor.</strong> '
      + 'The question never reached the model, so there is no answer to check — which is the '
      + 'correct outcome, and cheaper than a fluent one about nothing.</div>';
    panel.innerHTML = head;
    return;
  }}

  const verified = r.citations.filter(c => c.verified).length;
  const stampCls = verified === r.citations.length ? 's-ok' : 's-warn';
  head += '<span class="stamp ' + stampCls + '">' + (verified === r.citations.length ? 'VERIFIED' : 'PARTIAL')
    + ' — ' + verified + '/' + r.citations.length + ' citations matched retrieved excerpts</span>';

  if (r.outcome === 'low_confidence') {{
    head += '<div class="banner"><strong>Low confidence.</strong> The best excerpt scored '
      + r.best_score.toFixed(2) + ' similarity, below the bar. The answer is shown with this '
      + 'warning rather than presented as solid ground — read the audit trail before using it.</div>';
  }}

  const rows = r.chunks.map(c =>
    '<tr><td class="src">' + esc(c.company) + ' ' + esc(c.year) + '</td>'
    + '<td>' + esc(c.section || '—') + '</td>'
    + '<td class="num">p.' + c.page + (c.end_page && c.end_page !== c.page ? '–' + c.end_page : '') + '</td>'
    + '<td class="num">' + (c.score === null ? '—' : c.score.toFixed(3)) + '</td>'
    + '<td class="src">' + esc(c.via) + '</td>'
    + '<td class="ex">' + esc(c.text) + '</td></tr>').join('');

  panel.innerHTML = head
    + '<div class="answer">' + r.answer_html + '</div>'
    + '<p class="sub">Audit trail — the excerpts the model was given</p>'
    + '<div class="scroll"><table><thead><tr><th>Source</th><th>Section</th><th>Pages</th>'
    + '<th>Score</th><th>Via</th><th>Excerpt</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
    + '<p class="usage">' + r.usage.input_tokens.toLocaleString() + ' in · '
    + r.usage.output_tokens.toLocaleString() + ' out · USD ' + r.cost.toFixed(4) + '</p>';
}}

show(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build(from_cache="--from-cache" in sys.argv)
