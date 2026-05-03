import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a senior finance and strategy analyst reviewing DAX 40 annual reports.
Your role is to extract and synthesize insights with the precision of a top-tier management consultant.

When answering:
- Lead with the direct answer, then support with evidence from the reports
- Cite company and approximate page number for key claims (e.g. "Siemens, p.45")
- Flag meaningful differences between companies when relevant
- Use finance and strategy terminology appropriately
- If the context doesn't contain enough information, say so clearly — do not fabricate

Format: concise paragraphs. Use bullet points only for direct comparisons or lists of findings."""

def ask(query: str, chunks: list[dict], compare_mode: bool = False) -> str:
    context = ""
    for c in chunks:
        context += f"\n---\n[{c['company']} {c['year']} | {c['section']} | p.~{c['approx_page']}]\n{c['text']}\n"

    if compare_mode:
        user_prompt = f"""Compare across companies based on the following context.

Question: {query}

Context:
{context}

Highlight similarities, differences, and any notable outliers."""
    else:
        user_prompt = f"""Answer the following question based solely on the provided context.

Question: {query}

Context:
{context}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text