import os
import anthropic
from dotenv import load_dotenv

from app.documents import load_company_documents

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a senior finance and strategy analyst reviewing DAX 40 annual reports.
Your role is to extract and synthesize insights with the precision of a top-tier management consultant.

When answering:
- Lead with the direct answer, then support with evidence from the attached reports
- Flag meaningful differences between companies when relevant
- Use finance and strategy terminology appropriately
- If the attached reports don't contain enough information, say so clearly — do not fabricate

Citations are attached automatically from the source PDFs — you do not need
to write page numbers yourself; just make claims that are directly
supported by the attached documents.

Format: concise paragraphs. Use bullet points only for direct comparisons or lists of findings."""

def ask(query: str, companies: list[str], compare_mode: bool = False) -> dict:
    """Returns {"text": answer prose, "citations": [...]}.

    Citations come from the Claude API's own PDF citations feature
    (citations: enabled on each document block) rather than being written
    into the prose by the model from memory -- each citation's page range
    is extracted by the API directly from the attached PDF. Two corrections
    are applied before returning it, both confirmed against a live test call
    rather than assumed from the docs:

    1. end_page_number is exclusive (a citation entirely on page 2 comes
       back as start=2, end=3 -- a half-open range, like a Python slice),
       not the inclusive last page. Subtract 1 for a real inclusive range.
    2. Page numbers are local to the attached (trimmed) PDF, not the
       original report -- add the file's own page offset, same as the
       ChromaDB path in pipeline/ingest.py, via the same stamped metadata.
    """
    document_blocks, page_offsets = load_company_documents(companies)
    if not document_blocks:
        return {"text": "No matching reports found for this query.", "citations": []}

    instruction = f"Question: {query}"
    if compare_mode:
        instruction += ("\n\nStructure the answer as a cross-company comparison: "
                         "highlight similarities, differences, and any notable outliers.")

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [*document_blocks, {"type": "text", "text": instruction}],
        }],
    )

    text_parts, citations = [], []
    for block in response.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        for c in (block.citations or []):
            if c.type != "page_location":
                continue
            offset = page_offsets.get(c.document_title, 1)
            citations.append({
                "document_title": c.document_title,
                "cited_text": c.cited_text,
                "start_page": c.start_page_number + offset - 1,
                "end_page": (c.end_page_number - 1) + offset - 1,  # -1: exclusive -> inclusive
            })

    return {"text": "".join(text_parts), "citations": citations}
