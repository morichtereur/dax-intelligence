"""Model access behind one interface, so the provider is configuration.

The generation path here needs exactly one capability: a system prompt, a
user prompt, and text plus token counts back. That is the common
denominator across the Anthropic API, Bedrock, Vertex and OpenAI-compatible
endpoints, so it is the whole contract — anything wider would abstract over
differences this project does not actually use.

What is NOT abstracted, stated so the claim stays honest:

- `eval/eval_grounding.py`'s faithfulness judge uses tool calling with a
  provider-specific schema. Tool calling exists everywhere, but the schema
  does not port unchanged, so a second provider needs an adapter there.
- Token accounting assumes an `input_tokens`/`output_tokens` shape. Every
  major provider reports both; the field names differ.

Swapping the generation model to another provider means adding a class here
and setting `LLM_PROVIDER` — not touching `app/llm.py`. That is the claim,
and it is bounded to the generation path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    usage: dict  # {"input_tokens": int, "output_tokens": int}


class Provider(Protocol):
    """The one capability this project asks of a model."""

    name: str

    def complete(self, *, system: str, user: str, model: str, max_tokens: int) -> Completion:
        ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        import anthropic  # imported here so the module loads without the SDK

        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, *, system: str, user: str, model: str, max_tokens: int) -> Completion:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return Completion(
            text=response.content[0].text,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )


PROVIDERS: dict[str, type] = {
    "anthropic": AnthropicProvider,
}


def get_provider(name: str | None = None) -> Provider:
    """Resolve the configured provider. `LLM_PROVIDER` selects; unknown names
    fail loudly rather than silently falling back to a default the caller did
    not ask for."""
    resolved = (name or os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
    if resolved not in PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER {resolved!r}. Available: {sorted(PROVIDERS)}. "
            "Adding one means implementing Provider.complete() in app/provider.py."
        )
    return PROVIDERS[resolved]()
