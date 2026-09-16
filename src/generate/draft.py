"""
draft.py

Reply drafting: retrieves grounding examples from the RetrievalIndex and
asks the LLM to draft a reply using only what those examples support.
Prompt construction is unit-testable without a live API key; draft_reply()
itself needs one (or a MockClient).
"""
from dataclasses import dataclass

from src.llm_client import LLMClient
from src.retrieve.index import RetrievalIndex

BRAND = "AmazonHelp"


def build_prompt(message: str, intent: str, grounding) -> str:
    examples_block = "\n\n".join(
        f'Past customer message: "{r.customer_text}"\nActual {BRAND} reply: "{r.company_text}"'
        for r in grounding.itertuples()
    )
    return f"""You are drafting a Twitter customer-support reply for {BRAND}.

Customer's message (intent: {intent}): "{message}"

Here is how {BRAND} has actually replied to similar past messages. Match
this tone, and use these as your ONLY source of policy/process facts. Do
not state any specific fact (dates, amounts, policies) that isn't
supported by these examples. If none of them actually cover what this
customer needs, say so plainly instead of guessing.

{examples_block}

Write ONLY the reply text, under 280 characters, no hashtags."""


@dataclass
class Draft:
    reply: str
    grounding_ids: list
    top_similarity: float


def draft_reply(message: str, intent: str, index: RetrievalIndex, client: LLMClient, k: int = 3) -> Draft:
    grounding = index.query(message, k=k)
    prompt = build_prompt(message, intent, grounding)
    reply = client.complete(prompt, max_tokens=400).strip()
    top_sim = float(grounding["similarity"].max()) if len(grounding) else 0.0
    return Draft(
        reply=reply,
        grounding_ids=grounding["company_tweet_id"].tolist() if len(grounding) else [],
        top_similarity=top_sim,
    )
