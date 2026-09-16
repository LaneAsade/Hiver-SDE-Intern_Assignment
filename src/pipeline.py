"""
pipeline.py

End-to-end: classify -> retrieve -> draft -> decide.

Model choice (cost-tiered on purpose, overridable via env vars since
Google's Gemini lineup moves fast -- gemini-2.5-flash-lite got blocked
for new API keys mid-build here, ahead of the whole 2.5 series' announced
retirement). Classification defaults to llama3, the
cheapest currently-active text model with no deprecation notice as of
Sep 2026 ($0.25/$1.50 per 1M tokens). Drafting defaults to
llama3 ($0.75/$3.75, promotional through Dec 2026 -- reverts to
$1.50/$7.50 after), since generation quality matters more there.

If either 404s on your account, override with HIVER_CLASSIFY_MODEL /
HIVER_DRAFT_MODEL -- no code edit needed. If Google's own error message
names a specific replacement model for your account, trust that over
these defaults; it knows your account's access, this doesn't.

Run with --mock for a dependency-free dry run (no key, no network, no
quota) that exercises the full wiring using canned responses -- useful for
verifying the pipeline shape and for the README's <15-minute repro check.
"""
import argparse
import os

import pandas as pd

from src.classify.classifier import classify
from src.escalate.decide import decide
from src.generate.draft import draft_reply
from src.llm_client import GeminiClient, OpenAIClient, OllamaClient, MockClient
from src.retrieve.index import RetrievalIndex

CLASSIFY_MODEL = os.environ.get("HIVER_CLASSIFY_MODEL", "llama3")
DRAFT_MODEL = os.environ.get("HIVER_DRAFT_MODEL", "llama3")


def build_index(pairs_path: str = "data/amazonhelp_pairs.csv") -> RetrievalIndex:
    pairs = pd.read_csv(pairs_path)
    return RetrievalIndex(backend="auto").fit(pairs)


def run(message: str, index: RetrievalIndex, classify_client, draft_client) -> dict:
    c = classify(message, classify_client)
    d = draft_reply(message, c.intent, index, draft_client)
    decision = decide(message, c.intent, c.confidence, d.top_similarity)
    return {
        "message": message,
        "intent": c.intent,
        "confidence": c.confidence,
        "reply": d.reply,
        "top_similarity": d.top_similarity,
        "action": decision.action,
        "reason": decision.reason,
    }


def _mock_responder(prompt: str) -> str:
    # branches on which prompt it received -- classify prompts ask for the
    # {"intent": ...} JSON shape, draft prompts don't
    if '"intent"' in prompt:
        return '{"intent": "delivery_delay", "confidence": 0.83}'
    return "Sorry for the wait! Can you share your order ID so we can check what's holding up delivery?"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="dry run, no API key/network/quota needed")
    p.add_argument("--message", default="i still have not received my package after waiting the rare 36 hours.")
    p.add_argument("--rpm", type=int, default=10, help="requests/minute cap for live Gemini calls")
    p.add_argument("--max-calls", type=int, default=None, help="hard cap on live calls this run (per client)")
    args = p.parse_args()

    index = build_index()

    if args.mock:
        classify_client = MockClient(responder=_mock_responder)
        draft_client = MockClient(responder=_mock_responder)
    else:
        classify_client = OllamaClient(CLASSIFY_MODEL, requests_per_minute=args.rpm, max_calls=args.max_calls)
        draft_client = OllamaClient(DRAFT_MODEL, requests_per_minute=args.rpm, max_calls=args.max_calls)

    result = run(args.message, index, classify_client, draft_client)
    for k, v in result.items():
        print(f"{k}: {v}")

    if not args.mock:
        print(f"\n[classify usage] {classify_client.usage.summary()}")
        print(f"[draft usage]    {draft_client.usage.summary()}")
