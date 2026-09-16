import os
"""
04_run_agent.py

Runs the real LLM agent over the golden-set candidates and writes
eval/agent_predictions.csv, keyed by golden_id (same shape as
eval/baseline_predictions.csv so scoring can just join them).

This is the only script that spends API credits. It is built to be
interrupted and resumed: every LLM call goes through the disk cache in
llm_client.py, so re-running after a crash, a rate-limit stall, or a
--limit bump re-reads cached rows for free and only pays for new ones.

Cost shape (203 candidates, 2 calls each = ~406 calls):
  classification  ~700 input + ~30 output tokens  @ flash-lite rates
  drafting        ~900 input + ~80 output tokens  @ flash rates
Order of a few cents total at current published rates. The --limit flag
exists so you can sanity-check 5 rows before committing to all 203.

Usage:
    export GEMINI_API_KEY=...
    PYTHONPATH=. python3 notebooks/04_run_agent.py --limit 5     # smoke test
    PYTHONPATH=. python3 notebooks/04_run_agent.py               # full run
"""
import argparse

import pandas as pd

from src.classify.classifier import classify
from src.escalate.decide import decide
from src.generate.draft import draft_reply
from src.llm_client import GeminiClient, OpenAIClient, OllamaClient, MockClient
from src.pipeline import CLASSIFY_MODEL, DRAFT_MODEL
from src.retrieve.index import RetrievalIndex

OUT_PATH = "eval/agent_predictions.csv"


def _mock_responder(prompt: str) -> str:
    if '"intent"' in prompt:
        return '{"intent": "delivery_delay", "confidence": 0.78}'
    return "Sorry about that! Can you DM us your order ID so we can look into it?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="only process the first N candidates")
    p.add_argument("--rpm", type=int, default=10, help="requests/minute cap")
    p.add_argument("--max-calls", type=int, default=None, help="hard cap on live calls per client")
    p.add_argument("--mock", action="store_true", help="dry run with canned responses, no key/network/quota")
    p.add_argument("--k", type=int, default=3, help="number of grounding examples to retrieve")
    args = p.parse_args()

    candidates = pd.read_csv("eval/golden_candidates.csv")
    if args.limit:
        candidates = candidates.head(args.limit)

    pairs = pd.read_csv("data/amazonhelp_pairs.csv")
    with open("eval/golden_tweet_ids.txt") as f:
        golden_ids = {int(x) for x in f.read().split()}
    # Leakage guard, same as the candidate builder: the agent must never
    # retrieve the golden-set example it is currently being asked about.
    remainder = pairs[~pairs["company_tweet_id"].isin(golden_ids)].reset_index(drop=True)
    print(f"Retrieval corpus: {len(remainder):,} pairs ({len(golden_ids)} golden-set pairs excluded)")

    index = RetrievalIndex(backend="auto").fit(remainder)
    print(f"Retrieval backend: {index.backend}")

    if args.mock:
        classify_client = MockClient(responder=_mock_responder)
        draft_client = MockClient(responder=_mock_responder)
    else:
        classify_client = OllamaClient(CLASSIFY_MODEL, requests_per_minute=args.rpm, max_calls=args.max_calls)
        draft_client = OllamaClient(DRAFT_MODEL, requests_per_minute=args.rpm, max_calls=args.max_calls)

    rows = []
    for n, (_, r) in enumerate(candidates.iterrows(), start=1):
        msg = str(r["customer_text"])
        c = classify(msg, classify_client)
        d = draft_reply(msg, c.intent, index, draft_client, k=args.k)
        dec = decide(msg, c.intent, c.confidence, d.top_similarity)

        grounding = index.query(msg, k=args.k)
        grounding_blob = " ||| ".join(
            f"Q: {row.customer_text} -> A: {row.company_text}" for row in grounding.itertuples()
        )

        rows.append({
            "golden_id": r["golden_id"],
            "agent_intent": c.intent,
            "agent_confidence": c.confidence,
            "agent_reply": d.reply,
            "agent_top_similarity": d.top_similarity,
            "agent_action": dec.action,
            "agent_reason": dec.reason,
            "grounding_examples": grounding_blob,
        })

        if n % 25 == 0:
            print(f"  {n}/{len(candidates)} | classify: {classify_client.usage.summary() if not args.mock else 'mock'}")
            # Write incrementally so an interrupted run isn't lost.
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} agent predictions to {OUT_PATH}")
    print("\nagent_action distribution:")
    print(out["agent_action"].value_counts().to_string())
    print("\nagent_intent distribution:")
    print(out["agent_intent"].value_counts().to_string())

    if not args.mock:
        print(f"\n[classify usage] {classify_client.usage.summary()}")
        print(f"[draft usage]    {draft_client.usage.summary()}")
        print("\nReal token counts and spend: https://aistudio.google.com -- the "
              "estimate above is char/4 and is only a sanity check.")


if __name__ == "__main__":
    main()
