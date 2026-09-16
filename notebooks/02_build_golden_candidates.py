"""
02_build_golden_candidates.py

Builds the golden-set candidate pool for human labeling:
- Stratified across the taxonomy using the keyword heuristic (no real
  classifier exists yet), deliberately oversampling rare/high-stakes
  intents relative to their natural frequency.
- A targeted supplement of messages matching guardrail-trigger phrasing
  (legal/fraud terms, explicit requests for a human), regardless of their
  heuristic bucket, so the escalation guardrail tier gets real coverage.
- retrieval_top_similarity is computed against an index that EXCLUDES the
  selected candidates themselves -- otherwise a candidate would retrieve
  itself at ~1.0 similarity, exactly the leakage bug flagged in Phase 1.

Output is a *candidate* pool: suggested_intent/suggested_action are
starting points for a human to confirm or correct, never final labels.
"""
import pandas as pd

from src.classify.heuristic import guess_intent
from src.escalate.decide import RISK_TERM_RE, _is_genuine_human_request
from src.retrieve.index import RetrievalIndex

TARGET_PER_INTENT = {
    "delivery_delay": 28,
    "delivery_discrepancy": 20,
    "order_cancellation": 18,
    "refund_billing_payment": 28,
    "account_access": 22,
    "product_defect_warranty": 18,
    "app_website_technical": 14,
    "service_complaint_general": 18,
    "praise_gratitude": 18,
    "other": 16,
}
N_GUARDRAIL_SUPPLEMENT = 12
RANDOM_STATE = 42


def main():
    pairs = pd.read_csv("data/amazonhelp_pairs.csv")
    pairs = pairs.drop_duplicates(subset="customer_text").reset_index(drop=True)

    guesses = pairs["customer_text"].astype(str).apply(guess_intent)
    pairs["heuristic_intent"] = guesses.apply(lambda t: t[0])
    pairs["heuristic_matched_on"] = guesses.apply(lambda t: "; ".join(t[1]))
    pairs["heuristic_multi_match"] = guesses.apply(lambda t: t[2])

    text = pairs["customer_text"].astype(str)
    pairs["guardrail_hit"] = "none"
    pairs.loc[text.str.contains(RISK_TERM_RE), "guardrail_hit"] = "risk_term"
    pairs.loc[text.apply(_is_genuine_human_request), "guardrail_hit"] = "human_request"

    selected_idx = set()
    for intent, n in TARGET_PER_INTENT.items():
        pool = pairs[pairs["heuristic_intent"] == intent]
        picked = pool.sample(min(n, len(pool)), random_state=RANDOM_STATE)
        selected_idx.update(picked.index)

    guardrail_pool = pairs[(pairs["guardrail_hit"] != "none") & (~pairs.index.isin(selected_idx))]
    n_sup = min(N_GUARDRAIL_SUPPLEMENT, len(guardrail_pool))
    if n_sup:
        selected_idx.update(guardrail_pool.sample(n_sup, random_state=RANDOM_STATE).index)

    candidates = pairs.loc[sorted(selected_idx)].reset_index(drop=True)

    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0

    def is_english(t: str) -> bool:
        try:
            return detect(t) == "en"
        except LangDetectException:
            return False

    before = len(candidates)
    candidates = candidates[candidates["customer_text"].astype(str).apply(is_english)].reset_index(drop=True)
    dropped = before - len(candidates)
    if dropped:
        print(f"Dropped {dropped} non-English candidate(s) (project scope is English-only)")

    candidates["golden_id"] = [f"g{i:04d}" for i in range(len(candidates))]

    # Leakage guard: fit the retrieval index on everything EXCEPT the
    # selected candidates before scoring them.
    remainder = pairs.drop(index=selected_idx).reset_index(drop=True)
    index = RetrievalIndex(backend="auto").fit(remainder)
    sims = [
        float(index.query(t, k=1)["similarity"].iloc[0])
        for t in candidates["customer_text"]
    ]
    candidates["retrieval_top_similarity"] = sims

    # Only the deterministic guardrail tier gets pre-filled -- confidence-
    # gated auto-handle needs a real classifier confidence this sandbox
    # doesn't have, so that call is left to the human labeler.
    candidates["suggested_action"] = candidates["guardrail_hit"].apply(
        lambda g: "escalate" if g != "none" else ""
    )

    out = candidates[[
        "golden_id", "customer_text", "company_text",
        "heuristic_intent", "heuristic_matched_on", "heuristic_multi_match",
        "guardrail_hit", "retrieval_top_similarity", "suggested_action",
        "company_tweet_id",
    ]].rename(columns={"company_text": "actual_historical_reply", "heuristic_intent": "suggested_intent"})

    for col in ["gold_intent", "gold_action_reason", "must_mention_facts", "notes"]:
        out[col] = ""
    out["gold_action"] = out["suggested_action"]

    out.to_csv("eval/golden_candidates.csv", index=False)
    with open("eval/golden_tweet_ids.txt", "w") as f:
        f.write("\n".join(out["company_tweet_id"].astype(str)))

    print(f"Wrote {len(out)} candidates to eval/golden_candidates.csv\n")
    print("suggested_intent distribution:")
    print(out["suggested_intent"].value_counts().to_string())
    print("\nguardrail_hit distribution:")
    print(out["guardrail_hit"].value_counts().to_string())
    print(f"\nheuristic_multi_match: {out['heuristic_multi_match'].sum()} of {len(out)}")


if __name__ == "__main__":
    main()
