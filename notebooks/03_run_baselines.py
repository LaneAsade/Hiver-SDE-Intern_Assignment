"""
03_run_baselines.py

Runs the trivial and simple baselines against the golden-set candidates
and saves predictions to eval/baseline_predictions.csv, keyed by
golden_id. No LLM calls, no training data -- both baselines are fully
computable right now. Real scoring against gold labels happens once
eval/golden_candidates.csv has its gold_* columns filled in; this just
produces the predictions side so scoring is a join + metrics call away.
"""
import pandas as pd

from src.baselines.simple import simple_classify, simple_decide, simple_reply
from src.baselines.trivial import compute_majority_intent, trivial_classify, trivial_decide, trivial_reply
from src.retrieve.index import RetrievalIndex


def main():
    candidates = pd.read_csv("eval/golden_candidates.csv")
    pairs = pd.read_csv("data/amazonhelp_pairs.csv")

    with open("eval/golden_tweet_ids.txt") as f:
        golden_ids = {int(x) for x in f.read().split()}
    remainder = pairs[~pairs["company_tweet_id"].isin(golden_ids)].reset_index(drop=True)

    majority_intent = compute_majority_intent(
        remainder["customer_text"].astype(str).sample(20000, random_state=1)
    )
    print(f"trivial baseline's fixed intent guess (majority, from heuristic over 20k messages): {majority_intent}")

    index = RetrievalIndex(backend="auto").fit(remainder)

    rows = []
    for _, r in candidates.iterrows():
        msg = r["customer_text"]

        t_intent = trivial_classify(msg, majority_intent)
        t_reply = trivial_reply(msg)
        t_action, t_reason = trivial_decide(msg)

        s_intent = simple_classify(msg)
        s_reply = simple_reply(msg, index)
        s_action, s_reason = simple_decide(msg, s_intent)

        rows.append({
            "golden_id": r["golden_id"],
            "trivial_intent": t_intent,
            "trivial_reply": t_reply,
            "trivial_action": t_action,
            "trivial_reason": t_reason,
            "simple_intent": s_intent,
            "simple_reply": s_reply,
            "simple_action": s_action,
            "simple_reason": s_reason,
        })

    out = pd.DataFrame(rows)
    out.to_csv("eval/baseline_predictions.csv", index=False)
    print(f"\nWrote {len(out)} baseline predictions to eval/baseline_predictions.csv")
    print("\nsimple_action distribution:")
    print(out["simple_action"].value_counts().to_string())
    print("\ntrivial_action distribution:")
    print(out["trivial_action"].value_counts().to_string())


if __name__ == "__main__":
    main()
