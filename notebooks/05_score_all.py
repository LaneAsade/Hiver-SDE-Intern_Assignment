"""
05_score_all.py

Scores trivial vs. simple vs. LLM agent against the hand-labeled gold
labels, and prints the comparison table that goes in the report.

Requires eval/golden_candidates.csv to have its gold_* columns filled in.
Agent columns are optional -- if eval/agent_predictions.csv doesn't exist
yet, the baselines are still scored, so you can see the floor before
spending any credits.

Reply quality (LLM judge) is scored separately in 06_run_judge.py, since
that costs money and this doesn't.
"""
import os

import pandas as pd

from src.eval.load_golden import validate
from src.eval.metrics import classification_metrics, escalation_metrics

SYSTEMS = ["trivial", "simple", "agent"]


def main():
    gold = pd.read_csv("eval/golden_candidates.csv")

    problems = validate(gold)
    if problems:
        print("Gold labels aren't ready yet:")
        for p in problems:
            print(f"  - {p}")
        print("\nFill in gold_intent / gold_action / gold_action_reason in")
        print("eval/golden_candidates.csv, then re-run. See eval/labeling_guidelines.md.")
        return

    df = gold.merge(pd.read_csv("eval/baseline_predictions.csv"), on="golden_id")

    if os.path.exists("eval/agent_predictions.csv"):
        agent = pd.read_csv("eval/agent_predictions.csv")
        df = df.merge(agent, on="golden_id", how="left")
        scored_systems = [s for s in SYSTEMS if f"{s}_intent" in df.columns]
        n_agent = df["agent_intent"].notna().sum() if "agent_intent" in df else 0
        if n_agent < len(df):
            print(f"NOTE: agent predictions cover {n_agent}/{len(df)} rows "
                  f"-- agent metrics below are on that subset only.\n")
    else:
        scored_systems = ["trivial", "simple"]
        print("NOTE: eval/agent_predictions.csv not found -- scoring baselines only.")
        print("Run notebooks/04_run_agent.py to add the LLM agent.\n")

    rows = []
    for sys_name in scored_systems:
        sub = df[df[f"{sys_name}_intent"].notna()]
        if len(sub) == 0:
            continue

        cls = classification_metrics(sub["gold_intent"].tolist(), sub[f"{sys_name}_intent"].tolist())
        esc = escalation_metrics(
            sub["gold_action"].tolist(),
            sub[f"{sys_name}_action"].tolist(),
            intents=sub["gold_intent"].tolist(),
        )
        auto_rate = (sub[f"{sys_name}_action"] == "auto_handle").mean()

        rows.append({
            "system": sys_name,
            "n": len(sub),
            "intent_accuracy": round(cls["accuracy"], 3),
            "intent_macro_f1": round(cls["macro_f1"], 3),
            "escalate_precision": round(esc["precision_escalate"], 3),
            "escalate_recall": round(esc["recall_escalate"], 3),
            "auto_handle_rate": round(auto_rate, 3),
            "missed_escalations": esc["fn"],
        })

    table = pd.DataFrame(rows)
    print("=" * 78)
    print("HEADLINE COMPARISON")
    print("=" * 78)
    print(table.to_string(index=False))
    print()
    print("Read escalate_recall first, not accuracy: a missed escalation "
          "(auto-handling something\nthat needed a human) is the expensive "
          "error. missed_escalations is that count in absolute terms.")
    table.to_csv("eval/results_summary.csv", index=False)

    # Per-intent escalation recall -- a pooled number can hide that an
    # entire high-stakes category was missed.
    for sys_name in scored_systems:
        sub = df[df[f"{sys_name}_intent"].notna()]
        if len(sub) == 0:
            continue
        esc = escalation_metrics(
            sub["gold_action"].tolist(), sub[f"{sys_name}_action"].tolist(),
            intents=sub["gold_intent"].tolist(),
        )
        print(f"\n--- {sys_name}: escalation recall by gold intent ---")
        by = pd.DataFrame(esc["by_intent"]).T.sort_values("n_gold_escalate", ascending=False)
        print(by.to_string())

    # Every row any system got wrong -- the raw material for the report's
    # failure analysis section.
    err_rows = []
    for sys_name in scored_systems:
        sub = df[df[f"{sys_name}_intent"].notna()]
        wrong = sub[
            (sub[f"{sys_name}_intent"] != sub["gold_intent"])
            | (sub[f"{sys_name}_action"] != sub["gold_action"])
        ]
        for _, r in wrong.iterrows():
            err_rows.append({
                "system": sys_name,
                "golden_id": r["golden_id"],
                "customer_text": r["customer_text"],
                "gold_intent": r["gold_intent"],
                "pred_intent": r[f"{sys_name}_intent"],
                "gold_action": r["gold_action"],
                "pred_action": r[f"{sys_name}_action"],
                "gold_reason": r.get("gold_action_reason", ""),
                "pred_reason": r.get(f"{sys_name}_reason", ""),
                "missed_escalation": r["gold_action"] == "escalate" and r[f"{sys_name}_action"] == "auto_handle",
            })
    errors = pd.DataFrame(err_rows)
    errors.to_csv("eval/errors.csv", index=False)
    print(f"\n\nWrote {len(errors)} error rows to eval/errors.csv (failure-analysis material)")
    print(f"Wrote summary table to eval/results_summary.csv")


if __name__ == "__main__":
    main()
