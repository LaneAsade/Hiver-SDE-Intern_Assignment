"""
load_golden.py

Converts the hand-labeled eval/golden_candidates.csv into
eval/golden_set.jsonl, validating as it goes.

Validation matters more than it looks: a typo'd intent label or a blank
gold_action silently becomes a wrong metric later, and a wrong metric is
worse than a missing one because nobody questions it. This fails loudly
instead.
"""
import json
import sys

import pandas as pd

from src.classify.classifier import TAXONOMY

VALID_ACTIONS = {"auto_handle", "escalate"}


def validate(df: pd.DataFrame) -> list[str]:
    problems = []

    unlabeled = df[df["gold_intent"].isna() | (df["gold_intent"].astype(str).str.strip() == "")]
    if len(unlabeled):
        problems.append(f"{len(unlabeled)} row(s) missing gold_intent: {unlabeled['golden_id'].tolist()[:10]}")

    bad_intent = df[~df["gold_intent"].isin(TAXONOMY.keys()) & df["gold_intent"].notna()]
    bad_intent = bad_intent[bad_intent["gold_intent"].astype(str).str.strip() != ""]
    if len(bad_intent):
        pairs = bad_intent[["golden_id", "gold_intent"]].head(10).to_dict("records")
        problems.append(f"{len(bad_intent)} row(s) with an intent not in the taxonomy: {pairs}")

    bad_action = df[~df["gold_action"].isin(VALID_ACTIONS)]
    if len(bad_action):
        pairs = bad_action[["golden_id", "gold_action"]].head(10).to_dict("records")
        problems.append(f"{len(bad_action)} row(s) with gold_action not in {VALID_ACTIONS}: {pairs}")

    missing_reason = df[
        (df["gold_action"] == "escalate")
        & (df["gold_action_reason"].isna() | (df["gold_action_reason"].astype(str).str.strip() == ""))
    ]
    if len(missing_reason):
        problems.append(
            f"{len(missing_reason)} escalate row(s) missing gold_action_reason "
            f"(the assignment requires a stated reason): {missing_reason['golden_id'].tolist()[:10]}"
        )

    return problems


def main(strict: bool = True):
    df = pd.read_csv("eval/golden_candidates.csv")
    problems = validate(df)

    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        if strict:
            print("\nFix these in eval/golden_candidates.csv and re-run. "
                  "(Pass --lenient to write out only the valid rows instead.)")
            sys.exit(1)
        print("\n--lenient: writing only fully-valid rows.\n")
        df = df[
            df["gold_intent"].isin(TAXONOMY.keys())
            & df["gold_action"].isin(VALID_ACTIONS)
        ]

    records = df.to_dict("records")
    with open("eval/golden_set.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps({k: (None if pd.isna(v) else v) for k, v in r.items()}) + "\n")

    print(f"Wrote {len(records)} labeled examples to eval/golden_set.jsonl")
    print("\ngold_intent distribution:")
    print(df["gold_intent"].value_counts().to_string())
    print("\ngold_action distribution:")
    print(df["gold_action"].value_counts().to_string())


if __name__ == "__main__":
    main(strict="--lenient" not in sys.argv)
