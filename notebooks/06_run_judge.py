"""
06_run_judge.py

Two modes:

  --run     Score agent replies with the LLM judge -> eval/judge_scores.csv
  --sheet   Emit a blind human-scoring sheet     -> eval/human_scoring_sheet.csv
  --agree   Compare your scores to the judge's   -> agreement stats

The sheet deliberately does NOT contain the judge's scores. The
assignment asks for evidence the judge agrees with a human, and that
evidence is worthless if the human can see the judge's answer while
scoring -- you'd be measuring anchoring, not agreement. Fill in the blank
columns, then run --agree.

Typical flow:
    PYTHONPATH=. python3 notebooks/06_run_judge.py --run --limit 40
    PYTHONPATH=. python3 notebooks/06_run_judge.py --sheet --n 40
    # ... score eval/human_scoring_sheet.csv by hand ...
    PYTHONPATH=. python3 notebooks/06_run_judge.py --agree
"""
import argparse
import os

import pandas as pd

from src.eval.judge import DIMENSIONS, judge
from src.eval.metrics import judge_human_agreement
from src.llm_client import GeminiClient, OpenAIClient, OllamaClient, MockClient

JUDGE_MODEL = os.environ.get("HIVER_JUDGE_MODEL", "gemini-3.5-flash")  # deliberately != DRAFT_MODEL
SCORES_PATH = "eval/judge_scores.csv"
SHEET_PATH = "eval/human_scoring_sheet.csv"


def _mock_responder(prompt: str) -> str:
    return '{"groundedness": 4, "helpfulness": 3, "tone": 4, "conciseness": 4, "rationale": "mock"}'


def run_judge(limit, rpm, max_calls, mock):
    preds = pd.read_csv("eval/agent_predictions.csv")
    cands = pd.read_csv("eval/golden_candidates.csv")[["golden_id", "customer_text"]]
    df = preds.merge(cands, on="golden_id")
    if limit:
        df = df.head(limit)

    client = MockClient(responder=_mock_responder) if mock else OllamaClient(
        JUDGE_MODEL, requests_per_minute=rpm, max_calls=max_calls
    )

    rows = []
    for n, (_, r) in enumerate(df.iterrows(), start=1):
        s = judge(
            str(r["customer_text"]),
            str(r["agent_reply"]),
            str(r.get("grounding_examples", "")),
            client,
        )
        rows.append({
            "golden_id": r["golden_id"],
            **{f"judge_{d}": getattr(s, d) for d in DIMENSIONS},
            "judge_mean": s.mean,
            "judge_rationale": s.rationale,
            "parse_failed": s.parse_failed,
        })
        if n % 20 == 0:
            print(f"  judged {n}/{len(df)}")
            pd.DataFrame(rows).to_csv(SCORES_PATH, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(SCORES_PATH, index=False)
    n_failed = out["parse_failed"].sum()
    print(f"\nWrote {len(out)} judge scores to {SCORES_PATH}")
    if n_failed:
        print(f"WARNING: {n_failed} response(s) failed to parse and were scored 1s -- "
              f"inspect before trusting the means.")
    print("\nMean score per dimension:")
    for d in DIMENSIONS:
        print(f"  {d}: {out[f'judge_{d}'].mean():.2f}")
    if not mock:
        print(f"\n[judge usage] {client.usage.summary()}")


def make_sheet(n):
    scores = pd.read_csv(SCORES_PATH)
    preds = pd.read_csv("eval/agent_predictions.csv")
    cands = pd.read_csv("eval/golden_candidates.csv")[["golden_id", "customer_text"]]

    # Sample from rows the judge actually scored, so both raters cover the
    # same examples.
    df = scores[["golden_id"]].merge(preds, on="golden_id").merge(cands, on="golden_id")
    df = df.sample(min(n, len(df)), random_state=42)

    sheet = df[["golden_id", "customer_text", "agent_reply", "grounding_examples"]].copy()
    for d in DIMENSIONS:
        sheet[f"human_{d}"] = ""          # <- you fill these in, 1-5
    sheet["human_notes"] = ""
    sheet.to_csv(SHEET_PATH, index=False)

    print(f"Wrote {len(sheet)} rows to {SHEET_PATH}")
    print("Judge scores are deliberately NOT in this file -- seeing them first")
    print("would measure anchoring rather than agreement.")
    print(f"\nScore each row 1-5 on: {', '.join(DIMENSIONS)}")
    print("Rubric definitions: src/eval/judge.py::RUBRIC")


def agreement():
    human = pd.read_csv(SHEET_PATH)
    scores = pd.read_csv(SCORES_PATH)
    df = human.merge(scores, on="golden_id")

    filled = df[df[f"human_{DIMENSIONS[0]}"].notna() & (df[f"human_{DIMENSIONS[0]}"].astype(str).str.strip() != "")]
    if len(filled) == 0:
        print(f"No human scores filled in yet -- complete {SHEET_PATH} first.")
        return

    print(f"Judge vs. human agreement on {len(filled)} examples\n")
    all_h, all_j = [], []
    for d in DIMENSIONS:
        h = filled[f"human_{d}"].astype(int).tolist()
        j = filled[f"judge_{d}"].astype(int).tolist()
        all_h += h
        all_j += j
        m = judge_human_agreement(h, j)
        print(f"{d}:")
        print(f"  weighted kappa {m['quadratic_weighted_kappa']:.3f} | "
              f"spearman {m['spearman']:.3f} | within-1 {m['within_1']:.0%} | "
              f"human mean {m['mean_human']:.2f} vs judge mean {m['mean_judge']:.2f}")

    overall = judge_human_agreement(all_h, all_j)
    print(f"\nPOOLED across all dimensions (n={overall['n']}):")
    print(f"  quadratic weighted kappa: {overall['quadratic_weighted_kappa']:.3f}")
    print(f"  spearman:                 {overall['spearman']:.3f}")
    print(f"  within 1 point:           {overall['within_1']:.0%}")
    print(f"  judge - human mean bias:  {overall['mean_judge'] - overall['mean_human']:+.2f}")
    print("\nRough reading: >0.6 kappa is reasonable agreement; <0.4 means the judge")
    print("isn't measuring what you're measuring, and reply-quality numbers need a")
    print("stated caveat. A positive mean bias is the self-preference-bias signature.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--sheet", action="store_true")
    p.add_argument("--agree", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n", type=int, default=40, help="rows in the human scoring sheet")
    p.add_argument("--rpm", type=int, default=10)
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    if args.run:
        run_judge(args.limit, args.rpm, args.max_calls, args.mock)
    elif args.sheet:
        if not os.path.exists(SCORES_PATH):
            print(f"{SCORES_PATH} not found -- run with --run first.")
        else:
            make_sheet(args.n)
    elif args.agree:
        agreement()
    else:
        p.print_help()
