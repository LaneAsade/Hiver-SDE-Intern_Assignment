"""
metrics.py

Classification and escalation metrics against gold labels. Pure
functions, no LLM/API dependency -- fully testable with synthetic data
before any real predictions or gold labels exist.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score


def judge_human_agreement(human_scores: list, judge_scores: list) -> dict:
    """
    Agreement between a human's scores and the LLM judge's, on the same
    examples. The assignment requires evidence the judge is trustworthy --
    this is that evidence.

    Quadratic-weighted Cohen's kappa is the headline: these are ordinal
    1-5 ratings, so being off by 1 should be penalized far less than
    being off by 3, which plain kappa doesn't capture. Spearman is
    reported alongside because kappa can look poor when variance is low
    (e.g. a human who rates almost everything 4-5), and the two failing
    in different ways is itself informative.

    Rough reading: >0.6 weighted kappa is reasonable agreement, <0.4
    means the judge is not measuring what the human is measuring and the
    reply-quality numbers shouldn't be trusted without saying so.
    """
    h = np.asarray(human_scores)
    j = np.asarray(judge_scores)
    if len(h) != len(j):
        raise ValueError(f"score lists must be same length, got {len(h)} and {len(j)}")

    out = {
        "n": len(h),
        "mean_human": float(h.mean()),
        "mean_judge": float(j.mean()),
        "mean_abs_diff": float(np.abs(h - j).mean()),
        "exact_agreement": float((h == j).mean()),
        "within_1": float((np.abs(h - j) <= 1).mean()),
    }
    # Both degenerate when a rater gives one value throughout -- report
    # nan rather than a misleading 0.0.
    if len(set(h.tolist())) > 1 and len(set(j.tolist())) > 1:
        out["quadratic_weighted_kappa"] = float(
            cohen_kappa_score(h, j, weights="quadratic", labels=[1, 2, 3, 4, 5])
        )
        out["spearman"] = float(spearmanr(h, j).statistic)
    else:
        out["quadratic_weighted_kappa"] = float("nan")
        out["spearman"] = float("nan")
        out["note"] = "one rater gave a constant score -- correlation undefined"
    return out


def classification_metrics(gold: list, pred: list) -> dict:
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
        "report": classification_report(gold, pred, zero_division=0),
    }


def escalation_metrics(gold_action: list, pred_action: list, intents: list | None = None) -> dict:
    """
    Precision/recall with "escalate" as the positive class -- a missed
    escalation (gold=escalate, pred=auto_handle) is the costly error, so
    recall_escalate is the number to watch most closely, not accuracy.
    """
    pairs = list(zip(gold_action, pred_action))
    tp = sum(1 for g, p in pairs if g == "escalate" and p == "escalate")
    fp = sum(1 for g, p in pairs if g == "auto_handle" and p == "escalate")
    fn = sum(1 for g, p in pairs if g == "escalate" and p == "auto_handle")
    tn = sum(1 for g, p in pairs if g == "auto_handle" and p == "auto_handle")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    out = {"precision_escalate": precision, "recall_escalate": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    if intents is not None:
        # Disaggregated recall -- a pooled number can hide that every
        # safety-relevant intent got missed.
        df = pd.DataFrame({"gold": gold_action, "pred": pred_action, "intent": intents})
        by_intent = {}
        for i, grp in df.groupby("intent"):
            g_esc = grp[grp["gold"] == "escalate"]
            rec = (g_esc["pred"] == "escalate").mean() if len(g_esc) else float("nan")
            by_intent[i] = {"n": len(grp), "n_gold_escalate": len(g_esc), "recall_escalate": rec}
        out["by_intent"] = by_intent
    return out
