from src.eval.metrics import classification_metrics, escalation_metrics, judge_human_agreement


def test_judge_agreement_perfect():
    scores = [5, 4, 3, 2, 1, 4, 5]
    m = judge_human_agreement(scores, scores)
    assert m["quadratic_weighted_kappa"] == 1.0
    assert m["exact_agreement"] == 1.0
    assert m["mean_abs_diff"] == 0.0


def test_judge_agreement_off_by_one_still_high_kappa():
    human = [5, 4, 3, 2, 1]
    judge = [4, 3, 2, 1, 2]
    m = judge_human_agreement(human, judge)
    assert m["within_1"] == 1.0
    # quadratic weighting should keep near-misses from tanking the score
    assert m["quadratic_weighted_kappa"] > 0.5


def test_judge_agreement_inverted_scores_gives_negative_kappa():
    human = [5, 4, 3, 2, 1]
    judge = [1, 2, 3, 4, 5]
    m = judge_human_agreement(human, judge)
    assert m["quadratic_weighted_kappa"] < 0
    assert m["spearman"] < 0


def test_judge_agreement_constant_rater_is_nan_not_zero():
    m = judge_human_agreement([4, 4, 4, 4], [5, 4, 3, 4])
    assert m["quadratic_weighted_kappa"] != m["quadratic_weighted_kappa"]  # nan
    assert "note" in m


def test_judge_agreement_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        judge_human_agreement([1, 2, 3], [1, 2])


def test_classification_metrics_perfect():
    gold = ["a", "b", "a", "c"]
    pred = ["a", "b", "a", "c"]
    m = classification_metrics(gold, pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_classification_metrics_partial():
    gold = ["a", "b", "a", "c"]
    pred = ["a", "a", "a", "c"]
    m = classification_metrics(gold, pred)
    assert 0 < m["accuracy"] < 1.0


def test_escalation_metrics_perfect():
    gold = ["escalate", "escalate", "auto_handle"]
    pred = ["escalate", "escalate", "auto_handle"]
    m = escalation_metrics(gold, pred)
    assert m["recall_escalate"] == 1.0
    assert m["precision_escalate"] == 1.0


def test_escalation_metrics_missed_escalation_hurts_recall_not_precision():
    gold = ["escalate", "escalate", "auto_handle"]
    pred = ["auto_handle", "escalate", "auto_handle"]  # missed one real escalation
    m = escalation_metrics(gold, pred)
    assert m["recall_escalate"] == 0.5
    assert m["precision_escalate"] == 1.0


def test_escalation_metrics_by_intent_disaggregation():
    gold = ["escalate", "escalate"]
    pred = ["auto_handle", "auto_handle"]  # both missed
    intents = ["account_access", "delivery_delay"]
    m = escalation_metrics(gold, pred, intents=intents)
    assert m["by_intent"]["account_access"]["recall_escalate"] == 0.0
    assert m["by_intent"]["account_access"]["n_gold_escalate"] == 1
