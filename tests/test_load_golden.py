import pandas as pd

from src.eval.load_golden import validate


def _frame(**overrides):
    base = {
        "golden_id": ["g0001"],
        "gold_intent": ["delivery_delay"],
        "gold_action": ["escalate"],
        "gold_action_reason": ["customer is angry and has contacted twice"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_valid_row_has_no_problems():
    assert validate(_frame()) == []


def test_missing_intent_is_caught():
    problems = validate(_frame(gold_intent=[""]))
    assert any("missing gold_intent" in p for p in problems)


def test_intent_not_in_taxonomy_is_caught():
    problems = validate(_frame(gold_intent=["delivery_dealy"]))  # typo
    assert any("not in the taxonomy" in p for p in problems)


def test_invalid_action_is_caught():
    problems = validate(_frame(gold_action=["maybe"]))
    assert any("gold_action not in" in p for p in problems)


def test_escalate_without_reason_is_caught():
    problems = validate(_frame(gold_action_reason=[""]))
    assert any("missing gold_action_reason" in p for p in problems)


def test_auto_handle_without_reason_is_allowed():
    df = _frame(gold_action=["auto_handle"], gold_action_reason=[""])
    assert validate(df) == []
