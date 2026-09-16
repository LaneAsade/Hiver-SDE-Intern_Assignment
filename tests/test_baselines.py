from src.baselines.simple import simple_classify, simple_decide
from src.baselines.trivial import compute_majority_intent, trivial_decide, trivial_reply


def test_majority_intent_picks_the_most_common():
    texts = ["still waiting, hasn't arrived"] * 5 + ["thanks so much!"] * 2
    assert compute_majority_intent(texts) == "delivery_delay"


def test_trivial_reply_is_always_the_same():
    assert trivial_reply("anything") == trivial_reply("something totally different")


def test_trivial_always_escalates():
    action, _ = trivial_decide("anything at all")
    assert action == "escalate"


def test_simple_classify_uses_heuristic():
    assert simple_classify("still waiting, hasn't arrived") == "delivery_delay"


def test_simple_decide_allowlist_auto_handles():
    action, reason = simple_decide("still waiting on my order", "delivery_delay")
    assert action == "auto_handle"
    assert "allow-list" in reason


def test_simple_decide_non_allowlist_escalates():
    action, _ = simple_decide("my app keeps crashing", "app_website_technical")
    assert action == "escalate"


def test_simple_decide_guardrail_overrides_allowlist():
    # delivery_delay is on the allow-list, but a risk term should still win
    action, reason = simple_decide("still waiting, this is fraud, calling my lawyer", "delivery_delay")
    assert action == "escalate"
    assert "guardrail" in reason
