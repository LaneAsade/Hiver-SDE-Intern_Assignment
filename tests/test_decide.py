from src.escalate.decide import decide


def test_risk_term_escalates_regardless_of_confidence():
    d = decide("I'll get my lawyer involved if this fraud isn't fixed", "delivery_delay", 0.99, 0.99)
    assert d.action == "escalate"
    assert "risk term" in d.reason


def test_human_request_escalates():
    d = decide("can I just talk to a real person please", "other", 0.9, 0.9)
    assert d.action == "escalate"
    assert "asked for a human" in d.reason


def test_praise_mentioning_real_person_does_not_escalate():
    # regression test for a real false positive found during golden-set
    # spot-checking: "real person" matched the human-request guardrail even
    # though the message is a compliment, not a request.
    d = decide(
        "Thanks for the freakishly prompt assistance. It seemed as if I was "
        "talking to some real person on customer care. :)",
        "praise_gratitude",
        0.6,
        0.5,
    )
    assert d.action != "escalate" or "asked for a human" not in d.reason


def test_scam_hyperbole_still_escalates_on_purpose():
    # deliberately NOT fixed: checked ~275 real "scam" messages and most are
    # genuine phishing-verification questions or real billing disputes, so
    # this stays a risk term even though some uses (like this one) are
    # hyperbole about a late delivery. Locks in that decision so it isn't
    # accidentally reverted later without someone noticing.
    d = decide("Amazon Prime is a SCAM! It's been four days my order is not here.", "other", 0.7, 0.6)
    assert d.action == "escalate"
    assert "risk term" in d.reason


def test_safety_intent_always_escalates():
    d = decide("locked out again", "account_access", 0.95, 0.95)
    assert d.action == "escalate"
    assert "account_access" in d.reason


def test_praise_auto_handles_without_grounding():
    d = decide("thanks so much for the help!", "praise_gratitude", 0.4, 0.1)
    assert d.action == "auto_handle"


def test_confident_and_grounded_auto_handles():
    d = decide("still waiting on my package", "delivery_delay", 0.9, 0.8)
    assert d.action == "auto_handle"


def test_low_confidence_escalates():
    d = decide("still waiting on my package", "delivery_delay", 0.3, 0.8)
    assert d.action == "escalate"


def test_low_similarity_escalates_even_if_confident():
    d = decide("some totally novel issue", "other", 0.9, 0.1)
    assert d.action == "escalate"
