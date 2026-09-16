from src.classify.heuristic import guess_intent


def test_negated_praise_flips_to_complaint():
    label, terms, multi = guess_intent("Not impressed. Echo and Prime just money down the drain as unusable")
    assert label == "service_complaint_general"
    assert any("negated:impressed" in t for t in terms)


def test_unnegated_praise_stays_praise():
    label, _, _ = guess_intent("Thanks for being an awesome customer and reaching out to us today")
    assert label == "praise_gratitude"


def test_negated_complaint_is_suppressed_not_flipped():
    # should NOT become praise_gratitude -- negated complaint terms are
    # only suppressed, that inference is much less reliable than the
    # reverse (sarcasm, understatement).
    label, _, _ = guess_intent("well that wasn't the worst experience I suppose")
    assert label != "praise_gratitude"


def test_multi_match_labels_both_categories():
    label, terms, multi = guess_intent("I want to cancel my order and get a refund")
    assert multi is True
    joined = "; ".join(terms)
    assert "order_cancellation:" in joined
    assert "refund_billing_payment:" in joined


def test_single_match_no_multi_flag():
    label, terms, multi = guess_intent("still waiting, hasn't arrived yet")
    assert label == "delivery_delay"
    assert multi is False


def test_no_match_falls_to_other():
    label, terms, multi = guess_intent("asdf random text with no keywords")
    assert label == "other"
    assert terms == []
