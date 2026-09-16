from src.classify.classifier import TAXONOMY, build_prompt, parse_response


def test_prompt_contains_all_taxonomy_labels():
    prompt = build_prompt("where is my order")
    for label in TAXONOMY:
        assert label in prompt


def test_prompt_contains_the_message():
    prompt = build_prompt("a very specific customer message")
    assert "a very specific customer message" in prompt


def test_parse_valid_json():
    c = parse_response('{"intent": "delivery_delay", "confidence": 0.83}')
    assert c.intent == "delivery_delay"
    assert c.confidence == 0.83


def test_parse_json_with_surrounding_text():
    c = parse_response('Sure, here you go:\n{"intent": "refund_billing_payment", "confidence": 0.7}\nHope that helps!')
    assert c.intent == "refund_billing_payment"


def test_parse_unknown_intent_falls_back_to_other():
    c = parse_response('{"intent": "made_up_label", "confidence": 0.9}')
    assert c.intent == "other"


def test_parse_malformed_json_falls_back_to_other():
    c = parse_response("not json at all")
    assert c.intent == "other"
    assert c.confidence == 0.0


def test_parse_confidence_is_clamped():
    c = parse_response('{"intent": "other", "confidence": 5.0}')
    assert c.confidence == 1.0
