from src.eval.judge import RUBRIC, build_prompt, parse_response


def test_prompt_contains_rubric_dimensions():
    prompt = build_prompt("where's my order", "we're looking into it", "example grounding")
    for dim in RUBRIC:
        assert dim in prompt


def test_prompt_contains_message_and_reply():
    prompt = build_prompt("a specific customer message", "a specific drafted reply", "grounding text")
    assert "a specific customer message" in prompt
    assert "a specific drafted reply" in prompt


def test_parse_valid_json():
    s = parse_response('{"groundedness": 4, "helpfulness": 5, "tone": 3, "conciseness": 4, "rationale": "good"}')
    assert s.groundedness == 4
    assert s.helpfulness == 5
    assert s.mean == 4.0
    assert not s.parse_failed


def test_parse_clamps_out_of_range_scores():
    s = parse_response('{"groundedness": 9, "helpfulness": 0, "tone": 3, "conciseness": 3}')
    assert s.groundedness == 5
    assert s.helpfulness == 1


def test_parse_malformed_json_flags_failure():
    s = parse_response("not valid json")
    assert s.parse_failed is True


def test_parse_json_with_surrounding_text():
    s = parse_response('Here is my score:\n{"groundedness": 5, "helpfulness": 5, "tone": 5, "conciseness": 5}\nDone.')
    assert s.mean == 5.0
    assert not s.parse_failed
