"""
judge.py

LLM-as-judge for drafted reply quality. Scores four dimensions, 1-5 each:
- groundedness: doesn't state facts absent from the retrieved context
- helpfulness: actually addresses what the customer asked
- tone: matches AmazonHelp's brand voice
- conciseness: appropriate length for a Twitter reply

Prompt construction and parsing are unit-testable without a live key;
judge() itself needs an LLMClient (real or Mock).

Self-preference bias risk: if the judge model is the same model that
generated the reply, scores may be systematically inflated. Mitigation
here is partial -- classify/draft/judge already use different Gemini
tiers (flash-lite / flash / judge, configurable) rather than one model
for everything, but that doesn't fully remove same-family bias. The
human-agreement check (see eval/metrics.py::judge_human_agreement) is
what actually catches this if it's happening, not the tier choice alone.
"""
import json
import re
from dataclasses import dataclass

from src.llm_client import LLMClient

RUBRIC = {
    "groundedness": (
        "Does the reply avoid stating any fact (dates, amounts, policies) not "
        "supported by the provided grounding examples? 5 = fully grounded, "
        "1 = invents unsupported specifics."
    ),
    "helpfulness": (
        "Does the reply actually address what the customer asked, moving them "
        "toward resolution? 5 = directly helpful, 1 = ignores the actual issue."
    ),
    "tone": (
        "Does the reply match a realistic AmazonHelp support tone (professional, "
        "warm, concise)? 5 = on-brand, 1 = generic or off-brand."
    ),
    "conciseness": (
        "Is the reply an appropriate length for a Twitter reply, no unnecessary "
        "padding? 5 = tight, 1 = bloated or truncated mid-thought."
    ),
}
DIMENSIONS = list(RUBRIC.keys())


def build_prompt(customer_message: str, drafted_reply: str, grounding_examples: str) -> str:
    rubric_block = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    return f"""Score this customer-support reply on the rubric below, 1-5 per dimension.

Customer message: "{customer_message}"

Grounding examples the reply was supposed to draw from:
{grounding_examples}

Drafted reply: "{drafted_reply}"

Rubric:
{rubric_block}

Respond with ONLY a JSON object, no other text:
{{"groundedness": <1-5>, "helpfulness": <1-5>, "tone": <1-5>, "conciseness": <1-5>, "rationale": "<one sentence>"}}"""


@dataclass
class JudgeScore:
    groundedness: int
    helpfulness: int
    tone: int
    conciseness: int
    rationale: str
    raw: str
    parse_failed: bool = False

    @property
    def mean(self) -> float:
        return (self.groundedness + self.helpfulness + self.tone + self.conciseness) / 4


def parse_response(raw_text: str) -> JudgeScore:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return JudgeScore(1, 1, 1, 1, "PARSE_FAILURE", raw_text, parse_failed=True)
    try:
        obj = json.loads(match.group(0))

        def clamp(x):
            return min(max(int(x), 1), 5)

        return JudgeScore(
            groundedness=clamp(obj.get("groundedness", 1)),
            helpfulness=clamp(obj.get("helpfulness", 1)),
            tone=clamp(obj.get("tone", 1)),
            conciseness=clamp(obj.get("conciseness", 1)),
            rationale=str(obj.get("rationale", ""))[:300],
            raw=raw_text,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return JudgeScore(1, 1, 1, 1, "PARSE_FAILURE", raw_text, parse_failed=True)


def judge(customer_message: str, drafted_reply: str, grounding_examples: str, client: LLMClient) -> JudgeScore:
    prompt = build_prompt(customer_message, drafted_reply, grounding_examples)
    raw = client.complete(prompt, max_tokens=400)
    return parse_response(raw)
