"""
decide.py

Two-tier auto-handle / escalate decision:
1. Hard guardrails (deterministic, auditable) -- checked first, always win.
2. Confidence + grounding thresholds -- everything else.

Every decision returns a stated reason, per the assignment's requirement.
No LLM call involved, so this is pure logic and fully unit-testable.
"""
import re
from dataclasses import dataclass

# Intents that are always human-handled regardless of model confidence.
# Placeholder set -- revisit once the golden set surfaces real account/
# security-adjacent cases during labeling.
SAFETY_INTENTS = {"account_access"}

RISK_TERM_RE = re.compile(
    r"\b(?:lawyer|attorney|sue|lawsuit|legal action|fraud|scam|hacked|unauthorized charge)\b",
    re.IGNORECASE,
)
# NOTE: checked ~275 real messages containing "scam" before touching this --
# most are either genuine phishing-verification questions ("is this email a
# scam?") or real billing disputes, not hyperbole about routine delays. Left
# unchanged on purpose: removing it would trade a rare, cheap false positive
# (escalating a hyperbolic complaint) for a real chance of missing a genuine
# fraud/phishing signal, which is a worse trade.
HUMAN_REQUEST_RE = re.compile(
    r"\b(?:speak to a human|talk to a person|real person|human agent|manager)\b",
    re.IGNORECASE,
)
# Sanity-checked against ~35 real matches: the phrasing above is ~92%
# genuine escalation requests. The one real failure mode found was past-
# tense praise ("...talking to some real person on customer care" as a
# compliment) -- narrow fix below rather than rewriting the whole pattern
# and risking recall on the genuine cases.
_POSITIVE_SENTIMENT_RE = re.compile(
    r"\b(?:thanks?|thank you|great|awesome|appreciate|impressed|prompt)\b", re.IGNORECASE
)


def _is_genuine_human_request(message: str) -> bool:
    m = HUMAN_REQUEST_RE.search(message)
    if not m:
        return False
    preceding = message[: m.start()]  # whole prefix, not a fixed window -- tweets are short (<=280 chars)
    if _POSITIVE_SENTIMENT_RE.search(preceding):
        return False  # likely praise ("Thanks... talking to some real person"), not a request
    return True

# Tune both thresholds once the golden set gives a real precision/recall
# curve to pick them from -- these are reasonable starting points, not
# fitted values.
CONF_THRESHOLD = 0.65
SIM_THRESHOLD = 0.35


@dataclass
class Decision:
    action: str  # "auto_handle" | "escalate"
    reason: str


def decide(message: str, intent: str, classifier_conf: float, retrieval_sim: float) -> Decision:
    if RISK_TERM_RE.search(message):
        return Decision("escalate", "guardrail: message contains a legal/fraud/security risk term")
    if _is_genuine_human_request(message):
        return Decision("escalate", "guardrail: customer explicitly asked for a human")
    if intent in SAFETY_INTENTS:
        return Decision("escalate", f"guardrail: intent={intent} is always human-handled by policy")
    if intent == "praise_gratitude":
        return Decision("auto_handle", "praise/gratitude message -- brief acknowledgment, no resolution needed")

    if classifier_conf >= CONF_THRESHOLD and retrieval_sim >= SIM_THRESHOLD:
        return Decision(
            "auto_handle",
            f"confidence={classifier_conf:.2f} >= {CONF_THRESHOLD} and "
            f"retrieval_sim={retrieval_sim:.2f} >= {SIM_THRESHOLD}: confident precedent found",
        )
    return Decision(
        "escalate",
        f"confidence={classifier_conf:.2f} or retrieval_sim={retrieval_sim:.2f} "
        f"below threshold -- no confident precedent",
    )
