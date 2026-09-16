"""
simple.py

The simple baseline: real logic, no LLM, no training data required.
- Intent: the keyword heuristic (src/classify/heuristic.py) -- rule-
  based and fully explainable, already built for golden-set sampling.
- Reply: nearest-neighbor lookup -- returns the actual historical reply
  to the single most similar past customer message, verbatim. No
  generation at all.
- Escalation: the same guardrails as the real system (risk terms, human
  requests, safety intents), but the confidence+grounding threshold tier
  is replaced with a static category allow-list. This isolates what the
  real system's confidence/similarity tier actually buys over a fixed
  "these categories are safe" policy -- a clean ablation, not just a
  weaker copy of the real thing.
"""
from src.classify.heuristic import guess_intent
from src.escalate.decide import RISK_TERM_RE, SAFETY_INTENTS, _is_genuine_human_request
from src.retrieve.index import RetrievalIndex

# Deliberately conservative: only categories where a wrong auto-handle is
# low-stakes. account_access is a SAFETY_INTENT (handled below regardless
# of this set); refund/billing and product/warranty are left off since a
# naive verbatim reply could state a wrong amount or policy.
SIMPLE_AUTO_HANDLE_INTENTS = {"delivery_delay", "order_cancellation", "praise_gratitude"}


def simple_classify(message: str) -> str:
    label, _, _ = guess_intent(message)
    return label


def simple_reply(message: str, index: RetrievalIndex) -> str:
    top = index.query(message, k=1)
    if len(top) == 0:
        return "Thanks for reaching out -- we're looking into this."
    return top["company_text"].iloc[0]


def simple_decide(message: str, intent: str) -> tuple[str, str]:
    if RISK_TERM_RE.search(message):
        return "escalate", "guardrail: risk term"
    if _is_genuine_human_request(message):
        return "escalate", "guardrail: explicit human request"
    if intent in SAFETY_INTENTS:
        return "escalate", f"guardrail: intent={intent} always human-handled"
    if intent in SIMPLE_AUTO_HANDLE_INTENTS:
        return "auto_handle", f"intent={intent} is on the simple allow-list"
    return "escalate", f"intent={intent} not on the simple allow-list"
