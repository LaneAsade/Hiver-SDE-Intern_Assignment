"""
trivial.py

The trivial baseline: the floor every other system has to beat.
- Intent: always predicts the single most common intent (majority
  class), estimated from the heuristic bucketer's distribution over a
  large sample of the corpus, excluding the golden-set candidates.
- Reply: one fixed canned message, regardless of what was asked.
- Escalation: always escalates -- the 0%-automation reference point.
  (The other useful reference point, always-auto-handle, isn't built as
  a full baseline here since there's nothing to evaluate beyond "it
  would auto-handle even a fraud/legal-threat message" -- worth a line
  in the report rather than a whole second system.)
"""
from collections import Counter

from src.classify.heuristic import guess_intent

CANNED_REPLY = "Thanks for reaching out -- we're looking into this and will follow up as soon as we can."


def compute_majority_intent(texts) -> str:
    """Estimated from the heuristic over a sample of real messages,
    excluding "other" -- that's the heuristic's own "couldn't classify
    this" bucket, not a real intent a human labeler would pick as a
    single best guess. Counting it would make the "majority class"
    baseline degenerate into "the heuristic's blind spot" rather than a
    real answer.

    Caveat found by spot-checking the actual result (praise_gratitude,
    surprisingly, beating delivery/refund complaints): a real chunk of
    that count is "thanks"/"appreciate" used as polite filler in a
    message that's substantively about something else ("thanks, would
    appreciate delivery today"), or sarcastically ("thanks captain
    obvious"), inflating the count beyond genuine praise. This is a
    placeholder, not a precise estimate -- recompute from real gold
    labels once the golden set is labeled; a trivial baseline's fixed
    guess doesn't need to be exact, just a real floor to beat."""
    labels = [l for l in (guess_intent(t)[0] for t in texts) if l != "other"]
    if not labels:
        return "other"
    return Counter(labels).most_common(1)[0][0]


def trivial_classify(message: str, majority_intent: str) -> str:
    return majority_intent


def trivial_reply(message: str) -> str:
    return CANNED_REPLY


def trivial_decide(message: str) -> tuple[str, str]:
    return "escalate", "trivial baseline: everything escalates (0% automation floor)"
