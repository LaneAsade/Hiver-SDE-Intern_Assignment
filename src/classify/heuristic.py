"""
heuristic.py

Zero-dependency keyword heuristic for a *rough* intent guess. Used to
stratify-sample the golden-set candidate pool before any real classifier
has been run against real data (this dev sandbox has no LLM API key).
This is NOT the agent's actual classifier (see classify/classifier.py) --
it's intentionally crude, checked in priority order, and will get plenty
of things wrong, especially service_complaint_general vs. other and any
genuinely multi-intent message. That's fine for sampling; it is not fine
as a stand-in for real labels, which is why every suggestion here is
something a human confirms or corrects, never treated as ground truth.
"""
import re

# Order matters: more specific patterns are checked before generic ones,
# e.g. "says delivered" (discrepancy) before generic delay language.
_ORDER = [
    "delivery_discrepancy",
    "delivery_delay",
    "order_cancellation",
    "refund_billing_payment",
    "account_access",
    "product_defect_warranty",
    "app_website_technical",
    "praise_gratitude",
    "service_complaint_general",
]

_PATTERNS = {
    "delivery_discrepancy": re.compile(r"\b(says? delivered|marked as delivered|shows? delivered)\b", re.I),
    "delivery_delay": re.compile(
        r"\b(haven'?t received|still waiting|hasn'?t arrived|not arrived|"
        r"where is my (order|package)|tracking (says|shows))\b", re.I),
    "order_cancellation": re.compile(r"\b(cancel(led|ling|s)?|cancellation)\b", re.I),
    "refund_billing_payment": re.compile(
        r"\b(refund|charged|cashback|billing|overcharge|payment (failed|declined))\b", re.I),
    "account_access": re.compile(
        r"\b(locked (out|my account)|can'?t log ?in|login (issue|problem)|"
        r"password|verification email|reset my)\b", re.I),
    "product_defect_warranty": re.compile(r"\b(warranty|defective|damaged|broken|doesn'?t work|not working)\b", re.I),
    "app_website_technical": re.compile(
        r"\b(app (crash|froze|frozen|bug)|website (down|frozen|broken)|site (down|frozen)|error message)\b", re.I),
    "praise_gratitude": re.compile(r"\b(thanks?( you)?|appreciate|awesome|impressed|great (job|service|help))\b", re.I),
    "service_complaint_general": re.compile(r"\b(worst|pathetic|terrible|sucks|useless|disgusting|horrible)\b", re.I),
}


_NEGATION_RE = re.compile(r"\b(?:not|n't|never|no|hardly)\b", re.I)
_NEGATION_SENSITIVE = {"praise_gratitude", "service_complaint_general"}


def guess_intent(text: str):
    """Returns (primary_label, matched_terms, multi_match). 'other' if nothing hits.

    matched_terms are "label:term" strings, not bare words -- when more
    than one category matches, this makes it explicit which categories
    are competing instead of leaving it to be inferred from the word
    alone.

    Negation is handled asymmetrically on purpose: a negated praise term
    ("Not impressed") is flipped into a service_complaint_general signal,
    since negated positive sentiment reliably reads as a complaint. A
    negated complaint term ("not terrible") is only suppressed, not
    flipped to praise -- that inference is far less reliable (sarcasm,
    understatement), so it falls through to other patterns or "other"
    instead of confidently guessing wrong.

    multi_match=True means more than one category's pattern fired -- a
    useful signal on its own: these are good candidates for the golden set
    precisely because they're the ambiguous/multi-intent cases a real
    classifier will also struggle with.
    """
    matches = []
    for label in _ORDER:
        m = _PATTERNS[label].search(text)
        if not m:
            continue
        term = m.group(0)
        if label in _NEGATION_SENSITIVE:
            preceding = text[max(0, m.start() - 25) : m.start()]
            if _NEGATION_RE.search(preceding):
                if label == "praise_gratitude":
                    matches.append(("service_complaint_general", f"negated:{term}"))
                continue
        matches.append((label, term))
    if not matches:
        return "other", [], False
    primary_label = matches[0][0]
    return primary_label, [f"{label}:{term}" for label, term in matches], len(matches) > 1
