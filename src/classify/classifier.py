"""
classifier.py

LLM-based intent classifier: builds a taxonomy + few-shot prompt and calls
an injected LLMClient for structured {intent, confidence} output. Prompt
construction and response parsing are fully unit-testable without a live
API key -- only classify()'s call to client.complete() needs one (or use
MockClient, see llm_client.py).
"""
import json
import re
from dataclasses import dataclass

from src.llm_client import LLMClient

# Mirrors taxonomy.md -- keep these in sync; taxonomy.md is the source of
# truth for *why* each category exists, this is the machine-readable copy.
TAXONOMY = {
    "delivery_delay": "Item hasn't arrived yet / tracking shows it's running late. No dispute about whether it's lost.",
    "delivery_discrepancy": "Tracking/status says delivered but the customer disputes receiving it.",
    "order_cancellation": "Customer wants to cancel, or an order was auto-cancelled and they want an explanation/reversal.",
    "refund_billing_payment": "Refund status, wrong/failed charge, promo or cashback not credited.",
    "account_access": "Locked account, missing verification email, login trouble.",
    "product_defect_warranty": "Item arrived damaged/defective, or a warranty/return-replacement policy dispute.",
    "app_website_technical": "Site/app bug, frozen page, broken form -- not order-specific.",
    "service_complaint_general": "Frustration/venting about service quality without one specific actionable fact.",
    "praise_gratitude": "Not a request -- thank-yous and compliments.",
    "other": "Doesn't fit cleanly above, or genuinely ambiguous/multi-intent.",
}

FEW_SHOT = [
    ("i still have not received my package after waiting the rare 36 hours.", "delivery_delay"),
    ("how come my package was delivered saying that I recieved it? This is scary.", "delivery_discrepancy"),
    ("Amazon finally admitted it couldn't process my order. I'm just gonna cancel.", "order_cancellation"),
    ("Haven't received promised Amazon Pay Cashback of Rs. 500 yet!", "refund_billing_payment"),
    ("you never sent me the email.", "account_access"),
    ("Lenovo says they does not provide warranty 4 it", "product_defect_warranty"),
    ("Is your site frozen? I've been waiting several minutes with that orange circle going around.", "app_website_technical"),
    ("Support sucks! All chat support can say is 'Sorry.'", "service_complaint_general"),
    ("Thanks for being an awesome customer and reaching out to us today.", "praise_gratitude"),
]


def build_prompt(message: str) -> str:
    taxonomy_block = "\n".join(f"- {k}: {v}" for k, v in TAXONOMY.items())
    examples_block = "\n".join(f'Message: "{m}"\nIntent: {i}' for m, i in FEW_SHOT)
    return f"""Classify the customer message into exactly one intent from this taxonomy:

{taxonomy_block}

Examples:
{examples_block}

Message: "{message}"

Respond with ONLY a JSON object, no other text: {{"intent": "<one of the labels above>", "confidence": <0.0-1.0>}}"""


@dataclass
class Classification:
    intent: str
    confidence: float
    raw: str


def parse_response(raw_text: str) -> Classification:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return Classification("other", 0.0, raw_text)
    try:
        obj = json.loads(match.group(0))
        intent = obj.get("intent", "other")
        if intent not in TAXONOMY:
            intent = "other"
        confidence = float(obj.get("confidence", 0.5))
        confidence = min(max(confidence, 0.0), 1.0)
        return Classification(intent, confidence, raw_text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return Classification("other", 0.0, raw_text)


def classify(message: str, client: LLMClient) -> Classification:
    raw = client.complete(build_prompt(message), max_tokens=200)
    return parse_response(raw)
