# AmazonHelp Intent Taxonomy (draft v1)

Derived from open-coding real AmazonHelp customer messages plus a
TF-IDF + KMeans clustering pass (k=12) over a 4,000-message English-language
sample. See `notebooks/01_taxonomy_exploration.py` for the clustering code
and raw cluster output. This is a first draft: several clusters were noisy
grab-bags (short tweets + generic words like "thanks"/"just" don't separate
cleanly under TF-IDF), so the category boundaries below come from manually
reading examples, not the raw cluster assignments.

## Categories

1. **delivery_delay** — item hasn't arrived / tracking shows it's running
   late; no dispute about whether it's lost.
   > "i still have not received my package after waiting the rare 36 hours"

2. **delivery_discrepancy** — tracking/status says delivered but the
   customer disputes receiving it. Kept separate from delivery_delay
   because the correct response is an investigation, not "please wait."
   > "how come my package was delivered saying that I received it? This
   > is scary."

3. **order_cancellation** — customer wants to cancel, or an order was
   auto-cancelled and they want an explanation or reversal.
   > "Amazon finally admitted it couldn't process my order... I'm just
   > gonna cancel."

4. **refund_billing_payment** — refund status, wrong/failed charge, promo
   or cashback not credited.
   > "Bought FireTV... Haven't received promised Amazon Pay Cashback of
   > Rs. 500 yet!"

5. **account_access** — locked account, missing verification email,
   login trouble.
   > "you never sent me the email"

6. **product_defect_warranty** — item arrived damaged/defective, or a
   warranty / return-replacement policy dispute.
   > "Lenovo says they does not provide warranty 4 it" (re: an item sold
   > with an advertised 1yr warranty)

7. **app_website_technical** — site/app bug, frozen page, broken form;
   not an order-specific complaint.
   > "Is your site frozen? I've been waiting several minutes with that
   > orange circle going around."

8. **service_complaint_general** — frustration/venting about service
   quality without one specific actionable fact to resolve, often after
   a prior unresolved contact.
   > "Support sucks!... All chat support can say is 'Sorry.'"

9. **praise_gratitude** — not a request. Should never enter the full
   classify → retrieve → draft → escalate flow seriously: auto-handle
   with a brief acknowledgment, never escalate.
   > "Thanks for being an awesome customer and reaching out to us today"

10. **other** — doesn't fit cleanly above, or genuinely ambiguous /
    multi-intent. Default posture: escalate — no confident precedent to
    ground a reply in.

## Known weaknesses in this draft (for the report's "what I chose not to
build" section)
- Multilingual messages (~25% of the corpus: Japanese, Spanish, French,
  German, and others, per a 3,000-row langdetect sample) are excluded —
  this taxonomy and everything downstream is English-only by design, not
  by oversight.
- `service_complaint_general` and `other` will likely absorb mislabeled
  examples from the other 8 categories until the golden set forces
  tighter boundaries during labeling.
- Categories were read off a 4,000-message sample, not the full 168K
  corpus — worth re-checking cluster stability on a second sample before
  finalizing.
