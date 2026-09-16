# Golden Set — Labeling Guidelines

## How the candidates were sampled
203 candidates in `eval/golden_candidates.csv`, drawn from the 168,814
AmazonHelp (customer message → real reply) pairs:

- **Stratified across the 10-category taxonomy** (`taxonomy.md`), using a
  keyword heuristic (`src/classify/heuristic.py`) for initial bucketing —
  no real classifier existed yet to stratify by (no LLM API key in the
  dev sandbox this was built in). Rare/higher-stakes categories
  (`account_access`, `praise_gratitude`) are oversampled relative to
  their natural frequency on purpose.
- **+ a targeted supplement** of messages matching guardrail-trigger
  phrasing (legal/fraud terms, explicit requests for a human), regardless
  of heuristic bucket, so the escalation guardrail tier gets real
  evaluation coverage rather than relying on it showing up by chance.
- **English-only** by the project's stated scope — 9 non-English
  candidates were caught and dropped at build time.
- Duplicate customer messages were removed before sampling.
- `retrieval_top_similarity` is computed against an index that
  **excludes every selected candidate** — otherwise a candidate would
  retrieve itself at ~1.0 similarity. The company_tweet_ids of all 203
  candidates are logged in `eval/golden_tweet_ids.txt` and must stay
  excluded from whatever retrieval corpus the real pipeline uses.

## ⚠ Do not rubber-stamp `suggested_intent`

This is the single biggest threat to the validity of the whole
evaluation. The **simple baseline uses the exact same keyword heuristic**
that generated the `suggested_intent` column. So every time you accept a
suggestion without independently judging it, you inflate the simple
baseline's apparent accuracy toward 100% — and the headline comparison
between the baseline and the real agent becomes circular and meaningless.

This isn't hypothetical: dry-running the scoring script with gold labels
set equal to `suggested_intent` produced `intent_accuracy = 1.000,
macro_f1 = 1.000` for the simple baseline. A perfect score, entirely an
artifact of the labels mirroring the model being scored.

Read `customer_text` and decide for yourself first; treat
`suggested_intent` as a prompt to react to, not an answer to confirm.
Disagreeing with it reasonably often is a sign the process is working.
This belongs in the report's "what is misleading about my headline
number" section regardless of how carefully you label.

## How to label
Per row:
1. Read `customer_text`. Check `suggested_intent` and
   `heuristic_matched_on` (what keyword triggered the guess) — confirm or
   overwrite into `gold_intent`.
2. Decide `gold_action` (`auto_handle` / `escalate`) and write one line in
   `gold_action_reason` — your own reasoning, not a restatement of the
   suggestion. Rows with `guardrail_hit != none` are pre-filled
   `escalate`; override if you disagree (see known issue below — some of
   these are false positives).
3. Use `actual_historical_reply` as real ground truth for what AmazonHelp
   actually said, and optionally trim it into `must_mention_facts`.
4. Label from this sheet only — don't run the real pipeline on these
   examples first. Seeing the model's answer before labeling anchors your
   judgment on it.

## Heuristic fixes made after spot-checking real rows
Three issues were found by reading real candidate rows, checked against
the broader corpus (not just the one example that surfaced each one), and
handled differently based on what that check showed:

- **Guardrail false positive on "real person" — fixed.** *"Thanks... it
  seemed as if I was talking to some real person on customer care"* is
  praise, but matched the human-request guardrail on the literal phrase.
  Checked ~35 real matches of that phrasing first: ~92% were genuine
  escalation requests, so rather than rewriting the pattern (risking
  recall on those), `decide()` now only suppresses the match when
  positive-sentiment language ("thanks", "great", "impressed", "prompt",
  etc.) appears earlier in the same message. Verified directly against
  both the original false-positive text and a genuine request — the fix
  changes the first and preserves the second.
- **"SCAM" as a risk term — checked, deliberately NOT changed.** The
  original example (*"Amazon Prime is a SCAM! ... my order is not
  here"*) is hyperbole about a late delivery. But checking ~275 real
  messages containing "scam" showed most are either genuine phishing-
  verification questions ("is this email a scam?") or real billing
  disputes — not hyperbole. Removing it would trade a cheap false
  positive (escalating a hyperbolic complaint costs almost nothing) for a
  real chance of missing a genuine fraud/phishing signal. Kept as-is;
  locked in with a regression test so it isn't silently changed later.
- **Negated sentiment — fixed, asymmetrically.** *"Not impressed...
  unusable"* was mis-bucketed as `praise_gratitude` via the word
  "impressed." It's now flipped to `service_complaint_general` instead of
  just falling back to `other` — negated positive sentiment reliably
  reads as a complaint. The reverse (negated complaint → praise, e.g.
  "not terrible") is deliberately NOT done: that inference is far less
  reliable (sarcasm, understatement), so those still fall through to
  `other` rather than risk a confident wrong guess in the more costly
  direction.

## Still a real, open limitation
- **Multi-match rows are meant to be hard, not a bug** — `heuristic_
  matched_on` now shows which categories are competing (e.g.
  `order_cancellation:cancellation; refund_billing_payment:refund`), so
  you can see the ambiguity directly instead of reverse-engineering it.
  17 of 203 rows have this. The taxonomy is single-label by design (per
  the assignment's "classify into a small set of intents"); genuinely
  multi-intent messages are a real design tension worth naming in the
  report's "what I chose not to build" section, not something a better
  heuristic resolves.
- Keyword matching has a ceiling regardless of these fixes — it cannot
  understand sentiment or literal-vs-figurative usage in general, only
  the specific patterns checked above. The real fix for that is the LLM
  classifier itself once a live key is wired in; this heuristic only
  exists to stratify-sample before that's possible.
