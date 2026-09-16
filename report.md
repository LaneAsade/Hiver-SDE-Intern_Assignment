# AmazonHelp Support Agent — Report

> **⚠ DRAFT — CONTAINS `<<<FILL>>>` PLACEHOLDERS.**
> Every `<<<FILL>>>` marks a number or example that must come from a real
> run. Search for `<<<` before submitting; if any remain, the report is
> not finished. Nothing in this file is an estimated or invented result.

---

## 1. Problem framing

### The brand
AmazonHelp, chosen over higher-profile candidates on two measured
criteria: highest outbound volume in the dataset (169,840 tweets), and
the lowest rate of deflecting customers to another channel — 11.3%,
against 51% for Uber_Support, 61% for AppleSupport, and 81% for
TMobileHelp. That second number decided it. The agent's core premise is
grounding replies in how the brand historically resolved issues, and a
brand whose replies are mostly "please DM us" offers almost nothing to
ground in.

### What "good" means here
Not "high accuracy." Support automation has asymmetric costs, so the
target is stated as a set of constraints rather than one metric:

1. **Never auto-handle something that needed a human.** Wrongly
   escalating a simple question wastes a few minutes of agent time.
   Wrongly auto-handling a fraud report, a locked account, or a legal
   threat produces an angry customer and a public tweet. Escalation
   **recall** is the primary metric; accuracy is secondary.
2. **Never invent policy.** A reply that states a refund window, a
   delivery date, or an eligibility rule not present in the retrieved
   precedent is a failure even if it reads perfectly and the customer is
   satisfied. Judged as `groundedness`.
3. **Automate enough to matter.** A system that escalates everything is
   perfectly safe and worthless — which is exactly why the trivial
   baseline is defined that way. Auto-handle rate must be meaningfully
   above zero at acceptable recall.
4. **Every decision is explainable.** Each auto-handle/escalate carries a
   stated reason, so a human reviewer can audit the call without
   re-deriving it. This is why the escalation logic is rule-based with an
   LLM-confidence tier, rather than asking a model "should this
   escalate?"

### What I chose not to build, and why
- **Multilingual support.** ~25.5% of the corpus is non-English (mostly
  Japanese, Spanish, French, German, per langdetect on a 3,000-message
  sample). Doing it properly needs per-language grounding corpora and a
  judge that can read them. Excluded deliberately, and it means every
  number here describes roughly three-quarters of real traffic.
- **Fine-tuning.** Prompting plus retrieval is testable in a week and
  its failures are legible. A fine-tuned classifier would probably beat
  the prompted one on intent accuracy, but the assignment weights proof
  over performance, and a fine-tune's errors are much harder to explain.
- **Multi-turn conversation.** Threads are reconstructed (the code
  handles chains up to depth 6) but the agent classifies and replies to a
  single message. Full dialogue state is a different, larger problem.
- **A live Twitter integration.** Offline evaluation only.
- **Multi-label intents.** The taxonomy is single-label per the
  assignment's framing, but 17 of 203 golden candidates trip more than
  one category. This is a real, known limitation rather than an
  oversight — see §5.
- **Confirmed resolution outcomes.** The dataset has no ground-truth
  signal that an issue was actually solved. "How the brand historically
  resolved similar issues" is operationalized as "what the brand replied,"
  which is a proxy, not an outcome.

### The taxonomy
Ten categories derived from real messages by open-coding a sample,
cross-checked against a TF-IDF + KMeans pass (k=12) over 4,000 English
messages. Clusters bootstrapped the categories but did not define them —
raw clusters were noisy, since short tweets and generic tokens
("thanks", "just") don't separate cleanly under TF-IDF. Full definitions
with examples: `taxonomy.md`.

Two boundary decisions worth defending:
- `delivery_delay` vs. `delivery_discrepancy` are split because "it
  hasn't arrived yet" and "tracking says delivered but I don't have it"
  demand different responses — reassurance versus investigation.
  Collapsing them would let the agent give the soothing answer to the
  case that needs a human.
- `praise_gratitude` exists so genuine thank-you tweets get a short
  acknowledgment instead of being pushed through problem-solving
  machinery.

---

## 2. Method

Pipeline: **classify → retrieve → draft → decide**.

| Stage | Implementation |
|---|---|
| Classify | Prompted LLM, 10-way taxonomy + few-shot, returns intent and confidence. `gemini-3.1-flash-lite` — a bounded 10-way task doesn't warrant a larger model. |
| Retrieve | Top-k similar historical (customer message → real reply) pairs from a 168,611-pair corpus. TF-IDF cosine, auto-upgrading to sentence-transformers embeddings when installed. |
| Draft | Prompted LLM conditioned on retrieved pairs, instructed to state no fact unsupported by them. `gemini-3.6-flash`. |
| Decide | Two tiers: deterministic guardrails (risk terms, explicit human requests, always-escalate intents) checked first, then a confidence × retrieval-similarity threshold. Always emits a reason. |

Model names are read from environment variables with the above as
defaults, not hardcoded — the original choices (`gemini-2.5-flash-lite` /
`gemini-2.5-flash`) were blocked for new API keys partway through this
project, ahead of Google's announced retirement of the whole 2.5 series.
Decision log #18.

**Baselines.** Both are full alternate pipelines run through the
identical harness, not per-metric reference points.
- *Trivial*: majority-class intent, one canned reply, always escalate.
  The 0%-automation floor.
- *Simple*: keyword-heuristic intent, verbatim nearest-neighbour
  historical reply (no generation), guardrails plus a static category
  allow-list. Deliberately an **ablation** rather than a strawman: it
  shares the real system's guardrails and differs only in the
  confidence/similarity tier and generation, so the comparison isolates
  what those two things actually buy.

**Golden set.** 203 hand-labelled examples. Stratified across the
taxonomy with rare and high-stakes intents deliberately oversampled,
plus a targeted supplement of guardrail-triggering messages so the
guardrail tier gets real coverage instead of relying on chance. Sampling
and labelling procedure: `eval/labeling_guidelines.md`.

**Leakage control.** Golden-set examples are excluded from the retrieval
corpus before the index is built; their IDs are recorded in
`eval/golden_tweet_ids.txt` so the exclusion is enforceable on every
subsequent run, not done once and forgotten. Without this, each example
retrieves itself at ~1.0 similarity and every grounding number is
inflated.

---

## 3. Results

> `<<<FILL: paste the table from eval/results_summary.csv>>>`

| System | n | Intent acc. | Macro-F1 | Escal. precision | Escal. recall | Auto-handle rate | Missed escalations |
|---|---|---|---|---|---|---|---|
| Trivial | `<<<FILL>>>` | | | | | | |
| Simple | `<<<FILL>>>` | | | | | | |
| **Agent** | `<<<FILL>>>` | | | | | | |

Read **escalation recall** first. The trivial baseline achieves recall
1.00 by escalating everything at a 0% auto-handle rate — the correct way
to read the table is what recall survives as automation rises.

**Per-intent escalation recall:** `<<<FILL from 05_score_all.py output>>>`
Pooled recall can conceal a category that was missed entirely, which is
why this breakdown is reported alongside.

### Reply quality (LLM judge)
> `<<<FILL: per-dimension means from eval/judge_scores.csv>>>`

| Dimension | Trivial | Simple | Agent |
|---|---|---|---|
| Groundedness | | | |
| Helpfulness | | | |
| Tone | | | |
| Conciseness | | | |

### Does the judge agree with a human?
`<<<FILL from 06_run_judge.py --agree>>>` — quadratic-weighted kappa,
Spearman, within-1-point rate, and judge-minus-human mean bias, across
`<<<FILL>>>` examples I scored by hand, blind to the judge's scores.

Weighted kappa is the headline because these are ordinal 1–5 ratings:
being off by one should cost far less than being off by three. Spearman
is reported alongside because kappa degrades when a rater's variance is
low. A positive mean bias is the signature of self-preference bias.

---

## 4. Failure analysis

> `<<<FILL: derive the top 5 from eval/errors.csv after a real run.>>>`
> The candidates below are failures **already observed** while building
> and inspecting the data — each is real, with a real example — but they
> must be confirmed and ranked against actual error counts before being
> presented as the top 5.

**Candidate 1 — Keyword guardrails can't read intent behind a word.**
*"Thanks for the freakishly prompt assistance. It seemed as if I was
talking to some real person on customer care"* tripped the
human-request guardrail on the literal phrase "real person," despite
being praise. Hypothesis: guardrails match surface forms, not speech
acts. Partially fixed by suppressing the match when positive sentiment
precedes it — chosen over rewriting the pattern because ~92% of ~35 real
matches were genuine escalation requests, so the phrase is a good signal.

**Candidate 2 — Nearest-neighbour retrieval returns confidently
irrelevant replies.** For *"my three out of 4 items have not been
delivered but it was marked as delivered,"* the simple baseline's
verbatim nearest-neighbour reply was *"I'm unable to comprehend your
concern, could you please elaborate."* Hypothesis: lexical similarity
retrieves a message with shared vocabulary but a different underlying
problem; without generation there's no step that notices the mismatch.
This is direct evidence for why retrieval-without-generation is the right
ablation to run.

**Candidate 3 — Multi-intent messages have no correct single label.**
17 of 203 golden candidates trip more than one category (e.g. *"cancel my
order and get a refund"* → `order_cancellation` + `refund_billing_payment`).
Hypothesis: forced single-label classification will show systematically
depressed accuracy on exactly these rows, and the error will look like
model weakness when it's actually taxonomy design.

**Candidate 4 — Sentiment keywords invert under negation.** *"Not
impressed… money down the drain as unusable"* initially classified as
`praise_gratitude` via the token "impressed." Fixed asymmetrically:
negated praise now flips to complaint, but negated complaint is only
suppressed, never flipped to praise — sarcasm and understatement make
that inference unreliable, and the error is costlier in that direction.

**Candidate 5 — `<<<FILL from real errors.csv>>>`** Likely candidates to
check first: hallucinated specifics under weak grounding; boundary cases
sitting just either side of the confidence/similarity thresholds; and
very short context-free messages ("??", "any update?").

---

## 5. What is misleading about my headline number?

Nine reasons, ordered roughly by how much damage each does.

1. **The golden set may be partly circular.** The `suggested_intent`
   column shown to the labeller is produced by the same keyword heuristic
   the *simple baseline* uses. Every rubber-stamped suggestion inflates
   that baseline toward a perfect score and hollows out the comparison.
   This is not hypothetical: dry-running the scorer with gold labels set
   equal to the suggestions returned accuracy 1.000 / macro-F1 1.000 for
   the simple baseline. Mitigated by labelling instructions requiring
   independent judgement first, but the residual risk cannot be measured
   from within the labelled set itself — it would need a second,
   suggestion-free annotator.
2. **n = 203, so the confidence intervals are wide.** A per-intent cell
   may hold only 14–29 examples, and escalation recall for a single
   intent can rest on a handful. Differences of a few points between
   systems are probably noise.
3. **The set is deliberately not representative.** Rare and high-stakes
   intents are oversampled and guardrail-triggering messages are
   supplemented in. This is correct for stress-testing escalation and
   wrong for estimating production performance — the automation rate here
   should not be read as the rate in deployment.
4. **Single annotator, no inter-annotator agreement.** All labels come
   from one person with no second rater, so there is no measure of how
   reproducible the "ground truth" is. Ambiguous categories
   (`service_complaint_general` vs. `other`) are where this bites hardest.
5. **"Resolution" is a proxy.** Grounding uses what AmazonHelp *replied*,
   not what actually resolved anything. The dataset contains no outcome
   signal. A reply that mirrors historical practice may be mirroring
   historically ineffective practice.
6. **The judge may prefer its own family's output.** Judge and drafter
   are both Gemini. Using different tiers is a partial mitigation, not a
   fix. The human-agreement number in §3 is the only real check, and it
   rests on ~40 examples scored by one person.
7. **English-only.** ~25.5% of real traffic is excluded. The headline
   describes about three-quarters of the problem.
8. **Thresholds are not fitted.** `CONF_THRESHOLD` and `SIM_THRESHOLD`
   are reasonable starting values, not values chosen from a
   precision/recall curve. If they're later tuned on this same golden
   set, the reported numbers become optimistic — tuning and evaluating on
   one set is overfitting, and would need a held-out split to avoid.
9. **Retrieval is lexical, not semantic.** TF-IDF matches shared
   vocabulary. Grounding quality is therefore bounded by word overlap
   rather than meaning, which likely *understates* what this architecture
   achieves with proper embeddings — the one item on this list that
   probably biases the number downward.

---

## 6. What I'd do with one more week

1. **A second annotator on ~50 examples**, without seeing the
   suggestions, to produce a real inter-annotator agreement number and
   quantify how much of item (1) above is actually present. This is the
   single highest-value item: it puts an error bar on the ground truth
   everything else is measured against.
2. **Swap TF-IDF for sentence-transformers embeddings** and re-run
   unchanged. The harness already supports this via one constructor
   argument, so it's a clean A/B on grounding quality.
3. **Tune the escalation thresholds properly** on a held-out split,
   producing a precision/recall curve rather than two hand-picked
   constants — and report the operating point chosen, with its cost
   justification.
4. **Enable Gemini context caching and the Batch API.** ~90% off the
   reused prompt prefix and 50% off batch respectively. Deliberately
   deferred until the pipeline was verified correct, since optimizing
   spend on unverified code optimizes the wrong thing.
5. **Multi-label classification for the 17 multi-intent rows**, or an
   explicit "route to the higher-stakes intent" rule, with a measurement
   of how much accuracy the single-label constraint currently costs.
6. **A time-based train/test split** (train on earlier tweets, evaluate
   on later ones) to test whether grounding degrades as brand policy
   drifts — currently untested, and the corpus spans a multi-year window.

---

## Credits
- Dataset: Customer Support on Twitter (Kaggle,
  `thoughtvector/customer-support-on-twitter`).
- Filtering channel-redirect replies as uninformative noise, and the
  time-based split in §6: Hardalov et al., *Towards Automated Customer
  Support*.
- LLM-as-judge framing and judge–human agreement validation: Zheng et
  al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*.
- Code written with AI assistance; all design decisions, the empirical
  checks behind them, and the labelling are documented in
  `decision_log.md`.
