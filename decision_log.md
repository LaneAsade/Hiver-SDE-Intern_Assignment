# Decision Log

Non-obvious decisions made building this, with reasoning. Several were
driven by checking the data rather than by prior intuition — those are
marked with what the check actually showed.

1. **Picked AmazonHelp over higher-profile brands.** Highest volume
   (169,840 outbound tweets) *and* the lowest rate of handing the
   customer off to another channel — 11.3%, vs. 51% for Uber_Support,
   61% for AppleSupport, 81% for TMobileHelp. A brand that mostly
   replies "please DM us" gives almost nothing to ground a generated
   reply in, which would have quietly undermined the whole
   retrieval-grounding premise.

2. **Broadened the "canned reply" detector before trusting it.** The
   first version only matched literal "DM"/"direct message" and made
   AmazonHelp look artificially clean. Reading actual replies showed it
   hands off via *"reach out to our support team here: [link]"* — same
   behavior, different words. Rates above are from the corrected
   version; the initial numbers were wrong in a flattering direction.

3. **English-only, stated as scope rather than discovered late.**
   langdetect on 3,000 sampled messages: ~74.5% English, the rest mostly
   Japanese/Spanish/French/German. Handling those well needs per-language
   grounding corpora and a judge that can read them — out of scope for a
   take-home, but it means the headline numbers describe ~three-quarters
   of real traffic, which belongs in the report, not a footnote.

4. **Split `delivery_delay` from `delivery_discrepancy`.** They look
   like one category but demand different responses: "hasn't arrived
   yet" wants patience-and-tracking, "tracking says delivered but I
   don't have it" wants an investigation. Collapsing them would let the
   agent give the reassuring answer to the case that needs escalation.

5. **Gave `praise_gratitude` its own category.** A real share of inbound
   volume isn't a request at all. Without this, thank-you tweets get
   pushed through classify→retrieve→escalate machinery built for
   problems, and either waste a human's time or get an absurdly
   over-engineered reply.

6. **Used clustering to *bootstrap* the taxonomy, not define it.**
   TF-IDF + KMeans (k=12) over 4,000 English messages, then hand-read
   and merged/split. Raw clusters were noisy — short tweets plus generic
   words ("thanks", "just") don't separate cleanly — so shipping the
   cluster assignments as categories would have encoded that noise as
   structure.

7. **Cost-tiered the models deliberately.** Classification is a bounded
   10-way task → `gemini-2.5-flash-lite` ($0.10/$0.40 per 1M tokens).
   Drafting benefits from generation quality → `gemini-2.5-flash`
   ($0.30/$2.50). Paying Flash rates for a 10-way label would be pure
   waste at eval-set scale.

8. **Built four cost controls into the LLM client up front**: disk cache
   keyed by (model, prompt) hash, a configurable rate limiter,
   retry-with-backoff on 429s, and a hard `max_calls` circuit breaker.
   The cache matters most — iterating on prompts re-runs for free, so a
   full eval run is paid for once, not once per debugging pass.

9. **Deferred Gemini context caching and the Batch API.** Both are real
   further savings (~90% off a reused prompt prefix, 50% off batch), and
   the golden-set run is a textbook batch workload. Deferred because
   optimizing spend on a pipeline that isn't yet verified correct
   optimizes the wrong thing first.

10. **Made the LLM client injectable, with a `MockClient`.** Every
    component takes a client rather than constructing one. This is why
    the full pipeline could be dry-run end-to-end (`--mock`) and 51 tests
    pass with no API key and no network — prompt construction, response
    parsing, and all escalation branches are verified independently of
    whether the API behaves.

11. **Excluded golden-set candidates from the retrieval index before
    scoring their similarity.** Otherwise each candidate retrieves itself
    at ~1.0 and every grounding number is inflated. Their tweet IDs are
    written to `eval/golden_tweet_ids.txt` so the exclusion is
    enforceable later, not just done once.

12. **Kept "scam" as an escalation trigger after checking, despite a
    clear false positive.** *"Amazon Prime is a SCAM! ... my order is not
    here"* is hyperbole. But ~275 real messages containing "scam" were
    mostly genuine phishing-verification questions ("is this email a
    scam?") or real billing disputes. Escalating a hyperbolic complaint
    is cheap; missing a real fraud signal isn't. Locked in with a
    regression test so it can't be "fixed" later by someone who only
    sees the bad example.

13. **Fixed the "real person" false positive narrowly, not by rewriting
    the pattern.** ~92% of ~35 real matches were genuine escalation
    requests, so the phrase itself is a good signal. The fix suppresses
    it only when positive sentiment appears earlier in the message
    ("Thanks… it seemed as if I was talking to some real person").
    Rewriting the pattern would have cost recall on the 92%.

14. **Handled negation asymmetrically.** Negated praise ("Not
    impressed") flips to `service_complaint_general`; negated complaint
    ("not terrible") is only suppressed, never flipped to praise —
    sarcasm and understatement make that inference unreliable, and the
    error is costlier in that direction (auto-handling an unhappy
    customer as if they were happy).

15. **Excluded `other` from the trivial baseline's majority vote.** With
    it included the baseline predicted `other` — the heuristic's own
    "couldn't classify" bucket, making the floor a measure of the
    heuristic's blind spot rather than a real majority class. Spot-check
    of the corrected result (`praise_gratitude`) found it partly inflated
    by polite filler ("thanks, would appreciate delivery today") and
    sarcasm ("thanks captain obvious") — documented rather than chased,
    since a trivial baseline only needs to be a real floor.

16. **Made the simple baseline an ablation, not a weaker clone.** It
    shares the real system's guardrails but replaces the
    confidence+similarity tier with a static category allow-list. That
    isolates what the confidence tier actually buys, instead of
    comparing against a strawman that fails for unrelated reasons.

17. **Made the golden-set validator fail loudly by default.** A typo'd
    intent or blank action silently produces a wrong metric, which is
    worse than a missing one because nobody questions it. `--lenient`
    exists for partial labeling, but it's opt-in.

18. **Made model selection an env-var override, after it broke.**
    `gemini-2.5-flash-lite` and `gemini-2.5-flash` (decision #7) were
    hardcoded on the reasoning that they were the cheapest current
    options. They then got blocked for new API keys mid-project — Google
    is retiring the whole 2.5 series no earlier than October 16, 2026,
    and new accounts are cut off ahead of that date. Switched to
    `gemini-3.1-flash-lite` / `gemini-3.6-flash` (still-active, no
    deprecation notice, per Google's own pricing page) and made both
    read from `HIVER_CLASSIFY_MODEL` / `HIVER_DRAFT_MODEL` env vars with
    those as fallback defaults. The lesson generalizes: anything this
    close to a fast-moving vendor's model lifecycle should be
    configuration, not a hardcoded constant, from the start — this is
    the second time in the project a hardcoded assumption about "current"
    state needed correcting (see #2, the canned-reply detector).
