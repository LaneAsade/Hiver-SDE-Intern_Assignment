# AmazonHelp Support Agent

An AI customer-support agent for AmazonHelp built on the Customer Support on Twitter dataset: classifies an incoming message into a data-derived intent taxonomy, drafts a reply grounded in how AmazonHelp has actually resolved similar issues, and decides auto-handle vs. escalate with a stated reason.

## Status
Honest accounting of what is verified vs. what isn't:

| Component | Status |
|---|---|
| Data loading, thread reconstruction, pairs corpus | Working — 168,814 pairs built from the real dataset |
| Intent taxonomy (10 categories) | Derived from clustering + manual reading (`taxonomy.md`) |
| Retrieval index (TF-IDF, auto-upgrades to embeddings) | Working |
| Classify → retrieve → draft → decide pipeline | Wiring verified end-to-end via `--mock`; **live Gemini calls untested** (no API key in the dev environment) |
| Trivial + simple baselines | Working — run against all 203 candidates |
| Escalation logic | Working, 51 unit tests |
| LLM-as-judge + human-agreement metrics | Code written and tested; **not yet run on real outputs** |
| Golden set | 203 candidates sampled and pre-filled; **gold labels not yet filled in** |

Two things block real headline numbers: (1) the golden set needs its
`gold_*` columns hand-labeled, (2) Ollama running locally.
Everything else runs now.

## Setup

```bash
pip install -r requirements.txt
# Ollama is used by default. Make sure it is running locally!
```

Place `twcs.csv` (Kaggle: `thoughtvector/customer-support-on-twitter`)
where `src/data_prep/load_data.py::RAW_PATH` points, or edit that
constant.

## Reproducing


> [!NOTE]
> **Windows PowerShell Users:** The `PYTHONPATH=. python ...` syntax shown below is for Linux/Mac bash.
> If you are on Windows using PowerShell, you must use `$env:PYTHONPATH="."; python ...` instead!
> For example: `$env:PYTHONPATH="."; python notebooks/04_run_agent.py`

All of the below run in well under 15 minutes on a subsample; none
require an API key.

```bash
# 1. Build the (customer message -> real reply) corpus  (~2 min)
python -m src.data_prep.load_data

# 2. Taxonomy exploration: clustering over a 4k English sample  (~1 min)
PYTHONPATH=. python notebooks/01_taxonomy_exploration.py

# 3. Build the golden-set candidate pool  (~3 min)
PYTHONPATH=. python notebooks/02_build_golden_candidates.py

# 4. Run both baselines against the candidates  (~2 min)
PYTHONPATH=. python notebooks/03_run_baselines.py

# 5. Full test suite  (~2 sec, no network)
python -m pytest tests/ -q

# 6. Pipeline dry run -- no key, no network, no quota spent
python -m src.pipeline --mock
```

With a key, live single-message run (cost-capped):

```bash
python -m src.pipeline --max-calls 5 --rpm 10
```

## The full evaluation run

Once the golden set is labeled and `OLLAMA_HOST` is set:

```bash
# Smoke-test the agent on 5 rows before committing to all 203
PYTHONPATH=. python notebooks/04_run_agent.py --limit 5

# Full agent run -- ~406 calls, resumable (cached), order of cents
PYTHONPATH=. python notebooks/04_run_agent.py

# Headline comparison table + per-intent breakdown + errors.csv
PYTHONPATH=. python notebooks/05_score_all.py

# LLM judge on reply quality
PYTHONPATH=. python notebooks/06_run_judge.py --run --limit 40

# Blind human-scoring sheet (deliberately excludes judge scores)
PYTHONPATH=. python notebooks/06_run_judge.py --sheet --n 40
#   ... score eval/human_scoring_sheet.csv by hand ...
PYTHONPATH=. python notebooks/06_run_judge.py --agree
```

Every script above accepts `--mock` for a zero-cost dry run.
`05_score_all.py` works without agent predictions too — it scores the
baselines alone so you can see the floor before spending anything.

## Cost controls

Built into `src/llm_client.py` because eval loops are where API credits
actually disappear:

- **Disk cache** keyed by `hash(model, prompt)` — an identical call is
  never re-sent. Re-running the pipeline while debugging costs nothing
  after the first pass.
- **Rate limiter** (`--rpm`) — free-tier limits are account- and
  model-specific and only visible in your AI Studio console, so this
  defaults conservatively rather than assuming a number.
- **Backoff** on 429s (5s → 40s).
- **`--max-calls` circuit breaker** — raises instead of silently
  spending past a set budget. Set it low while iterating on prompts.

Model tiering is deliberate: `llama3 (local)` for classification
(a bounded 10-way task), `llama3 (local)` for drafting. **Update, mid-
project**: the original defaults (`gemini-2.5-flash-lite` /
`gemini-2.5-flash`) got blocked for new API keys — Google is retiring the
whole 2.5 series no earlier than October 16, 2026, and new accounts are
already cut off ahead of that date. Both model constants now read from
`HIVER_CLASSIFY_MODEL` / `HIVER_DRAFT_MODEL` env vars with the new
defaults as fallback, specifically so the next deprecation is a one-line
override instead of a code change. If Google's own error message for a
blocked model names a specific replacement for your account, trust that
over these defaults.

Not yet implemented, both real further savings once the pipeline is
verified: Gemini context caching (~90% off the reused taxonomy/few-shot
prefix) and the Batch API (50% off, a natural fit for the offline
golden-set run).

## Golden set

`eval/golden_candidates.csv` — 203 candidates, stratified across the
taxonomy with rare/high-stakes intents deliberately oversampled, plus a
targeted supplement of guardrail-triggering messages. Non-English
candidates dropped per project scope. Similarity scores computed against
an index that **excludes the candidates themselves** (their IDs are in
`eval/golden_tweet_ids.txt`) to prevent retrieval leakage.

Sampling method, labeling instructions, and known heuristic limitations:
`eval/labeling_guidelines.md`.

Once labeled:

```bash
PYTHONPATH=. python -m src.eval.load_golden     # validates, then writes golden_set.jsonl
```

The validator fails loudly on unknown intents, invalid actions, and
escalations missing a reason — a typo'd label silently becomes a wrong
metric otherwise. `--lenient` writes only valid rows for partial
labeling.

## Baselines

- **Trivial** (`src/baselines/trivial.py`) — majority-class intent, one
  canned reply, always escalate. The 0%-automation floor.
- **Simple** (`src/baselines/simple.py`) — keyword-heuristic intent,
  verbatim nearest-neighbor historical reply (no generation), guardrails
  plus a static category allow-list. Deliberately an *ablation* of the
  real system: same guardrails, no confidence/similarity tier, so the
  comparison isolates what that tier buys.

Both run with no LLM and no training data. Current split on the 203
candidates: trivial escalates 203/203; simple escalates 140, auto-handles
63.

## Evaluation

- `src/eval/metrics.py` — accuracy, macro-F1, and escalation
  precision/recall **disaggregated by intent** (a pooled number can hide
  that every safety-relevant case was missed). Recall on `escalate` is
  the number to watch: a missed escalation is the costly error.
- `src/eval/judge.py` — LLM-as-judge, 4 dimensions (groundedness,
  helpfulness, tone, conciseness), 1–5.
- `judge_human_agreement()` — quadratic-weighted Cohen's kappa (ordinal
  scores: off-by-one should hurt far less than off-by-three) plus
  Spearman. Score ~30–50 examples by hand, blind to the judge's scores,
  before trusting any reply-quality number.

## Layout

```
src/
  data_prep/load_data.py     corpus + thread reconstruction
  classify/classifier.py     LLM intent classifier
  classify/heuristic.py      keyword bucketer (sampling + simple baseline only)
  retrieve/index.py          TF-IDF / embeddings retrieval
  generate/draft.py          grounded reply drafting
  escalate/decide.py         guardrails + threshold tiers
  eval/                      metrics, judge, golden-set loader
  baselines/                 trivial + simple
  llm_client.py              Gemini client, cache, rate limiting, mock
notebooks/                   reproducible scripts (not .ipynb)
tests/                       51 tests, no network required
```

## Credits

- Dataset: Customer Support on Twitter (Kaggle,
  `thoughtvector/customer-support-on-twitter`).
- The idea of filtering channel-redirect replies ("please DM us") as
  uninformative noise is from Hardalov et al., *Towards Automated
  Customer Support* — as is the time-based train/test split approach
  noted as future work below.
- LLM-as-judge framing and the practice of validating judge–human
  agreement: Zheng et al., *Judging LLM-as-a-Judge* (MT-Bench).

## Known limitations

- English-only (~74.5% of the corpus; the rest is mostly Japanese,
  Spanish, French, German).
- "Resolution" is a proxy — the dataset has no ground-truth signal that
  an issue was actually solved.
- The retrieval index currently uses TF-IDF; semantic matching would
  likely improve grounding (install `sentence-transformers` to switch
  automatically).
- The intent taxonomy is single-label, but real messages are sometimes
  genuinely multi-intent (17 of 203 candidates trip multiple categories).
- Escalation thresholds (`CONF_THRESHOLD`, `SIM_THRESHOLD`) are
  reasonable starting values, not fitted — they need a real
  precision/recall curve from the labeled golden set.
