# ⚡ Semantic Cache & Cost-Aware Model Router

An engineering layer that sits between user prompts and LLM calls. It **reuses answers to
semantically equivalent questions** and **sends each remaining query to the cheapest model that
can handle it**, while logging the telemetry needed to prove the savings.

```
 prompt + tags
     │
     ▼
 ① validate & normalize ──► ② semantic cache (MiniLM embeddings, cosine, tag filter, qualifier guard)
                                  │ hit ≥ threshold                     │ miss
                                  ▼                                     ▼
                             CACHE_HIT ($0)              ③ router: heuristic | llm_classifier | hybrid
                                                            │ SIMPLE               │ COMPLEX
                                                            ▼                      ▼
                                                       CHEAP_MODEL ─fails──► STRONG_MODEL
                                                            │   validation        │
                                                            └─► store in cache ◄──┘
                                                                     │
                                                    ④ telemetry (SQLite) ─► Streamlit dashboard
```

## Quick start

```bash
git clone https://github.com/kmsamrudh577-lgtm/semantic-cache-model-router.git
cd semantic-cache-model-router            # always run from inside the project folder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8502   # dashboard on http://localhost:8502
python eval.py                            # benchmark + threshold sweep -> results/
```

> The dashboard uses port **8502** so it never clashes with another Streamlit app on the default
> port 8501. Check that the page title reads "Semantic Cache & Cost-Aware Model Router".

**No API key? No problem.** With no key set, the app and `eval.py` use a deterministic
**mock LLM** (same prompt → same answer, tokens and latency), so every number is reproducible.
The first run downloads `all-MiniLM-L6-v2` (~90 MB). Fully offline? Set `EMBEDDER=hashing`
(or `python eval.py --embedder hashing`) to use a built-in hashing embedder with no downloads.

### Live LLMs (optional)

Create a `.env` file (loaded automatically):

```bash
# OpenAI (default provider): gpt-4o-mini (cheap) / gpt-4o (strong)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# or Gemini through Google's OpenAI-compatible endpoint
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=...

# or a local Ollama server (free)
# LLM_PROVIDER=ollama

# optional overrides
# CHEAP_MODEL=gpt-4.1-mini
# STRONG_MODEL=gpt-4.1
# OPENAI_BASE_URL=https://api.groq.com/openai/v1   # any OpenAI-compatible server
```

Then turn off "Offline / mock LLM mode" in the sidebar, or run `python eval.py --live`.
If a live call fails (network, quota), the client falls back to the mock and records
`provider = mock-fallback` in telemetry.

## Files

| File | Responsibility |
|---|---|
| `pricing.py` | Model registry, per-1M-token prices, `compute_cost`, token estimate, savings helpers |
| `llm_client.py` | One client for OpenAI-compatible APIs (OpenAI / Gemini / Ollama / Groq) + deterministic mock |
| `cache.py` | Prompt/tag normalization, embedders, metadata-aware semantic cache, qualifier guard, SQLite persistence |
| `router.py` | Heuristic scorer, LLM classifier, hybrid strategy, response validation for escalation |
| `gateway.py` | Orchestrates cache → router → model → escalation → cache write → telemetry |
| `telemetry.py` | SQLite request log + KPI aggregation (hit rate, p95, savings) |
| `app.py` | Streamlit playground, telemetry dashboard, cache inspector |
| `eval.py` | 39-prompt benchmark, threshold × guard × strategy sweep, CSV/JSON reports |

Data lives in `data/` (`semantic_cache.db`, `telemetry.db`) and survives restarts until you
press **Clear cache** / **Reset logs**.

## How it works

**1. Normalize & validate.** Whitespace collapsed, lower-cased, trailing `?.!` stripped for
embedding; empty, symbol-only or >4000-char prompts are rejected. Tags (`topic`, `language`,
`context_tag`) are slugified into a canonical *context key* such as `language=en|topic=python`.

**2. Semantic cache.** Prompts are embedded with `all-MiniLM-L6-v2` (384-d, L2-normalized, so
cosine similarity = dot product, done in numpy — no vector DB needed at this scale). Only entries
with the **same context key** are compared, so a `python` answer can never be served for a
`javascript` question. The best match must reach `similarity_threshold`.

**Qualifier guard.** Embeddings are bad at small-but-critical differences: "Convert **5** km to
miles" vs "Convert **50** km to miles" score ~0.95. Before accepting a hit, the guard compares
numbers, units (with aliases like km→kilometers), negations, comparatives (top/bottom, min/max),
programming languages/data structures, and capitalized named entities. Any mismatch — or the
same unit pair in a different order (km→miles vs miles→km) — vetoes the hit. The reason is shown
in the UI and stored in telemetry (`blocked_reason`).

**3. Cost-aware routing** (cache misses only):
- `heuristic` — free. Scores length, reasoning verbs (prove, design, analyze, compare, debug…),
  code/math markers, constraints, multi-part questions; "what is / define / translate" lowers it.
  Score ≥ 0.5 → COMPLEX.
- `llm_classifier` — the cheap model labels the prompt SIMPLE/COMPLEX (costs ~60 tokens; included
  in actual cost).
- `hybrid` — heuristic first; only uncertain prompts (score 0.25–0.60) pay for the classifier.

**Escalation.** A cheap answer that is empty, too short for the prompt, hedging ("I'm not sure…")
or truncated is retried on the strong model. Both calls are billed, so telemetry shows the real
cost of a routing mistake.

**4. Telemetry.** One row per request: timestamp, query, matched query, similarity, route, model,
latency, tokens, actual cost, **baseline cost** (the same tokens priced on the strong model —
i.e. what you'd pay with no cache and no router), and cost saved. Cache hits cost $0 because
embeddings run locally.

> Mock-mode latency = real cache/embedding time + simulated model time (cheap ≈ 200–500 ms,
> strong ≈ 700–2000 ms). Toggle "Simulate real latency" to actually sleep for demos.

## Benchmark (`eval.py`)

39 prompts in order: 7 **seeds** (populate the cache), 10 **A — paraphrases** (should hit),
10 **B — near misses** (should *not* hit: Germany vs France, list vs string, 5 vs 50 km,
km→miles vs miles→km, Celsius vs Fahrenheit, benefits vs side effects, REST vs SOAP, same text
with a different tag, an added year), 12 **C — routing** (6 easy, 6 hard).

Each configuration starts from an empty in-memory cache. Default sweep: thresholds
`0.80 / 0.88 / 0.93` × guard `off / on` on the hybrid router, plus `heuristic` and
`llm_classifier` at the middle threshold.

| Column | Meaning |
|---|---|
| `paraphrase_recall_%` | A prompts served from cache (higher = more savings) |
| `near_miss_false_hits` / `false_hit_rate_%` | B prompts wrongly served from cache (lower = safer) |
| `cache_precision_%` | correct hits / all hits |
| `routing_acc_%` | C prompts sent to the expected tier |
| `cache/cheap/strong` | routing split across all 39 requests |
| `saved_%` | cost saved vs. sending everything to the strong model |

Outputs: `results/eval_summary.csv`, `results/eval_summary.json`, `results/eval_details.csv`
(per prompt, per config — useful for screenshots of individual failures).

```bash
python eval.py --thresholds 0.75 0.80 0.85 0.90 0.95   # finer sweep
python eval.py --guard off                              # raw embedding behaviour only
python eval.py --strategies heuristic                   # one strategy
python eval.py --embedder hashing                       # fully offline
```

## Demo script (5 minutes)

1. Ask *"What is the capital of France?"* → `CHEAP_MODEL` (simple lookup).
2. Ask *"what's the capital city of France"* → `CACHE_HIT`, similarity ≈ 0.9, latency drops to
   ~10–30 ms, saved 100%.
3. Ask *"What is the capital of Germany?"* → high similarity but **not** a hit (guard: entity
   mismatch, visible under *Routing details*). Turn the guard off and lower the threshold to
   0.80 to show the false hit you just prevented.
4. Ask the URL-shortener design question → `STRONG_MODEL`, reasons listed.
5. Switch to `llm_classifier` and ask a borderline prompt to show classifier cost/escalation.
6. Open **Telemetry**: hit rate, routing split, p95 latency, cumulative cost vs. baseline.
7. Run `python eval.py` and walk through the threshold trade-off table.

## Viva talking points

- **Why semantic, not exact-match caching?** Users rephrase; exact-match keys miss most repeats.
  Embeddings capture meaning, the threshold controls how loose "same question" is.
- **Threshold trade-off.** Lower → more hits and savings but more false hits (wrong answers
  served confidently). Higher → safe but fewer savings. `eval.py` quantifies this as recall vs.
  precision.
- **Why the guard?** Cosine similarity is dominated by shared words; it can't tell 5 from 50.
  A cheap symbolic check catches the most dangerous near misses without raising the threshold
  and losing paraphrase recall. Known limit: synonyms and implicit qualifiers it has no rule for.
- **Why metadata filtering?** Same words can mean different things in different contexts
  (language, tenant, product); partitioning by context key prevents cross-context leakage.
- **Router design.** Heuristics are free but brittle; an LLM classifier is smarter but costs
  tokens and latency on every request. Hybrid pays for the classifier only when uncertain.
  Escalation bounds the damage of a wrong SIMPLE call.
- **Baseline definition.** Savings are measured against "every request goes to the strong
  model", using the same token counts — a conservative, explainable comparison.
- **Production next steps:** TTL/invalidation for time-sensitive answers, per-user privacy
  scoping of the cache, ANN index (FAISS/HNSW) beyond ~100k entries, a learned router trained
  on telemetry, and quality evaluation of cached answers (LLM-as-judge).

## Customizing

- Prices / models: `pricing.py` (`PRICE_TABLE`, `PROVIDER_DEFAULTS`) or `CHEAP_MODEL`/`STRONG_MODEL`.
- Router sensitivity: `ModelRouter(threshold=0.5, band_low=0.25, band_high=0.6)` in `router.py`.
- Guard vocabulary: `_CRITICAL_WORDS` and `_ALIASES` in `cache.py`.
- Test prompts: the `SUITE` list in `eval.py`.
- Cache TTL: `SemanticCache(..., ttl_seconds=3600)`.
