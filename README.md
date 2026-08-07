# Financial RAG with Hallucination Detection

A question-answering system for company financial filings (FinQA), plus a detector that
checks whether its answers are actually supported by the evidence it read.

CS6120 Natural Language Processing course project.

**[→ Read the full report](docs/REPORT.md)**

---

## What it does

Given a question about a 10-K page, the system finds the relevant evidence, computes an
answer, then checks whether the answer follows from that evidence:

```
"what is the change in finished goods between 2002 and 2003?"
   │
   ├─ RETRIEVAL   86,421 chunks in Weaviate, filtered to this question's page (~29
   │              chunks), first stage returns 20, cross-encoder re-scores to 5
   ├─ GENERATION  llama-3.3-70b reads the 5 chunks   →  177.6
   └─ DETECTION   is 177.6 derivable from those 5 chunks?
```

The third step exists because FinQA answers are *calculated*, not quoted — only **3.4%** of
correct answers appear literally in their evidence — so a wrong answer looks exactly as
confident as a right one, and in real use there is no answer key to compare against.

## Results

**Retrieval** — held-out test split, n=1147:

| retriever | Recall@1 | Recall@5 | ms/query |
|---|---|---|---|
| BM25 | 43.4% | 77.1% | 0 |
| dense (Weaviate) | 47.7% | 79.8% | 58 |
| hybrid (Weaviate, alpha=0.6) | 50.7% | 82.1% | 53 |
| **rerank — hybrid first stage** | **52.6%** | **84.3%** | 112 |
| rerank — BM25 first stage | 52.5% | 84.5% | 32 |

The system runs `rerank` with a hybrid first stage. The two first stages land within 0.2pp
of each other once the cross-encoder runs.

**Generation** — 150 dev questions, all four strategies on the same questions:

| strategy | accuracy | latency |
|---|---|---|
| zero-shot | 60.0% | 1063 ms |
| few-shot | 60.7% | **346 ms** |
| chain-of-thought | 61.3% | 1713 ms |
| structured | 61.3% | 1265 ms |

Accuracy differences are not significant (all pairwise McNemar p ≥ 0.80). Latency differs
5× and tracks output length.

**Detection** — 560 labeled examples, 29% hallucinations:

| verifier | precision | recall | F1 |
|---|---|---|---|
| rule-based re-derivation | 0.38 | 0.11 | 0.17 |
| LLM judge — llama-3.3-70b | 0.49 | 0.35 | 0.41 |
| **LLM judge — claude-haiku-4.5** | **0.61** | **0.49** | **0.54** |
| *baseline: flag everything* | 0.29 | 1.00 | 0.45 |

Changing only the judging model moved F1 from 0.41 to 0.54 (paired McNemar p = 0.001).
Full analysis in the [report](docs/REPORT.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| variable | what it's for |
|---|---|
| `LLM_PROVIDER` | `groq` (free, 100K tokens/day) or `bedrock` (AWS, no cap) |
| `GROQ_API_KEY` | if using Groq |
| `AWS_REGION`, `BEDROCK_MODEL` | if using Bedrock — uses your normal AWS credentials |
| `WEAVIATE_URL`, `WEAVIATE_API_KEY` | vector store for dense/hybrid retrieval |

Both providers run the same `llama-3.3-70b`, so results stay comparable either way.

Build the data (takes a few minutes, no API calls):

```bash
python scripts/build_corpus.py --splits train dev test
python scripts/ingest_weaviate.py --recreate --split train dev test
```

## Running the experiments

```bash
# Retrieval — free, no LLM calls
python scripts/eval_retrieval.py --split test --retriever rerank --k 1 3 5 10
python scripts/eval_rerank.py --split dev --pools 10 20 30 --bases hybrid bm25

# Generation — one LLM call per question per strategy
python scripts/compare_prompts.py --split dev --limit 150 --retriever rerank --k 5

# Detection
python scripts/build_detection_set.py --runs data/runs/prompts_dev_rerank.jsonl  # free
python scripts/run_rule_verifier.py --input data/processed/detection_eval.jsonl  # free
python scripts/run_llm_judge.py --input data/processed/detection_eval.jsonl      # 1 call/row
python scripts/score_detection.py --input data/runs/judge_detection_eval.jsonl   # free

python -m pytest                                                # 109 tests
```

## Layout

```
src/
  data/        chunking + loading the processed corpus
  retrieval/   bm25 · dense · hybrid · rerank (all share one retrieve() contract)
  generation/  prompts · answer parsing · groq + bedrock clients
  detection/   llm_judge · rule_based · category taxonomy
  eval/        recall@k, per-type recall, retriever comparison
scripts/       one runnable experiment each
data/
  processed/   the corpus, the answer key, the labeled detection set
  runs/        experiment outputs — the evidence behind every number in the report
docs/          REPORT.md plus one detail doc per stage
```

## Docs

| | |
|---|---|
| **[REPORT.md](docs/REPORT.md)** | the full write-up — start here |
| [data_schema.md](docs/data_schema.md) | how raw FinQA becomes the processed corpus |
| [rerank_results.md](docs/rerank_results.md) | retrieval ablation, pool sweep, failure analysis |
| [prompt_comparison.md](docs/prompt_comparison.md) | the four prompt strategies head-to-head |
| [detection_dataset.md](docs/detection_dataset.md) | labeling rule, judge comparison, error analysis |

## Team

Swarali Degaonkar · Jaya Sriharshita Koneti · Kushal Pendekanti
