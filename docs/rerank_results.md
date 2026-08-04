# Cross-encoder reranking (Stage 3, Member 3)

Second-stage reranking of retrieved FinQA evidence. Implementation:
[`src/retrieval/rerank.py`](../src/retrieval/rerank.py). Ablation driver:
[`scripts/eval_rerank.py`](../scripts/eval_rerank.py). Every number here is produced
offline — no LLM calls — so the whole sweep is free to re-run.

## Summary

Reranking a 20-candidate pool with `cross-encoder/ms-marco-MiniLM-L-6-v2` beats the tuned
hybrid retriever at **every** k on both splits. On the held-out test split: **+1.9pp
Recall@1, +2.2pp Recall@5, +1.2pp Full@10**.

Two findings beyond the headline:

- **The cross-encoder makes dense retrieval redundant.** A BM25-only first stage reranked
  to the same depth matches — slightly beats — a hybrid first stage, at **1/3.5 the
  latency** and with no Weaviate and no embedding model in the query path.
- **The gain is not uniform.** Reranking is much better at picking the right *table row*
  and marginally *worse* at ranking *text sentences* at k≥3. Since ~60% of FinQA gold
  evidence is table-only, the net is positive — but the mechanism matters for Stage 4.

## Method

BM25 and the dense bi-encoder score a question and a chunk independently: the chunk's
representation is fixed before the question is seen. That is what makes them cheap enough
to run over a corpus, and also what makes them weak here, because on a FinQA filing page
the distinguishing signal is usually an interaction — *which* of these near-identical
linearized table rows is the 2009 one. A cross-encoder reads `(question, chunk)` jointly
in a single forward pass and models exactly that, but is far too slow to score a corpus.

So: over-fetch `pool` candidates with the cheap first-stage retriever, re-score only those
with the cross-encoder in one batched call, and keep the top `k`.

## Results — dev split (tuning), n = 883

| retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Full@10 | ms/query |
|---|---|---|---|---|---|---|
| bm25 | 44.9% | 67.2% | 77.4% | 88.4% | 81.2% | 0 |
| dense | 48.2% | 71.7% | 82.1% | 91.7% | 85.5% | 57 |
| hybrid (a=0.6) | 50.1% | 74.7% | 82.8% | 92.0% | 86.1% | 53 |
| rerank (hybrid, pool=10) | 52.8% | 76.6% | 85.4% | 92.0% | 86.1% | 100 |
| rerank (hybrid, pool=20) | **52.8%** | 76.5% | 85.4% | 94.1% | 88.3% | 103 |
| rerank (hybrid, pool=30) | 52.8% | 76.5% | 85.3% | 94.5% | 89.1% | 108 |
| rerank (bm25, pool=10) | 52.7% | 74.8% | 83.2% | 88.4% | 81.2% | 19 |
| rerank (bm25, pool=20) | 52.6% | **76.7%** | **85.5%** | 94.1% | 88.4% | 34 |
| rerank (bm25, pool=30) | 52.6% | 76.7% | 85.4% | **94.5%** | **89.4%** | 46 |

The `hybrid (a=0.6)` row reproduces the alpha-sweep numbers in
[`data_schema.md`](data_schema.md) exactly (50.1 / 74.7 / 82.8), which confirms this
harness is measuring the same thing as the Stage 2 one.

## Results — test split (held out), n = 1147

| retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Full@10 | ms/query |
|---|---|---|---|---|---|---|
| bm25 | 43.4% | 66.5% | 77.1% | 87.1% | 78.8% | 0 |
| dense | 47.7% | 71.0% | 79.8% | 89.5% | 82.0% | 58 |
| hybrid (a=0.6) | 50.7% | 74.2% | 82.1% | 91.1% | 84.4% | 53 |
| **rerank (hybrid, pool=20)** | **52.6%** | **75.4%** | 84.3% | 92.0% | 85.6% | 112 |
| **rerank (bm25, pool=20)** | 52.5% | 75.3% | **84.5%** | **92.1%** | **86.0%** | **32** |

Gains over hybrid are smaller on test than on dev (+1.9 vs +2.7 at k=1) but hold in the
same direction at every k — which is what honest held-out generalization looks like.
These are also the repo's first test-split retrieval numbers; Stage 5 needs them anyway.

## `pool` is a hard ceiling on recall

Reranking can only reorder what the first stage hands it, so **Recall@k of the reranker is
bounded by Recall@pool of the first stage**. The dev table shows this exactly, not
approximately:

- `rerank (hybrid, pool=10)` has Recall@10 = 92.0% and Full@10 = 86.1% — *identical* to
  plain `hybrid`. With pool = k = 10 the reranker permutes the same ten candidates.
- `rerank (bm25, pool=10)` has Recall@10 = 88.4% and Full@10 = 81.2% — *identical* to
  plain `bm25`, for the same reason.

Practical reading: pool only needs to exceed the k you actually consume. Generation uses
k=5, where pool=10 is already saturated (85.4% vs 85.4% at pool=20). Deeper pools buy
recall only at deeper k, for ~5ms each. **`pool=20` is the default** — it covers k=5 with
headroom and still improves k=10.

## The cross-encoder makes dense retrieval redundant

On test, `rerank (bm25, pool=20)` matches or beats `rerank (hybrid, pool=20)` on every
metric while running **3.5× faster** (32ms vs 112ms):

| | Recall@1 | Recall@5 | Full@10 | ms/query | needs Weaviate? |
|---|---|---|---|---|---|
| rerank (hybrid, pool=20) | 52.6% | 84.3% | 85.6% | 112 | yes |
| rerank (bm25, pool=20) | 52.5% | 84.5% | 86.0% | 32 | **no** |

BM25 alone is a much weaker retriever than hybrid (43.4% vs 50.7% Recall@1), so this is
not "BM25 was fine all along" — it is that a first stage only has to get the gold evidence
*somewhere in the top 20*, and BM25 does that nearly as well as hybrid. Everything dense
retrieval was contributing to final ranking quality, the cross-encoder recovers on its own.

This is a real deployment option for the final system: BM25 + cross-encoder needs no
vector database, no embedding model at query time, and no `HYBRID_ALPHA` to tune.

## Where reranking helps and where it hurts

Recall@k split by gold evidence type (test split, micro-averaged over gold items):

| retriever | table@1 | text@1 | table@3 | text@3 | table@5 | text@5 | table@10 | text@10 |
|---|---|---|---|---|---|---|---|---|
| hybrid (a=0.6) | 36.0% | 50.3% | 64.0% | **70.8%** | 75.1% | **76.8%** | 88.0% | **86.3%** |
| rerank (hybrid, pool=20) | **37.0%** | **54.0%** | **66.5%** | 70.3% | 79.1% | 76.2% | 90.8% | 84.3% |
| rerank (bm25, pool=20) | 37.0% | 53.7% | 66.5% | 69.6% | **79.7%** | 75.7% | **91.4%** | 83.2% |

The reranker is consistently better on table rows (+4.0pp at k=5, +2.8pp at k=10) and
consistently slightly worse on text sentences at k≥3 (−0.6pp at k=5, −2.0pp at k=10). The
same asymmetry appears on dev, more sharply (text@3 71.0% → 67.9%).

The likely cause: `ms-marco-MiniLM-L-6-v2` is trained on web passages, and a linearized
FinQA table row (`the of october 31 2009 is october 31 2009 ;`) is far out of that
distribution — yet it is *exactly* the discrimination task the cross-encoder is good at
(matching the year/label in the question against the row). Prose sentences are already
in-domain for the first-stage retrievers, so there is less to gain and some to lose.

Because ~60% of FinQA gold evidence is table-only, the table gain dominates and total
recall improves. But an answer needing a text sentence is now slightly *more* likely to be
generated without its evidence — directly relevant to Stage 4's hallucination categories.

## Failure analysis

Questions whose Recall@1 changed, `rerank (hybrid, pool=20)` vs `hybrid (a=0.6)`:

| split | improved | regressed | net |
|---|---|---|---|
| dev (n=883) | 73 | 38 | **+35** |
| test (n=1147) | 110 | 74 | **+36** |

Roughly three questions fixed for every two broken. The aggregate gain is real, but it is
a *net* of substantial churn in both directions — worth stating plainly in the report.

**Fixed** — the reranker pulls the right table row over a superficially topical sentence:

```
[HOLX/2015/page_98.pdf-2]  recall@1 0% -> 100%
  q:        what is the expected growth rate in amortization expense from 2016 to 2017?
  gold:     ['table_1']
  baseline: ['text_0']       <- prose that merely mentions amortization
  rerank:   ['table_1']      <- the row actually holding the 2016/2017 figures
```

**Broken** — the mirror image, a text chunk displacing a correct table row:

```
[PNC/2013/page_62.pdf-2]  recall@1 100% -> 0%
  q:        in millions what was total residential mortgages balance for 2013 and 2012?
  gold:     ['table_6']
  baseline: ['table_6']
  rerank:   ['text_15']
```

```
[MAS/2018/page_35.pdf-3]  recall@1 100% -> 0%
  q:        what was the percentage growth in the operating profit as reported from 2017 to 2018
  gold:     ['table_1']
  baseline: ['table_1']
  rerank:   ['table_5']      <- right table, wrong row
```

The last one is the interesting residual: `table_5` over `table_1` means the cross-encoder
found the right table but the wrong row. Row-level disambiguation among near-duplicate
financial rows is where the remaining headroom is.

## Generation pilot (paired, n = 30) — underpowered, reported for honesty

Does better retrieval produce better *answers*? A paired run over the same 30 dev
questions, `zero_shot` / k=5 / `llama-3.3-70b-versatile`, 60 Groq calls total:

| | accuracy | Recall@5 (this subset) | avg latency |
|---|---|---|---|
| hybrid (a=0.6) | 15/30 (50.0%) | 82.2% | 1218 ms |
| rerank (pool=20) | 15/30 (50.0%) | 80.0% | 1554 ms |

**No measurable difference, and none should be expected at this n.** The margin of error
on 30 binary trials is roughly ±18pp against a retrieval gain of ~2.6pp; resolving that
needs several hundred paired questions, which the free tier's 100K tokens/day does not
allow. Note also that this particular subset is one where reranking's Recall@5 is *lower*
than hybrid's (80.0% vs 82.2%) — the reverse of the full-dev result — which is itself a
demonstration of how noisy n=30 is.

Retrieval differed on 25 of the 30 questions, so the calls were informative; 3 answers
went wrong→right and 3 went right→wrong.

The useful output is qualitative. This question flipped on **ordering alone**:

```
[CME/2017/page_97.pdf-5]  "increase in class a common stock issued and outstanding"
  gold:   ['table_2']    correct answer: 995
  hybrid: [table_2, table_3, table_5, table_4, table_6] -> 1095   WRONG
  rerank: [table_2, table_3, table_4, table_6, table_5] ->  995   RIGHT
```

The same five chunks, permuted, change the answer. That is direct evidence that
context *ordering* is a live lever on generation quality independent of retrieval quality
— and reordering costs nothing at inference time.

The regressions are the mirror image, and both are the type asymmetry above: reranking
dropped a gold chunk hybrid had kept (`table_1` in `C/2017/page_328.pdf-1`, `text_34` in
`DVN/2007/page_58.pdf-2`), and the answer degraded from `93.5` to `-2013` and `24.69` to
`27.54` respectively.

Reproduce with:

```bash
python scripts/compare_prompts.py --split dev --limit 30 --strategies zero_shot \
    --retriever hybrid --k 5 --out data/runs/paired_hybrid.jsonl
python scripts/compare_prompts.py --split dev --limit 30 --strategies zero_shot \
    --retriever rerank --pool 20 --k 5 --out data/runs/paired_rerank.jsonl
```

## Reproducing

```bash
python scripts/eval_retrieval.py --split test --retriever rerank --pool 20 --k 1 3 5 10
python scripts/eval_rerank.py --split dev  --pools 10 20 30 --bases hybrid bm25
python scripts/eval_rerank.py --split test --pools 20 --bases hybrid bm25
```

The first run downloads the cross-encoder (~90MB) from HuggingFace and caches it under
`~/.cache/huggingface`; every run after that is offline. No new pip install is needed —
`sentence-transformers` (already required for the bi-encoder) provides `CrossEncoder`.

Configuration (`.env`, all optional — defaults shown):

```
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_POOL=20
RERANK_BASE=hybrid
```

`rerank.retrieve(question, doc_id, k)` implements the shared retriever contract, so it
drops into `eval.metrics.evaluate_retriever` and into
`scripts/compare_prompts.py --retriever rerank` with no other change. Latency figures are
single-threaded CPU, one batched `predict` call per question.

## Limitations and next steps

- **Generation accuracy is not yet resolved.** Recall@k is a proxy. A paired pilot was run
  (see below) but is far too small to detect the effect.
- **The text regression is fixable.** The reranker currently *replaces* the first-stage
  ranking. Fusing the two scores (or reciprocal-rank fusion) should keep the table gain
  without giving up text recall. Cheapest high-value follow-up, and still free to evaluate.
- **Context ordering is worth revisiting.** The `CME/2017` case above shows a permutation
  of identical evidence flipping an answer. Re-sorting the selected chunks into document
  order before prompting was scoped out of Stage 3 to protect the token budget, but it is
  free at inference time and now has evidence behind it.
- **A finance- or table-adapted cross-encoder** would likely beat an MS MARCO web-passage
  model on linearized rows.
- **No statistical test.** The dev/test gap (+2.7 vs +1.9 at k=1) sits in the range where a
  paired bootstrap over questions would be worth reporting.
