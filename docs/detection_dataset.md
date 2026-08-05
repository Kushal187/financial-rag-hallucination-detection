# Hallucination evaluation set (Stage 4, Member 3)

The labeled dataset we use to test the hallucination verifiers, plus the first results
for Member 2's LLM judge.

- Built by [`scripts/build_detection_set.py`](../scripts/build_detection_set.py)
- Dataset: `data/processed/detection_eval.jsonl` (40 rows)
- Scored with [`scripts/score_detection.py`](../scripts/score_detection.py)

## How we label the answers

We don't hand-label anything. We take answers the pipeline already generated and work out
the label from data we already have:

| gold evidence retrieved? | answer correct? | label |
|---|---|---|
| yes | yes | supported |
| yes | no | hallucinated |
| no | — | skip the row |

The skip matters. If retrieval never found the right chunk, the model may have read the
wrong chunk perfectly well and still given a wrong answer. That's a retrieval problem, not
a hallucination, so counting it would blame the detector for someone else's mistake.

"Correct" comes from `answers_match()`, which `compare_prompts.py` already ran. We just
read its `correct` field, so our labels can't disagree with the accuracy numbers the rest
of the project reports.

Refusals ("I cannot determine the answer...") count as supported — refusing is the safe
option, not a hallucination.

## What's in the dataset

Built from the 60 answers in `data/runs/paired_hybrid.jsonl` and `paired_rerank.jsonl`
(left over from the Stage 3 reranking work), at k=5 on the dev split.

```
Read 60 answers, kept 40
  correct                  21     -> supported
  wrong_despite_evidence   14     -> hallucinated
  abstention                5     -> supported
  incomplete_retrieval     20     -> skipped

supported 26   hallucinated 14
```

The 20 skipped rows are about a third of the data, which lines up with Full@5 on dev — a
third of FinQA questions never get all their evidence retrieved in the first place.

Each row carries its evidence text, so the file works on its own without rebuilding the
corpus.

## Results

The judge is `llama-3.3-70b-versatile`, same model that generated the answers.

| | all 40 rows | ignoring near-misses (33 rows) |
|---|---|---|
| precision | 0.80 | 0.78 |
| recall | 0.57 | **1.00** |
| F1 | 0.67 | **0.88** |
| accuracy | 0.80 | 0.94 |

## The main thing we found

The judge missed 6 hallucinations, which looks bad. But every single one is an answer
within 7% of the right value:

| id | judge said | model answered | right answer | off by |
|---|---|---|---|---|
| `LMT/2013/page_74.pdf-4` | supported | -2.0 | -1.9 | 5.3% |
| `PM/2017/page_38.pdf-1` | supported | 642 | 688 | 6.7% |
| `SPGI/2018/page_74.pdf-1` | supported | 1.107 | 1.157 | 4.3% |
| `PM/2017/page_38.pdf-3` | supported | 2.5% | 2.58% | 3.1% |

In all of these the model found the right evidence and did the right calculation, then
rounded differently. The judge says "you can work this out from the evidence", which is
true. Our label says wrong, because `answers_match` only allows 1%.

**7 of our 14 hallucinated rows are like this.** If we ignore them, the judge catches
every single real hallucination (7 out of 7).

So this isn't really the judge failing. It's that our labeling rule and the judge disagree
about whether a rounding error counts as a hallucination.

### Why the judge behaves this way

The judge's prompt tells it two things that conflict on these rows:

- "allow for rounding"
- flag `numeric_error` when "the arithmetic is wrong relative to the evidence"

A 3-7% gap fits both. The judge picks "supported" every time.

**The team should decide which one we mean**, because it changes F1 from 0.67 to 0.88.
Our suggestion: only count it as a hallucination if the answer isn't grounded in the
evidence at all. Being a bit off is a generation problem, and we already measure that with
accuracy — no need to count it twice.

### The two false alarms

Both were answers that were exactly right:

- `V/2008/page_17.pdf-1` — answer 127.4, correct answer 127.40. The judge redid the
  division and got it wrong itself.
- `MRO/2007/page_149.pdf-2` — a "what if costs rose the same amount next year" question.
  The judge said unsupported because the evidence has no 2008 figures, but projecting them
  is the whole question.

## Running it

```bash
# free
python scripts/build_detection_set.py \
    --runs data/runs/paired_hybrid.jsonl data/runs/paired_rerank.jsonl \
    --out data/processed/detection_eval.jsonl

# about 35 Groq calls (5 of the 40 are refusals, which skip the API)
python scripts/run_llm_judge.py --input data/processed/detection_eval.jsonl \
    --out data/runs/judge_detection_eval.jsonl

# free
python scripts/score_detection.py --input data/runs/judge_detection_eval.jsonl
python scripts/score_detection.py --input data/runs/judge_detection_eval.jsonl --skip-near-miss
```

All the results above cost 45 Groq calls (~45K tokens), under half the free tier's daily
limit.

## Problems with this

- **Only 40 rows, from 21 questions.** The two retrievers reuse the same questions, so the
  rows aren't independent. These numbers are a rough signal, not a solid result. Getting
  more needs ~200 more generation calls, which Stage 5 will produce anyway.
- **A lucky guess looks supported.** If the model gets the right number without really
  using the evidence, we label it supported. Probably rare.
- **The judge grades the same model that wrote the answers.** There's no setting to point
  it at a different one.
- **We report the judge's categories but can't check them.** We have no ground truth for
  which category each hallucination belongs to, so we can only say what the judge claimed:
  `numeric_error` 9, `unsupported_claim` 1, `abstention` 5, `supported` 25.
- **Member 1's rule-based verifier doesn't exist yet.** When it does, it can be scored on
  this same file for free. One warning for it: checking "is this number in the evidence?"
  won't work — only 3.4% of correct answers appear literally in their evidence, because
  FinQA answers are calculated. The numbers used in the calculation do appear (99.2%), so
  the rule has to redo the maths.
