# Detecting Hallucinations in Financial Question Answering

CS6120 Natural Language Processing — project report.

This is the full write-up: what we built, what we measured, and what we got wrong. It's
written to be read start to finish by someone who hasn't seen the project. Each section
links to a detailed doc with the full tables.

---

## 1. The problem

**FinQA** is a dataset of questions about real company financial filings. Each question
comes with one page from a 10-K — some paragraphs of text and a table — and the answer has
to be *calculated* from numbers on that page.

A real example from our data:

> **Page:** `ZBH/2003/page_58.pdf`
> **Table row:** *"the finished goods of 2003 is $ 384.3 ; the finished goods of 2002 is $ 206.7"*
> **Question:** *"what is the change in finished goods in millions between 2002 and 2003?"*
> **Answer:** `177.6`

Two things make this hard, and both shaped every design decision we made:

**The answer is almost never written on the page.** We measured it: only **3.4%** of
correct answers appear as a literal number in their evidence. They're computed —
`384.3 − 206.7`. But the numbers *used* in the calculation are there **99.2%** of the time.

**A wrong answer can look completely plausible.** The model produces a confident number
with no signal that it's wrong. One of ours answered `-2017` to a percentage question — it
grabbed a year off the page and reported it as a financial figure.

That second problem is what this project is about.

---

## 2. What we built

Three stages, each answering one question:

```
  Question  ─►  RETRIEVAL   ─►  which chunks of this page are relevant?
                    │              Weaviate (86,421 chunks, filtered by doc_id)
                    │              + cross-encoder rerank
                    ▼
                GENERATION  ─►  what's the answer, given those chunks?
                    │              llama-3.3-70b, via Groq or AWS Bedrock
                    ▼
                DETECTION   ─►  does that answer follow from those chunks?
                                   LLM judge and rule-based verifier
```

Stages 1 and 2 are a standard RAG pipeline. Stage 3 is what the project is about.

**Detection is a separate question from accuracy.** "Is this answer right?" is answered by
comparing to the gold answer — we measure that too, and call it accuracy. "Does this answer
follow from the evidence?" is a different question, and it's the one that can still be
asked at deployment time, when no gold answer exists. Both verifiers are therefore never
shown the gold answer; the gold answer is used only to build the labels we score them
against.

### Infrastructure

| component | what runs it |
|---|---|
| corpus | 86,421 chunks from 2,789 filing pages, `data/processed/finqa_chunks.jsonl` |
| vector store | Weaviate Cloud, collection `FinqaChunk`, self-provided vectors, cosine |
| embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, local CPU |
| generator | `llama-3.3-70b` — Groq (free tier) or AWS Bedrock (`LLM_PROVIDER`) |
| judge | same interface; run with llama or `claude-haiku-4.5` on Bedrock |

The Groq free tier caps at 100K tokens/day, which was not enough for the prompt comparison
(465K tokens). Bedrock hosts the same llama-3.3-70b, so `LLM_PROVIDER=bedrock` removed the
cap without changing the model or invalidating earlier measurements.

---

## 3. Retrieval

**Job:** given a question and a filing page, return the 5 chunks most likely to contain
the answer.

### How it works

The 86,421 corpus chunks are embedded with `all-MiniLM-L6-v2` and stored in **Weaviate**
(`scripts/ingest_weaviate.py`). At query time every search is filtered to a single
`doc_id`, because FinQA guarantees the evidence is on one filing page:

```
86,421 chunks in Weaviate
      ↓  filtered by doc_id
    ~29 chunks on this question's page   (median; mean 31)
      ↓  first-stage retrieval, top 20
      ↓  cross-encoder re-scores those 20
     5 chunks passed to the generator
```

Four retrievers share one interface, `retrieve(question, doc_id, k) -> [chunk_id]`:

| retriever | how it scores | where it runs |
|---|---|---|
| `bm25` | keyword overlap | in-process (`rank_bm25`) |
| `dense` | cosine similarity on embeddings | Weaviate |
| `hybrid` | Weaviate's fusion of the two, `alpha=0.6` | Weaviate |
| `rerank` | cross-encoder over a 20-candidate pool | Weaviate (hybrid first stage) |

**The system as configured uses `rerank`, whose first stage is `hybrid`** — so Weaviate is
on the query path for the results below.

### Results

| retriever | Recall@1 | Recall@5 | ms/query |
|---|---|---|---|
| BM25 | 43.4% | 77.1% | 0 |
| Dense | 47.7% | 79.8% | 58 |
| Hybrid | 50.7% | 82.1% | 53 |
| **Rerank (hybrid first stage)** | **52.6%** | **84.3%** | 112 |
| Rerank (BM25 first stage) | 52.5% | 84.5% | 32 |

*Held-out test split, n=1147. Recall@k = fraction of gold evidence in the top k.*

**Why reranking helps.** BM25 and dense embeddings score the question and each chunk
*separately* — a chunk's representation is fixed before the question is seen. That's what
makes them cheap enough to index 86,000 chunks, and also why they struggle here, where the
deciding signal is usually an interaction ("which of these near-identical rows is the
*2009* one?"). A cross-encoder reads question and chunk together in one pass. Too slow for
a whole corpus, fine for 20 candidates.

**The two first stages perform the same after reranking.** Hybrid and BM25 land within
0.2pp of each other at every k once the cross-encoder runs, and the BM25 path is 3.5×
faster because it makes no network call. The first stage only has to get the gold evidence
into the top 20; on a ~29-chunk page both do that about equally often.

**Reranking shifts recall from text to tables.** Table-row recall rises 4 points at k=5,
text-sentence recall falls under 1. Around 60% of FinQA evidence is table-only, so total
recall rises — but text-evidence questions became slightly more likely to be answered
without their evidence, which shows up again in stage 3.

→ Full ablation, pool sweep, and failure analysis: **[rerank_results.md](rerank_results.md)**

---

## 4. Generation

**Job:** given the question and the 5 retrieved chunks, produce an answer.

### How it works

The retrieved chunk ids are resolved to text and rendered into an evidence block:

```
Evidence [1] (table_1): the finished goods of 2003 is $ 384.3 ; ...
Evidence [2] (text_7):  inventories are stated at the lower of cost or market ...

Question: what is the change in finished goods in millions between 2002 and 2003?
Answer:
```

That block plus a system prompt goes to llama-3.3-70b, and the reply is parsed back into a
single value (`src/generation/answer.py`). Scoring uses `answers_match`, which accepts
either of FinQA's two answer fields at either scale — they disagree on about half the rows
(`'53%'` is stored as `0.53232`).

Four prompt strategies share the block and differ in what surrounds it:

| strategy | what it adds |
|---|---|
| `zero_shot` | instructions only |
| `few_shot` | 3 worked examples drawn from the train split |
| `cot` | asks for step-by-step reasoning, then `FINAL ANSWER:` |
| `structured` | asks for JSON with `reasoning` / `answer` / `answer_type` |

All four instruct the model to answer `"I cannot determine the answer from the provided
evidence."` when the evidence is insufficient. That refusal path matters in stage 3.

### Results

All four ran over the same 150 dev questions, retrieval fixed at rerank k=5:

| strategy | accuracy | answered | avg latency |
|---|---|---|---|
| zero-shot | 60.0% | 86.7% | 1063 ms |
| few-shot | 60.7% | 87.3% | **346 ms** |
| chain-of-thought | 61.3% | 87.3% | 1713 ms |
| structured JSON | 61.3% | 89.3% | 1265 ms |

**Accuracy is statistically indistinguishable.** Every pairwise McNemar test returns
p ≥ 0.80. The strategies disagree on 7–16 individual questions each, but equally in both
directions — they fail on different questions rather than one failing less.

**Latency differs by 5×**, and it traces to output length. Few-shot replies `127.4`;
zero-shot replies *"To calculate the percentage cumulative total return, we need..."* —
about 5 output tokens against 150. Chain-of-thought is the slowest at 1713 ms and lands in
the same accuracy band as the rest.

→ Full tables and the paired test: **[prompt_comparison.md](prompt_comparison.md)**

---

## 5. Hallucination detection

This is the main contribution, and the part that produced the most surprises.

### 5.1 Building a labeled dataset

To measure a detector you need examples where you already know the answer. We derive
labels automatically instead of hand-annotating, using three fields the pipeline already
records:

| gold evidence retrieved? | what the model did | label |
|---|---|---|
| yes | correct answer | **supported** |
| yes | wrong answer | **hallucinated** — it had what it needed |
| no | refused to answer | **supported** — refusing is the right call |
| no | wrong answer | **hallucinated** — it invented something |
| no | correct answer | *skipped* — lucky guess or reachable another way; can't tell |

Running 150 dev questions × 4 strategies gave **560 labeled examples, 161 of them
hallucinations (29%)**.

The positive class has two genuinely different kinds, and keeping them apart mattered:

- **120 arithmetic errors** — had the right evidence, computed wrongly
- **41 fabrications** — didn't have the evidence, answered anyway

We nearly missed the second kind entirely. Our first version dropped *every* row where
retrieval failed, on the grounds that the model might have read a wrong chunk faithfully.
That reasoning applies to only one of the four cases above, and dropping all of them
excluded the clearest hallucinations in the dataset — the `-2017` example is one of them.

### 5.2 The two verifiers

Both share the same `verify(question, context, answer)` signature, so one script scores
either:

```python
verify(question, context, answer) -> {"supported": bool, "category": str, ...}
```

Neither is shown the gold answer — at deployment there isn't one, so a verifier that
needed it would be measuring nothing.

**LLM-as-a-judge** (`src/detection/llm_judge.py`). Sends the question, the evidence block,
and the candidate answer to a model and asks whether the answer follows from the evidence.
The reply is JSON: `supported`, `category`, `computed_value`, `confidence`, `reasoning`,
`cited_evidence`. Answers that decline are short-circuited to supported without an API
call.

**Rule-based verifier** (`src/detection/rule_based.py`). Extracts every number from the
evidence and searches for a calculation that produces the candidate answer, within 1%. It
tries the operations FinQA's own reasoning programs use — divide (64% of dev programs),
subtract (19%), add (6%), multiply (5.5%) — plus percent change, over every ordered pair,
and compares at ×1 / ×100 / ÷100 because FinQA writes percentages both ways.

It re-derives rather than looks up because only **3.4%** of correct answers appear
literally in their evidence, while **99.2%** of the numbers used in the calculation do.

### 5.3 Two definitions, measured separately

"Hallucination" can mean two different things here, and we report both because they are
different tasks with different answers:

| | **A — ungrounded answer** | **B — answered without evidence** |
|---|---|---|
| counts as a hallucination | any answer that doesn't follow from the evidence, including arithmetic errors | only answers given when the evidence lacked the necessary figures |
| arithmetic error with correct evidence | hallucination | *not* a hallucination — a generation error |
| positives | 161 of 560 (29%) | 41 of 560 (7%) |

Definition A is what Member 2's taxonomy encodes (`numeric_error` = "the arithmetic is
wrong relative to the evidence"). Definition B is the narrower reading: the model asserted
something the page cannot support.

Their F1 scores are **not comparable to each other** — the positive rates differ by 4×, and
F1 depends on prevalence. Each is compared against its own flag-everything baseline.

### 5.4 Results under definition A (ungrounded answer)

| verifier | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| Rule-based re-derivation | 0.38 | 0.25 | 0.30 | 0.66 |
| LLM judge — llama-3.3-70b | 0.49 | 0.35 | 0.41 | 0.71 |
| **LLM judge — claude-haiku-4.5** | **0.61** | **0.49** | **0.54** | **0.76** |
| *baseline: flag everything* | 0.29 | 1.00 | 0.45 | 0.29 |

*n = 560. The best detector clears the baseline by 1.2×.*

**Changing the judging model moved the result more than any other change.** Same prompt,
same answers, same rows — only the model differs. llama-3.3-70b scores below the baseline
(0.41 vs 0.45); claude-haiku-4.5 scores above it (0.54). Paired McNemar over the same rows:
**p = 0.001**, and significant separately for both kinds of hallucination (arithmetic
p = 0.043, fabrication p = 0.008).

Judging with a different model than the one that wrote the answers also removes
self-evaluation bias.

**Recall by kind of hallucination:**

| | arithmetic errors (n=120) | fabrications (n=41) | false alarms (n=325) |
|---|---|---|---|
| rule-based | 28% | 20% | 21% |
| llama judge | 32% | 44% | 18% |
| **claude judge** | **41%** | **73%** | 16% |

The largest single gap is fabrications, 44% → 73% — the case where there is nothing to
compute and the judge only has to notice the evidence doesn't contain the answer.

### 5.5 The judge anchors on the answer it is grading

Decomposing Claude's 82 misses using the `computed_value` it reports:

| cause | n | |
|---|---|---|
| never computed a value | 2 | 2% |
| **computed a value, but it was wrong** | 58 | **71%** |
| computed the right value, approved anyway | 18 | 22% |

The dominant failure is the judge's own arithmetic being wrong *in a way that agrees with
the model's wrong answer*. That is not a coincidence. On rows where the model was wrong —
so the true answer and the candidate differ — the judge's own computation lands on:

- the **candidate answer**: 57%
- the **true answer**: 30%

And the control is decisive:

| | judge's arithmetic accuracy |
|---|---|
| model was right (answer = truth) | **90%** |
| model was wrong (answer ≠ truth) | **30%** |

The arithmetic is equally hard in both cases. A 60-point gap means the candidate answer is
steering the computation: the judge reads `1095` in its prompt, computes `1095`, and
concludes the answer checks out.

This also explains an apparent contradiction. Claude's *improvement over llama* came from
evidence assessment, not arithmetic (see below). But Claude's *remaining failures* are 71%
arithmetic — because that arithmetic is contaminated, not because it is inherently weak.

### 5.6 Results under definition B (answered without evidence)

If the judge's arithmetic is unreliable on exactly the rows that matter, checking
arithmetic with it is unsound. Definition B avoids the problem: deciding whether the
evidence contains the necessary figures requires no computation at all.

We added a second judge mode (`--mode evidence`) that asks only *"does this evidence
contain what is needed to answer this question?"* — **and never shows the judge the
candidate answer**, removing anchoring by construction rather than by instruction.

| judge prompt | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| grounding — sees the answer, computes | 0.23 | **0.73** | 0.35 | 0.80 |
| **evidence — no answer, no arithmetic** | **0.32** | 0.56 | **0.41** | **0.88** |
| *baseline: flag everything* | 0.07 | 1.00 | 0.14 | 0.07 |

*n = 560, 41 positives. The evidence-mode judge clears its baseline by **2.9×** — the
largest margin of any configuration in this project, against 1.2× under definition A.*

**Where the flags moved:**

| row type | grounding | evidence |
|---|---|---|
| fabricated — the target | 73% | 56% |
| arithmetic wrong | 41% | **15%** |
| correct answer | 16% | **10%** |
| refused, no evidence | 0% | 0% |

The evidence-mode judge largely stopped flagging arithmetic errors and correct answers.
That selectivity is where the precision and accuracy gains come from — it is answering the
question it was asked rather than a general "does this look wrong".

**The tradeoff is real:** recall fell 73% → 56%. Seeing the candidate answer did help the
judge notice when an answer referenced figures that weren't on the page; removing it took
that signal away along with the anchoring.

Neither mode dominates. With 41 positives the F1 gap (0.35 → 0.41) is directional rather
than conclusive.

### 5.7 Behaviour of the rule-based verifier

The first version marked **91% of all answers supported**. With every number on the page as
an operand and 5 operations over every ordered pair, the search reaches most targets — the
space of computable values is large relative to how precisely we require a match. Its recall
was 0.11.

Widening the search would make this worse, not better, so we constrained it instead. Two
changes, measured on the 560-row set:

| version | precision | recall | F1 | accuracy | % supported |
|---|---|---|---|---|---|
| all numbers, all operations | 0.38 | 0.11 | 0.17 | 0.69 | 91% |
| + drop year-like numbers | 0.47 | 0.17 | 0.25 | 0.71 | 90% |
| **+ operation chosen from question wording** | 0.38 | **0.25** | **0.30** | 0.66 | 81% |

**Dropping years** removes whole numbers in 1900–2030 from the operand pool. Filing pages
are full of them, they are never the answer, and they gave the search extra material to hit
targets with by chance.

**Choosing the operation from the question** reads the wording — "change in" implies a
subtraction, "what percentage" a division — and searches only those. A question that gives
no signal falls back to trying everything.

Other variants we measured but did not adopt: rejecting answers reachable by more than *k*
distinct calculations (F1 up to 0.49 at k=1, but accuracy falls to 0.48 and it flags 73% of
all answers — approaching the flag-everything baseline rather than beating it). Tightening
the match tolerance behaves the same way — F1 rises to 0.33 at 0.01% while accuracy falls
to 0.49.

**Restricting operands to a single chunk is the interesting rejection.** It scores F1 0.42,
better than any configuration in the table above and still below the flag-everything
baseline of 0.45 — but the gain is not detection. A chunk is one table *row*, so a question
comparing two line items needs two chunks by construction, and 45% of our questions do. On
correct answers whose gold derivation spans two or more chunks, the constraint wrongly
flags **86%** of them, against 32% unrestricted. What it actually measures is "did this
question need more than one table row", which correlates with hallucination only because
harder questions are answered wrong more often. We report it as a worked example of a
metric improving while the underlying behaviour gets worse.

**Why no operand rule can fix this.** Unconstrained, 80% of answers fabricated with *no*
supporting evidence are still derivable from the numbers on the page — against 79% of
correct answers. Derivability carries essentially no information about grounding; a filing
page simply has enough figures that some pair reaches almost any target within 1%. Given an
oracle that restricts operands to the gold evidence chunks, fabrications fall to 27%
derivable and the verifier reaches F1 0.53 — still below claude-haiku-4.5 at 0.54, which
gets no such help. That oracle is the ceiling for any operand-selection scheme, and it
requires knowing which chunk answers the question, which is most of the detection problem.

**`entity_error` is still not detectable in general.** That category means the right kind
of number for the wrong thing — the 2003 figure when the question asked for the change
between 2003 and 2002. Any figure printed on the page satisfies the "quoted directly"
check. Dropping years catches the subset where a model reports a year as an answer (one of
ours answered `-2017` to a percentage question), but the general case needs to know what
each number represents.

**Combined with the LLM judge** — flagging when either fires — recall rises 0.49 → 0.63 and
precision falls 0.61 → 0.50, for F1 0.56 against Claude's 0.54. That +0.02 is bought by
flagging 37% of answers instead of 23%, and accuracy falls 0.76 → 0.71: the union adds 23
true positives and 53 false ones. It is not evidence that the rule-based verifier
contributes anything the judge lacks.

→ Full analysis, prompt iterations, and quoted judge reasoning:
**[detection_dataset.md](detection_dataset.md)**

---

## 6. What we got wrong

Three mistakes we caught, all instructive.

**A 40-example evaluation set gave a confidently wrong answer.** Our first detection set
had 40 rows and reported F1 **0.67**, rising to 0.88 once "near-misses" were excluded. It
looked like the judge worked well and just disagreed about rounding tolerance. Scaling to
560 rows reversed it completely — F1 **0.28** on the same prompt, and near-misses turned
out to be 10% of hallucinations rather than 50%. The pilot had 14 positive examples; that
was never enough to estimate recall. **Numbers from that pilot are retracted.**

**The judge's prompt contradicted its own taxonomy.** `categories.py` defines
`numeric_error` as "the arithmetic is wrong relative to the evidence" — a category only
usable when `supported=false`. But the prompt said `supported=false` **ONLY** when the
answer uses a value not in the evidence, cites the wrong entity, or invents information.
No slot for bad arithmetic. The judge obeyed the prompt and waved through answers it had
itself computed differently:

> *"339235 − 338240 = **995**, but the candidate answer is **1095**, however, this is
> likely a rounding error and the evidence still supports the calculation"* → `supported=true`

Fixing the prompt (bounded tolerance, arithmetic added to the false conditions, forced the
judge to emit its computed value as a field) raised recall 0.21 → 0.32.

**We blamed the wrong bottleneck, twice.** First we assumed the ceiling was arithmetic
*ability* — a judge can't catch a wrong computation if it can't compute. Swapping to Claude
disproved that: arithmetic accuracy barely moved (69% → 71%) and follow-through on its own
number barely moved (84% → 88%), yet F1 rose 32%. The gain came from evidence assessment.

Then we found the arithmetic wasn't weak at all — it was **contaminated**. The judge
computes correctly 90% of the time when the model's answer happens to be right, and 30% of
the time when it's wrong. It anchors on the answer it is checking (§5.5). Both of our
earlier explanations were wrong about the mechanism, and only measuring `computed_value`
against the truth separately for right and wrong answers exposed it.

**We started with the wrong question.** Our whole detection setup asked "does this answer
follow from the evidence?", which requires the judge to redo the arithmetic — the one thing
it can't do reliably here. Asking instead "was there enough evidence to answer at all?"
needs no arithmetic, can't be anchored, and clears its baseline by 2.9× against 1.2× for
the original framing (§5.6). Both are reported; neither is strictly better, but the second
is a sounder question to put to an LLM.

---

## 7. Limitations

- **Rows aren't independent.** 560 rows come from 141 questions — each appears up to 4
  times, once per strategy, with the same evidence. Effective n is closer to 141.
- **Dev split only** for generation and detection. Test is held out; only retrieval has
  test-split numbers.
- **One generator model.** Everything was answered by llama-3.3-70b. Prompt-strategy
  conclusions in particular may not transfer — CoT tends to help more on weaker models.
- **A lucky guess counts as supported.** If the model produces the right number without
  really using the evidence, our rule labels it supported. Rare, but it makes the
  supported class slightly optimistic.
- **Categories are reported, not graded.** We have no per-category ground truth, so we can
  say what a judge *claimed* but not whether the claim was right.
- **We never tested a large judge.** Opus 5 and Sonnet 5 are in the Bedrock catalogue but
  return `AccessDeniedException` on this account. Given that a *small* newer model produced
  the biggest single improvement, this is the most promising thing left to try.

---

## 8. Untested directions

Things the results point at that we did not get to measure:

1. **A larger judge model.** Changing the judge produced the largest effect we observed,
   and claude-haiku-4.5 is the smallest current-generation model available on the account.
   Opus 5 and Sonnet 5 are listed in the Bedrock catalogue but return
   `AccessDeniedException`.
2. **Retrieval as a source of hallucinations.** 41 of 161 hallucinations occur on rows
   where retrieval did not return the gold evidence.
3. **Refusal rate.** 74 of 560 answers were refusals; both judges score them correctly by
   construction, so the refusal rate directly affects the measured hallucination rate.
4. **Per-category ground truth.** We report which category a judge assigned but have no
   labels to check those assignments against.

---

## 9. Who did what

| | Stage 1–2 | Stage 3 | Stage 4 |
|---|---|---|---|
| **Member 1** — Swarali Degaonkar | dataset exploration, BM25 | baseline pipeline | rule-based verifier* |
| **Member 2** — Jaya Sriharshita Koneti | preprocessing, dense + hybrid retrieval | 4 prompt strategies | LLM-as-a-judge |
| **Member 3** — Kushal Pendekanti | repo setup, data loading, evaluation harness | cross-encoder reranking | labeled dataset, detection evaluation |

\* implemented late in the project as part of the detection evaluation, so the two
verifiers could be compared head-to-head.

---

## 10. Reproducing

Everything below is free except the two generation/judging steps.

```bash
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY, or set LLM_PROVIDER=bedrock

# Data (free)
python scripts/build_corpus.py --splits train dev test
python scripts/ingest_weaviate.py --recreate --split train dev test

# Retrieval (free — no LLM calls)
python scripts/eval_rerank.py --split test --pools 20 --bases hybrid bm25

# Generation (~600 LLM calls)
python scripts/compare_prompts.py --split dev --limit 150 --retriever rerank --k 5 \
    --out data/runs/prompts_dev_rerank.jsonl

# Detection
python scripts/build_detection_set.py --runs data/runs/prompts_dev_rerank.jsonl \
    --out data/processed/detection_eval.jsonl                      # free
python scripts/run_rule_verifier.py --input data/processed/detection_eval.jsonl  # free
python scripts/run_llm_judge.py --input data/processed/detection_eval.jsonl      # ~486 calls
python scripts/score_detection.py --input data/runs/judge_detection_eval.jsonl   # free
```

All result files referenced in this report are committed under `data/runs/`.

Total LLM cost for every experiment here: about **$2** on AWS Bedrock (~2,500 calls).
