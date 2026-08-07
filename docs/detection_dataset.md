# Hallucination evaluation set (Stage 4, Member 3)

The labeled dataset we use to test the hallucination verifiers, plus results for
Member 2's LLM judge.

- Built by [`scripts/build_detection_set.py`](../scripts/build_detection_set.py)
- Dataset: `data/processed/detection_eval.jsonl` (560 rows)
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

Built from `data/runs/prompts_dev_rerank.jsonl` — 150 dev questions × 4 prompt strategies,
cross-encoder rerank at k=5, llama-3.3-70b via Bedrock.

```
Read 600 answers, kept 560
  correct                       325     -> supported
  wrong_despite_evidence        120     -> hallucinated  (had the evidence)
  fabricated_without_evidence    41     -> hallucinated  (no evidence to work from)
  abstention                     31     -> supported     (refused, had evidence)
  abstention_no_evidence         43     -> supported     (refused, no evidence)
  correct_without_evidence       40     -> skipped        (can't tell — lucky guess?)

supported 399   hallucinated 161     (29% positive class)
```

**The hallucinated class has two distinct kinds**, and keeping them apart is what makes
the results below interpretable:

| kind | n | what it means |
|---|---|---|
| arithmetic wrong | 120 | had the right evidence, computed wrongly |
| fabricated | 41 | didn't have the evidence, answered anyway |

560 rows from **119 unique questions** — each appears up to 4 times, once per strategy,
with the *same* evidence. The rows are not independent; effective n is closer to 119.

Only 40 rows are skipped: the model got the right answer *without* the gold evidence, so
we can't tell whether it guessed luckily or reached the figure another way.

> **An earlier version dropped all 124 rows where retrieval failed**, on the grounds that
> the model might have read a wrong chunk faithfully. That reasoning applies to only one of
> the cases above, and dropping them all excluded the clearest hallucinations in the
> dataset — including one that answered `-2017` to a percentage question, having grabbed a
> year off the page.

## Headline result

**Which model judges matters more than anything else we tried.** Same 560 rows, same
prompt, same answers — only the judging model changes:

| verifier | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| rule-based re-derivation | 0.38 | 0.11 | 0.17 | 0.69 |
| llama-3.3-70b (also wrote the answers) | 0.49 | 0.35 | 0.41 | 0.71 |
| **claude-haiku-4.5** | **0.61** | **0.49** | **0.54** | **0.76** |
| "flag every answer as hallucinated" | 0.29 | 1.00 | 0.45 | 0.29 |

This flips the conclusion. With llama judging llama the detector is **worse than a
one-line baseline** that flags everything (0.41 vs 0.45) — i.e. not worth deploying. Swap
in Claude Haiku and it clears the baseline comfortably (0.54). Same evidence, same
candidate answers, same prompt.

The gains are significant on paired tests over the same rows (McNemar):

| subset | claude caught, llama missed | llama caught, claude missed | p |
|---|---|---|---|
| all hallucinations | 33 | 10 | **0.001** |
| arithmetic errors | 18 | 7 | **0.043** |
| fabrications | 15 | 3 | **0.008** |

Using a different model to judge also removes the self-evaluation bias — llama wrote the
answers, Claude grades them.

### Recall by kind of hallucination

| kind | rule-based | llama | claude | n |
|---|---|---|---|---|
| arithmetic wrong, had the evidence | 13% | 32% | **41%** | 120 |
| fabricated, evidence missing | 5% | 44% | **73%** | 41 |
| false alarms on correct answers | 9% | 18% | 16% | 325 |

**The big gain is fabrications (44% → 73%).** That is the case where there is nothing to
compute — the answer simply isn't in the evidence, and the judge only has to notice.

### Why it improved — not the reason we expected

We assumed the bottleneck was arithmetic: the judge can't detect a wrong computation if it
can't do the computation. That turned out to be wrong.

| | llama | claude |
|---|---|---|
| judge's own `computed_value` matches the true answer | 69% | 71% |
| verdict agrees with its own value-vs-answer comparison | 84% | 88% |

Both barely move, yet F1 rises 32% relative. So the improvement is **not** better
arithmetic and **not** better follow-through on its own numbers — it is better *evidence
assessment*. Claude notices when the retrieved chunks don't contain what the question
needs; llama proceeds as if they do. That is exactly what the fabrication column shows.

Practical reading for the report: for RAG hallucination detection, judge capability is
mostly about **reading the evidence carefully**, not about doing the domain reasoning.
That is a cheaper property to buy than we assumed — Haiku is a small model.

(74 of the 560 rows are refusals the judge short-circuits with no API call. They're
correct by construction and inflate accuracy slightly.)

## The rule-based verifier

`src/detection/rule_based.py` — no LLM. It tries to **re-derive** the answer from the
numbers in the evidence: pull out every number, try the operations FinQA actually uses
(divide, subtract, add, multiply, percent change), and see if any combination produces the
candidate answer within tolerance.

It had to re-derive rather than look up, because only **3.4%** of correct answers appear
literally in their evidence, while **99.2%** of the numbers used in the calculation do.

**It scores F1 0.17 — the worst of the three, and worse than the trivial baseline.**

### Why: almost everything is derivable

At the 1% tolerance we hold the LLM judge to, **91% of all answers pass**. With 20–30
numbers on a page and 5 operations, the search reaches nearly any target by accident.
Tightening the tolerance confirms the cause:

| tolerance | precision | recall | F1 | % approved |
|---|---|---|---|---|
| 5% | 0.42 | 0.03 | 0.06 | 98% |
| 1% (default) | 0.38 | 0.11 | 0.17 | 91% |
| 0.1% | 0.29 | 0.32 | 0.30 | 68% |
| **0.01%** | 0.26 | **0.43** | **0.33** | 53% |

Even at its best it loses to both LLM judges and to "flag everything" (0.45).

### It structurally cannot catch entity errors

`entity_error` means the right kind of number for the wrong thing — reporting 2002's figure
when the question asked about 2003. This verifier can never flag it: any number printed on
the page passes the "quoted directly" check, so a wrong-but-present number always looks
supported. Catching it needs to know what each number *means*. That branch was written,
found to be unreachable, and removed — the limitation is pinned by
`test_known_limitation_a_number_lifted_off_the_page_looks_supported`.

### Combining it with the LLM judge doesn't help

Flagging when *either* verifier fires: recall rises 0.49 → **0.70**, but precision drops to
0.36, giving F1 **0.47** — below Claude alone at 0.54. Useful only if you care about recall
much more than precision, e.g. queueing flagged answers for human review.

## Prompt iterations

Three versions of the judge prompt, scored on the 476-row set (before fabrications were
added):

| | v1 original prompt | v2 fixed prompt | v2 + python comparison |
|---|---|---|---|
| precision | 0.42 | 0.39 | **0.58** |
| recall | 0.21 | 0.32 | **0.35** |
| F1 | 0.28 | 0.35 | **0.44** |
| accuracy | 0.73 | 0.70 | **0.77** |
| hallucinations caught | 25/120 | 38/120 | 42/120 |
| good answers wrongly flagged | 34 | 59 | 30 |

Raw outputs kept for both prompts: `data/runs/judge_v1_original_prompt.jsonl` and
`judge_v2_fixed_prompt.jsonl`. Ignoring near-misses changes nothing (F1 0.28 → 0.29 on
v1), so tolerance is not the issue — see below.

## The judge has a recall problem, and we know why

**It approves answers that are badly wrong.** Of the 95 hallucinations it let through:

| how far off | count |
|---|---|
| within 10% (rounding) | 11 (12%) |
| 10–50% off | 21 (22%) |
| 50–200% off | 34 (36%) |
| more than 200% off, or unparseable | 29 (31%) |

So this is not a tolerance problem. Two thirds are off by more than half.

**It often derives the right answer and approves anyway.** The reasoning field gives it
away:

```
answered 1095, correct answer 995
  "The increase ... can be calculated by subtracting the 2016 value from the 2017
   value, which is 339235 - 338240 = 995, but the candidate answer is 1095, however,
   this is likely a rounding error and the evidence still supports the calculation"
                                                              -> supported = true
```

```
answered 1, correct answer 995
  "... = 995, but since the question asks for the answer in thousands and the candidate
   answer is 1, it seems there might be a misunderstanding in the units or a rounding
   issue, however, the evidence does support calculating an increase"
                                                              -> supported = true
```

A 10% error gets called "rounding". An answer off by a factor of 995 gets called a "unit
misunderstanding". **12 of the 95 false negatives (13%) contain the correct answer inside
the judge's own reasoning and are still marked supported.**

The cause looks like the judge's own prompt. `llm_judge._JUDGE_SYSTEM` says an answer is
supported if the values needed "appear in the evidence AND the answer is consistent with
performing it", and separately tells it to "allow for rounding". In practice it keeps the
first half and drops the second — it grades whether the evidence is *sufficient*, not
whether the answer is *right*, and uses "allow for rounding" as a blanket excuse whenever
the two disagree.

### The prompt contradicted the taxonomy

`categories.py` defines `numeric_error` as "the answer is a number but its value, units,
or underlying **arithmetic is wrong** relative to the evidence" — a category that can only
be used when `supported=false`. But the original `_JUDGE_SYSTEM` said:

> `supported=false` **ONLY** when the answer relies on a value NOT in the evidence, cites
> the wrong entity/unit, or invents information.

That `ONLY` list has no slot for bad arithmetic. The prompt told the judge a wrong
computation is *not* grounds for `supported=false`, while the taxonomy defined a whole
category for exactly that. The judge obeyed the prompt.

**We fixed the prompt** (v2): bounded the rounding allowance to 1%, added bad arithmetic
to the `supported=false` conditions, named the three excuses we saw it using so it stops
reaching for them, and made it emit its own `computed_value` as a JSON field instead of
comparing inside prose. Recall went **0.21 → 0.32**, F1 **0.28 → 0.35**. Real, but partial.

### Why the fix only went so far

The `computed_value` field (populated on 87% of rows) lets us pull the judge apart:

| | |
|---|---|
| Judge's own computed value matches the true answer | **73%** |
| Its verdict agrees with its own number-vs-candidate comparison | **83%** |

So two separate failures. It gets the arithmetic wrong 27% of the time — that is a hard
ceiling no prompt can lift. And even when it computes correctly, 17% of the time it still
waves through an answer that disagrees with its own number.

**The second failure is fixable in code.** Take the judge's `computed_value`, do the 1%
comparison in Python instead of trusting the model to act on it, and F1 goes **0.35 →
0.44** with precision **0.39 → 0.58**. Same LLM output; we just stop asking it to make
the final call.

That points at a **hybrid verifier**: the LLM does what LLMs are good at (read the
evidence, work out what computation the question wants, produce a number), and code does
what code is good at (compare two numbers).

> **We later found this framing was only half right.** We predicted the 27% arithmetic
> error rate was the ceiling and that a stronger judge would lift it by computing better.
> Swapping to Claude Haiku raised F1 0.41 → 0.54, but its arithmetic barely improved
> (69% → 71%) and its follow-through on its own number barely improved (84% → 88%). The
> gain came from **evidence assessment** — noticing when the retrieved chunks don't contain
> what the question needs. See the fabrication column in the headline section.
>
> Separately, pairing the LLM judge with the standalone rule-based verifier does **not**
> help: recall rises to 0.70 but precision falls to 0.36 (F1 0.47, below Claude's 0.54).
> Using the judge's *own* `computed_value` in a Python comparison does help; bolting on a
> second, independent arithmetic searcher does not.

### Also worth noting

`confidence` is **1.00 on every single verdict** — right ones and wrong ones alike. It
carries no information and can't be used as a filter. Drop it or recalibrate it.

## A note on our earlier pilot (n=40)

An earlier version of this set had 40 rows and reported P=0.80 / R=0.57 / F1=0.67, rising
to F1=0.88 once "near-misses" were excluded. **Those numbers did not survive scaling up**
and should not be quoted.

At n=40 half the hallucinations happened to be near-misses (7 of 14), which made the
judge's misses look like a definitional disagreement about rounding. At n=476 only 10%
are (12 of 120), and excluding them changes F1 by 0.01. The pilot had 14 positive
examples; that was too few to estimate recall at all.

This is worth a sentence in the report on its own — it's a clean example of a small
evaluation set producing a confidently wrong conclusion.

## Running it

```bash
# free
python scripts/build_detection_set.py --runs data/runs/prompts_dev_rerank.jsonl \
    --out data/processed/detection_eval.jsonl

# free — the rule-based verifier makes no API calls
python scripts/run_rule_verifier.py --input data/processed/detection_eval.jsonl

# 486 calls (74 of the 560 are refusals and skip the API) — about $0.40 on Bedrock
LLM_PROVIDER=bedrock python scripts/run_llm_judge.py \
    --input data/processed/detection_eval.jsonl --out data/runs/judge_detection_eval.jsonl

# ...or with Claude as the judge, which is the configuration that works
LLM_PROVIDER=bedrock BEDROCK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0 \
    python scripts/run_llm_judge.py --input data/processed/detection_eval.jsonl \
    --out data/runs/judge_v4_claude_haiku.jsonl

# free
python scripts/score_detection.py --input data/runs/judge_v4_claude_haiku.jsonl
```

## Problems with this

- **Rows aren't independent.** 560 rows, 119 questions. Report per-strategy numbers, or
  treat n as ~119.
- **A lucky guess looks supported.** If the model gets the right number without really
  using the evidence, we label it supported. Probably rare.
- **The judge grades the same model that wrote the answers** (llama-3.3-70b both sides).
  Now that we're on Bedrock, running the judge on a *different* model would remove this
  and is cheap — the single best next experiment.
- **Categories are reported, not graded.** No per-category ground truth, so we can only
  say what the judge claimed: `numeric_error` 51, `unsupported_claim` 8, `abstention` 31,
  `supported` 386.
- **The rule-based verifier can't be fixed by tuning.** Its ceiling is set by how many
  numbers a filing page has, not by any parameter — see the tolerance sweep above. Making
  it useful would mean parsing what each number *represents* (which year, which line item),
  which is a different project.
- **Nobody re-ran the LLM judge after the fabrication rows were added at the v2 prompt.**
  v3 and v4 both use the fixed prompt on the full 560 rows, so they're comparable to each
  other; the v1/v2 prompt-iteration table is on the older 476-row set and shouldn't be
  compared against them directly.
