# Prompt strategy comparison

Member 2 wrote the four prompt strategies in `src/generation/prompts.py`; this is the
first time they've been run head-to-head. Interpretation for the report belongs with
Member 2 — this doc just records the numbers and how they were produced.

Run with `scripts/compare_prompts.py` on 150 dev questions, cross-encoder reranking at
k=5, llama-3.3-70b via AWS Bedrock. Raw output: `data/runs/prompts_dev_rerank.jsonl`
(600 rows = 150 questions × 4 strategies).

## Summary

**No strategy is measurably more accurate than any other.** All four land within 1.3
percentage points, and a paired test says even that is noise.

**But few-shot is 3–5× faster than the rest, for free.** That's the actual finding, and
it's the one that should drive which strategy the final system uses.

## Results

```
strategy      accuracy  attempted  %answered  avg_lat_ms     n
zero_shot       60.0%     69.2%     86.7%       1063   150
few_shot        60.7%     69.5%     87.3%        346   150
cot             61.3%     70.2%     87.3%       1713   150
structured      61.3%     68.7%     89.3%       1265   150
```

`accuracy` counts a refusal as wrong; `attempted` is accuracy over just the questions the
model actually answered.

## The accuracy differences are noise

Every strategy saw the same 150 questions, so we can compare them question by question
instead of just comparing averages. For each pair, count the questions where one strategy
was right and the other wrong (McNemar's test — a two-sided exact binomial on those
disagreements):

| A vs B | A only | B only | net | p |
|---|---|---|---|---|
| zero_shot vs few_shot | 12 | 13 | −1 | 1.00 |
| zero_shot vs cot | 7 | 9 | −2 | 0.80 |
| zero_shot vs structured | 9 | 11 | −2 | 0.82 |
| few_shot vs cot | 15 | 16 | −1 | 1.00 |
| few_shot vs structured | 12 | 13 | −1 | 1.00 |
| cot vs structured | 9 | 9 | 0 | 1.00 |

Nothing is close to significant. The strategies disagree on plenty of individual questions
(7–16 either way), but they disagree in *both directions equally* — so no strategy is
actually better, they just fail on different questions.

This is worth stating plainly in the report rather than picking whichever number is
highest: on FinQA with good retrieval, **how you ask doesn't change how often the model is
right.** Retrieval and the model do the work.

## Latency is where they differ

Same questions, so these are directly comparable:

| strategy | avg latency | vs few_shot |
|---|---|---|
| few_shot | 346 ms | 1.0× |
| zero_shot | 1063 ms | 3.1× |
| structured | 1265 ms | 3.7× |
| cot | 1713 ms | **4.9×** |

The cause is visible in the raw outputs. Few-shot answers with just the number:

```
few_shot    raw='127.4'      raw='93.5%'      raw='24.7%'
zero_shot   raw='To calculate the percentage cumulative total return, we need the start...'
```

The examples teach it to skip the preamble, so it emits roughly 5 output tokens instead of
~150. Latency here is almost entirely output tokens.

**Chain-of-thought costs 4.9× the latency for zero accuracy gain.** That's the clearest
practical result in this table — CoT is usually assumed to help, and on this task it
doesn't.

## Recommendation

Use **few_shot** for the final system: tied for best accuracy, 3–5× faster, and cheapest
in output tokens. If a later experiment needs the model's reasoning visible (for the
hallucination detector, say), `cot` or `structured` earn their cost — but not for accuracy.

## Cost, and why this run was possible

Groq's free tier caps at 100K tokens/day. This run used **392K input + 58.5K output tokens
across 601 calls** — about 4.5 days of Groq budget, which is why the comparison had never
been run.

Running it through AWS Bedrock on the *same* model (`us.meta.llama3-3-70b-instruct-v1:0`)
removed the cap without changing the model, so these numbers stay comparable with
everything measured earlier. Cost was roughly **$0.30**.

Switch backends with `LLM_PROVIDER=bedrock` in `.env` (see `src/generation/bedrock.py`).
It defaults to `groq`, so nobody's setup changes unless they opt in.

## Limitations

- **150 questions.** Enough to rule out a large accuracy difference, not enough to rule
  out a small one — the paired test can only say "no difference bigger than a few points."
- **One retriever.** All four strategies ran on cross-encoder rerank at k=5. A weaker
  retriever might separate them, since strategies could differ in how well they cope with
  missing evidence.
- **One model.** llama-3.3-70b only. CoT in particular tends to help more on models that
  reason less well by default, so this result may not transfer.
- **Dev split.** Held out `test` for final numbers.
