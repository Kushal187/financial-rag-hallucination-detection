# Financial RAG with Hallucination Detection

A retrieval-augmented question answering system for company 10-K filings (FinQA), with
hallucination detection built into it. We built all three stages: retrieval, generation,
and a detector that checks whether each answer is supported by the evidence the model read.

CS6120 Natural Language Processing course project.

## Quick start

The corpus and the saved experiment runs are in the repo, so the demo needs no API keys
and no configuration. On Python 3.10:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

Everything below is only needed to re-run the experiments yourself.

## What it does

Three stages:

1. **Retrieval.** The corpus is 86,421 chunks (text sentences and linearized table rows)
   from 2,789 filing pages, stored in Weaviate. A question is only searched against its
   own page. BM25 and vector search are fused, the top 20 go to a cross-encoder, and the
   best 5 are kept.
2. **Generation.** llama-3.3-70b reads those 5 chunks and answers the question.
3. **Detection.** Two verifiers decide whether that answer follows from those 5 chunks.
   Neither is shown the correct answer, since in real use there isn't one.

Detection is part of the pipeline rather than a separate step. It is there because FinQA
answers are calculated rather than quoted, so almost no correct answer appears word for
word in its evidence, and a wrong answer looks exactly as confident as a right one.

The two verifiers work differently:

- **Rule based.** Tries to re-derive the answer from the numbers on the page using the
  operations FinQA questions actually ask for. If nothing reaches the answer, flag it.
  No LLM, so it costs nothing.
- **LLM judge.** Asks a second model whether the answer follows from the evidence. It has
  two prompts: one that shows it the answer, and one that hides the answer and only asks
  whether the page contains what the question needs.

## Setup for live runs

Only needed to call the models yourself or rebuild the corpus.

```bash
cp .env.example .env
```

Anything not set falls back to a default in the code. These are the ones with no default:

| variable | what it's for |
|---|---|
| `LLM_PROVIDER` | `groq` (free, 100K tokens/day) or `bedrock` (AWS, no daily cap) |
| `GROQ_API_KEY` | if using Groq |
| `AWS_REGION`, `BEDROCK_MODEL` | if using Bedrock |
| `AWS_BEARER_TOKEN_BEDROCK` | a Bedrock API key, if you would rather not configure AWS credentials |
| `JUDGE_MODEL` | the model that grades answers, when it should differ from the one writing them |
| `WEAVIATE_URL`, `WEAVIATE_API_KEY` | the vector store |

Both providers run the same llama-3.3-70b, so results are comparable either way.

## Running the pipeline

Build the corpus and load it into Weaviate. A few minutes, no API calls.

```bash
python scripts/build_corpus.py --splits train dev test
python scripts/ingest_weaviate.py --recreate --split train dev test
```

Compare the retrievers. Free, no LLM calls.

```bash
python scripts/eval_retrieval.py --split test --retriever rerank --k 1 3 5 10
python scripts/eval_rerank.py --split dev --pools 10 20 30 --bases hybrid bm25
```

Generate answers. One LLM call per question per prompt strategy.

```bash
python scripts/compare_prompts.py --split dev --limit 150 --retriever rerank --k 5
```

Label those answers, run both verifiers over them, and score the result. Only the judge
costs API calls, one per row.

```bash
python scripts/build_detection_set.py --runs data/runs/prompts_dev_rerank.jsonl
python scripts/run_rule_verifier.py --input data/processed/detection_eval.jsonl
python scripts/run_llm_judge.py --input data/processed/detection_eval.jsonl
python scripts/score_detection.py --input data/runs/judge_detection_eval.jsonl
```

## Running the demo

One command takes a question through all three stages and prints what each one did.

```bash
python demo.py                            # replays saved runs, no network needed
python demo.py --live                     # runs retrieval and the LLM for real
python demo.py AAPL/2004/page_36.pdf-2    # a specific question, by id
python demo.py "gross margin"             # or by part of the question text
python demo.py --random                   # a random question
```

With no arguments it runs two questions. The first one works: retrieval finds the right
table row, the model computes the answer, and every verifier agrees it is supported. The
second one fails: the row the question needs is not retrieved, the model answers anyway,
and all three checks flag it. It then scores both verifiers over all 560 labeled answers.

The default replays saved runs, so it needs no API keys and makes no network calls.
`--live` needs Weaviate and an LLM provider, and works on any of the 8,281 questions
instead of only the 140 that have saved runs.

## Results

Retrieval, on the held-out test split:

| retriever | Recall@1 | Recall@5 |
|---|---|---|
| BM25 | 43.4% | 77.1% |
| dense | 47.7% | 79.8% |
| hybrid | 50.7% | 82.1% |
| rerank | 52.6% | 84.3% |

Generation reaches 60-61% accuracy, and the four prompt strategies are within about a
point of each other.

Detection, over 560 labeled answers, counting any answer that does not follow from the
evidence as a hallucination:

| verifier | precision | recall | F1 |
|---|---|---|---|
| rule based | 0.38 | 0.25 | 0.30 |
| LLM judge, llama grading itself | 0.49 | 0.35 | 0.41 |
| LLM judge, claude-haiku grading | 0.61 | 0.49 | 0.54 |
| flagging everything | 0.29 | 1.00 | 0.45 |

Changing only the judging model moved F1 from 0.41 to 0.54, which was a larger effect than
any prompt change. Both verifiers only just beat the baseline of flagging every answer.

## Team

Swarali Degaonkar, Jaya Sriharshita Koneti, Kushal Pendekanti
