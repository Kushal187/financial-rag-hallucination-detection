# Processed data schema

The raw FinQA files (`data/raw/finqa/{train,dev,test}.json`) are converted into two
JSONL files under `data/processed/`. Every downstream stage loads data through
[src/data/load_data.py](../src/data/load_data.py) so the schema lives in exactly one place.

The two files enforce a hard split:

- **Corpus** (`finqa_chunks.jsonl`) — everything the system is *allowed to see*. This is
  what gets embedded and pushed into the vector DB.
- **Answer key** (`finqa_qa.jsonl`) — the ground truth used only for grading. It is **never**
  embedded or shown to the retriever/generator.

Both files are written by `scripts/build_corpus.py` and tracked in git.

Keeping them in separate files is the physical guarantee that the system can't peek at the
answer key.

## Corpus record — `finqa_chunks.jsonl`

One row per retrievable chunk (one sentence of page text, or one linearized table row).

| Field        | Type  | Description                                                        |
|--------------|-------|--------------------------------------------------------------------|
| `chunk_id`   | str   | Globally unique id, `"{doc_id}::{local_id}"`.                       |
| `doc_id`     | str   | Source filing page (FinQA `filename`), e.g. `"ADI/2009/page_49.pdf"`. Retrieval filters on this for per-document scope. |
| `local_id`   | str   | `"text_N"` or `"table_N"`. **Matches FinQA `gold_inds` keys** so retrieval is gradeable. |
| `chunk_type` | str   | `"text"` or `"table_row"`.                                          |
| `content`    | str   | The text that gets embedded and shown to the model.                |
| `position`   | int   | Order of the chunk within its page (for optional neighbor windows).|
| `split`      | str   | Source FinQA split (`train`/`dev`/`test`). Lets ingestion scope the vector DB to a split. |

### The `local_id` scheme (must match FinQA `gold_inds` exactly)

- `text_N` = `(pre_text + post_text)[N]`. `text_0` is `pre_text[0]`; once the index passes
  `len(pre_text)`, it continues into `post_text`.
- `table_N` = `table[N]`, linearized against `table[0]` as the column header. Every row is
  emitted as a chunk, **including `table_0`** (the header linearized against itself), because
  FinQA's `gold_inds` can cite `table_0` (common in 2-column key/value tables where the first
  row is real data).

### Table row linearization

A table row is turned into a sentence with FinQA's template:

```
{header[0]} the {row[0]} of {header[j]} is {row[j]} ; the {row[0]} of {header[k]} is {row[k]} ; ... .
```

The `header[0]` prefix is included only when it is non-empty.

## Answer-key record — `finqa_qa.jsonl`

One row per question.

| Field               | Type            | Description                                                    |
|---------------------|-----------------|----------------------------------------------------------------|
| `id`                | str             | FinQA question id, e.g. `"ADI/2009/page_49.pdf-1"`.            |
| `question`          | str             | The natural-language question.                                 |
| `doc_id`            | str             | The page it belongs to (= a corpus `doc_id`).                  |
| `gold_answer`       | str             | Gold answer string (may be `""` for yes/no questions).         |
| `gold_answer_exe`   | float \| str    | Executed gold answer. Usually numeric; sometimes `"yes"`/`"no"`.|
| `gold_evidence_ids` | list[str]       | The `local_id`s that constitute the gold evidence (from `gold_inds`). Used for retrieval Recall@k and grounding checks. |
| `gold_program`      | str             | Gold reasoning program, e.g. `"divide(3.8, divide(100, 100))"`.|
| `split`             | str             | Source FinQA split (`train`/`dev`/`test`).                     |

## Chunk deduplication (one row per document fact)

Chunks are **document-level** facts, not question-level. Because FinQA has ~3 questions per
filing page, `scripts/build_corpus.py` deduplicates by `doc_id`: each document's chunks are
emitted **exactly once**, no matter how many questions reference that page. The build asserts
every `(doc_id, local_id)` pair is unique. (Without this, a page's chunks were duplicated ~3×,
which crowded distinct chunks out of the retrieval top-k and inflated the vector store.)

## Scope: train + dev + test

Both the processed files and the vector DB are **derived artifacts** built from the true
source of truth, `data/raw/finqa/*.json` (tracked in git). They are scoped **identically** to
all three splits. Deduplicated, the full corpus is ~86k chunks (train 65,379 · dev 9,177 ·
test 11,865), comfortably under the free Weaviate Cloud sandbox limit of **100,000 objects**.

Retrieval is **per-document**: a question only ever searches its own filing page (the `doc_id`
filter), matching FinQA's task definition — a question's gold evidence always lives on one page.
So the roles of the splits are about *which questions you run through the pipeline*, not about
what the DB contains:

- **test** — held-out final evaluation (report Recall@k / accuracy here).
- **dev** — tune retrieval choices (embedding model, k, hybrid alpha).
- **train** — development workbench: few-shot prompt examples, and — if the hallucination
  detector is a *trained classifier* — the split you run through retrieve→generate to produce
  labeled training data. Indexing it costs nothing extra now that it fits, and future-proofs
  that supervised path. (An *unsupervised* detector never queries train; indexing it is then
  just harmless.)

Build both artifacts:

    python scripts/build_corpus.py --splits train dev test
    python scripts/ingest_weaviate.py --recreate --split train dev test

## Integrity invariant

For every answer-key record, `gold_evidence_ids` is a subset of the `local_id`s of that
document's corpus chunks. `scripts/build_corpus.py` asserts this while building — if any
gold evidence id has no matching chunk, the build fails loudly, because retrieval could
never be scored correctly in that case.
