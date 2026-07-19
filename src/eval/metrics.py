"""Evaluation metrics for the FinQA RAG + hallucination-detection pipeline.

To implement once the dataset is ready:
  - retrieval:   recall_at_k
  - generation:  numeric-tolerant exact match (FinQA answers are mostly numbers)
  - detection:   precision / recall / f1
"""
