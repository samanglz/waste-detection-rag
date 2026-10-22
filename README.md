# Waste Detection RAG

A modular Retrieval-Augmented Generation (RAG) project for
recycling knowledge, designed for future integration with
the Smart Waste Detection computer-vision project.

## Project Status

The project implements document loading, text chunking,
dense retrieval, BM25 retrieval, hybrid retrieval with
Reciprocal Rank Fusion (RRF), FAISS indexing, context
construction, prompt building, citation validation, and
retrieval evaluation.

The project also includes source registries, corpus
validation scripts, and an expanded recycling corpus.

LLM generation integrations are available in the codebase.
End-to-end production deployment and integration with the
Smart Waste Detection project remain future work.

## Architecture

1. Source documents and source metadata
2. Document loading and text chunking
3. Text embeddings using multilingual E5
4. FAISS dense retrieval
5. BM25 keyword retrieval
6. Hybrid retrieval using Reciprocal Rank Fusion
7. Context construction and prompt preparation
8. Citation validation and optional LLM generation
9. Retrieval evaluation and diagnostics

## Data and Index Versions

- `data/raw/`: original recycling knowledge corpus
- `data/raw_v4_expanded/`: expanded English source corpus
- `data/evaluation/`: retrieval evaluation queries
- `data/processed/`: generated indexes and chunks;
  generated artifacts are excluded from Git
- `metadata/`: source registry and source-validation reports
- `reports/`: retrieval evaluation and diagnostic reports

V3, V4, and V4 Expanded indexes are separate artifacts.
The V4 Expanded build script refuses to overwrite an
existing output directory.

## Evaluation

The project supports evaluation of Dense, BM25, and
Hybrid retrieval using retrieval metrics such as Hit Rate,
MRR, Recall, and nDCG.

Evaluation results depend on the query dataset, relevance
labels, retrieval configuration, and evaluation cutoff.
Consult the corresponding reports for the exact settings
and results.

## Tests

Run the unit-test suite from the project root:

```bash
python -m pytest
```

Integration tests are excluded by default through
`pytest.ini`. Run them separately when the required
indexes, models, and external resources are available:

```bash
python -m pytest -m integration
```

## Reproducibility

Install dependencies using `requirements.txt`.
Source documents, manifests, evaluation datasets, code,
tests, and selected reports are version-controlled.
Generated indexes and local runtime artifacts are not
committed by default.

## Future Work

- Expand and validate the source corpus
- Improve retrieval using controlled evaluation
- Evaluate answer grounding and citation correctness
- Add reranking where evaluation justifies it
- Integrate recycling knowledge with object detection
- Build a deployable multimodal RAG API
