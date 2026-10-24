# Waste Detection RAG

A modular Retrieval-Augmented Generation (RAG) system for recycling knowledge, designed to complement the [Smart Waste Detection](https://github.com/samanglz/smart-waste-detection) computer vision project and support a future multimodal AI pipeline.

The system retrieves evidence from recycling-related documents, combines semantic and lexical search, reranks candidate passages, and prepares grounded answers with source attribution.

## Features

- **Multilingual semantic retrieval** using `intfloat/multilingual-e5-small`
- **Dense retrieval** with FAISS
- **Keyword retrieval** with BM25
- **Hybrid retrieval** using Reciprocal Rank Fusion (RRF)
- **Cross-Encoder reranking** using `BAAI/bge-reranker-v2-m3`
- **Context construction** with source identifiers
- **Answer generation** through the configured Gemini model
- **Citation validation** and retrieval evaluation utilities
- **FastAPI interface** for health checks and question answering
- **Automated tests** for core components and selected integration workflows

## Architecture

```text
User Query
    |
    v
Dense Retrieval + BM25 Search
    |
    v
Hybrid Retrieval (RRF)
    |
    v
Cross-Encoder Reranking
    |
    v
Context Construction
    |
    v
Grounded Answer Generation
    |
    v
Citation Validation + Response
```

## Project Structure

```text
waste-detection-rag/
├── src/
│   ├── api/              # FastAPI application
│   ├── chunking/         # Document chunking
│   ├── embeddings/       # Text embedding models
│   ├── generation/       # Answer generation and RAG pipeline
│   ├── retrieval/        # Dense, BM25, and hybrid retrieval
│   ├── reranking/        # Cross-Encoder reranking
│   └── scripts/          # Index building and utility scripts
├── tests/                # Unit and integration tests
├── data/
│   ├── raw/              # Source documents
│   ├── raw_v4_expanded/  # Expanded source corpus
│   └── processed/        # Generated indexes and evaluation artifacts
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

Generated indexes, local datasets, and other artifacts may be excluded from version control. A fresh clone may therefore require additional data setup.

## Requirements

- Python 3.11
- Git
- A Gemini API key for answer generation
- The source corpus required to build the expanded retrieval index

The first run may download embedding and reranking model weights. Download time and memory usage depend on your machine and network connection.

## Installation

Clone the repository:

```bash
git clone https://github.com/samanglz/waste-detection-rag.git
cd waste-detection-rag
```

Create and activate a virtual environment.

**Windows — PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows — Git Bash**

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the active Python environment:

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
```

## Configuration

Create a local environment file from the example.

**Git Bash**

```bash
cp .env.example .env
```

**PowerShell**

```powershell
Copy-Item .env.example .env
```

Configure the required values locally:

```dotenv
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=gemini-3.8-flash

EMBEDDING_MODEL=intfloat/multilingual-e5-small
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_CANDIDATE_K=20
RERANKER_BATCH_SIZE=16

RAG_INDEX_DIR=data/processed/retrieval_index_v4_expanded
```

Replace the API-key placeholder with your own key in your local environment. Never commit `.env` or expose real credentials in source code, logs, screenshots, or public repositories. `.env.example` should contain placeholders only.

**Environment-loading note:** The current API reads configuration from environment variables. Creating `.env` alone will not load its values unless a dotenv loader is configured. In Git Bash, export the variables in the current shell before starting the API, or configure an environment loader.

Example for Git Bash:

```bash
export GEMINI_API_KEY='your_actual_api_key'
export GEMINI_MODEL='gemini-3.8-flash'
export EMBEDDING_MODEL='intfloat/multilingual-e5-small'
export RERANKER_MODEL='BAAI/bge-reranker-v2-m3'
export RERANKER_CANDIDATE_K=20
export RERANKER_BATCH_SIZE=16
export RAG_INDEX_DIR='data/processed/retrieval_index_v4_expanded'
```

Do not paste your real API key into the README or chat messages.

## Building the Expanded Retrieval Index

The expanded source corpus is expected under:

```text
data/raw_v4_expanded/
```

The index builder writes generated artifacts to:

```text
data/processed/retrieval_index_v4_expanded/
```

The required initial source documents include:

- `eea_recycling_quality.txt`
- `eea_separate_collection.txt`
- `eea_circular_plastics.txt`

After obtaining the approved source corpus, run the following command from the repository root:

```bash
python -m src.scripts.build_retrieval_index_v4_expanded
```

The configured embedding model is `intfloat/multilingual-e5-small`, with a chunk size of 120 and an overlap of 15.

The builder refuses to overwrite an existing index directory. Check the target directory and preserve existing artifacts before attempting a rebuild.

## Running the API

After installing dependencies, configuring environment variables, and ensuring the configured index exists, start the API:

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8001
```

The application initializes its embedding model, retriever, reranker, and answer generator during startup. Startup can fail if dependencies, configuration, model downloads, or index artifacts are missing.

### Health Check

```bash
curl -i http://127.0.0.1:8001/healthz
```

Expected successful response:

```json
{"status": "ok"}
```

### Interactive API Documentation

With the API running, open:

http://127.0.0.1:8001/docs

### Ask a Question

The question-answering endpoint is `POST /ask`.

```bash
curl -X POST "http://127.0.0.1:8001/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What materials can be recycled?","top_k":5,"max_chars":6000}'
```

The request requires a running API, a usable local index, initialized models, and a valid Gemini API key. Check the actual response schema in `/docs` for the fields returned by your current version.

## Testing

Run the default test suite:

```bash
python -m pytest
```

The repository's pytest configuration excludes tests marked as `integration` by default and ignores `tests/trash`.

To run integration tests explicitly while retaining the exclusion of `tests/trash`:

```bash
python -m pytest -o addopts="--ignore=tests/trash" -m integration
```

Some integration tests may require real index artifacts, model downloads, or external resources.

## Retrieval Evaluation

The project includes evaluation utilities for comparing retrieval approaches using metrics such as:

- **Hit Rate@k** — whether at least one relevant chunk appears in the top-k results
- **MRR@k** — the reciprocal rank of the first relevant result, averaged across queries
- **Recall@k** — the proportion of relevant chunks retrieved in the top-k results
- **nDCG@k** — ranking quality that accounts for the positions of relevant results

Evaluation requires labeled queries and relevant chunk IDs. Results are meaningful only when relevance annotations and index contents are consistent.

The retrieval strategies include:

1. Dense retrieval using multilingual E5 embeddings and FAISS
2. BM25 lexical retrieval
3. Hybrid retrieval using Reciprocal Rank Fusion
4. Hybrid retrieval followed by Cross-Encoder reranking

When reporting results, record the index version, model configuration, candidate depth, and evaluation dataset.

## Reproducibility and Data Management

- Keep source documents traceable to their origins.
- Version evaluation queries and relevance labels.
- Avoid using the test set to tune retrieval parameters.
- Preserve existing index artifacts when experimenting with chunking or retrieval settings.
- Treat expanded corpora and generated indexes as separate, versioned artifacts.
- Never commit API keys, virtual environments, caches, or local secrets.

A fresh clone may require the approved source corpus and a rebuilt retrieval index before the API can start.

## Limitations

- Answer quality depends on corpus coverage, retrieval quality, and source reliability.
- Retrieval metrics depend on the quality and completeness of relevance annotations.
- Model initialization may require substantial downloads and memory.
- Answer generation requires a valid Gemini API key.
- API behavior and performance should be verified in the target environment before deployment.

## Roadmap

- Expand and audit the traceable recycling knowledge corpus.
- Complete relevance labels and controlled retrieval evaluations.
- Improve grounding and citation validation.
- Evaluate reranking and context-construction strategies.
- Integrate retrieval with the Smart Waste Detection computer vision pipeline.
- Develop a multimodal recycling assistant and deployment workflow.

## Related Project

[Smart Waste Detection](https://github.com/samanglz/smart-waste-detection) — a computer vision project for detecting recyclable waste categories.

## License

Check the repository's license and the licensing terms of individual source documents before redistributing the project or its collected corpus.
