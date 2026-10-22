"""Build a new, independent V4 Expanded retrieval index."""

from pathlib import Path

from src.embeddings.encoder import TextEncoder
from src.ingestion.text_loader import load_text_documents
from src.indexing.index_store import IndexStore
from src.pipeline import build_index_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw_v4_expanded"
INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_index_v4_expanded"
)

REQUIRED_INITIAL_SOURCES = {
    "eea_recycling_quality.txt",
    "eea_separate_collection.txt",
    "eea_circular_plastics.txt",
}


def main() -> None:
    print("=== V4 Expanded index preflight ===")
    print(f"Input directory : {DATA_DIR}")
    print(f"Output directory: {INDEX_DIR}")

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Expanded corpus directory not found: {DATA_DIR}"
        )

    # Accept all .txt corpus files while preserving the original
    # three validated EEA sources as mandatory inputs.
    actual_sources = {
        path.name
        for path in DATA_DIR.glob("*.txt")
        if path.is_file()
    }

    if not actual_sources:
        raise ValueError(
            f"No .txt source files found in {DATA_DIR}"
        )

    missing_required = sorted(
        REQUIRED_INITIAL_SOURCES - actual_sources
    )
    if missing_required:
        raise ValueError(
            "Required initial sources are missing: "
            f"{missing_required}"
        )

    # Never overwrite an existing index, even if incomplete.
    if INDEX_DIR.exists():
        raise FileExistsError(
            f"Output path already exists: {INDEX_DIR}. "
            "Nothing was overwritten. Inspect or choose a new destination."
        )

    documents = load_text_documents(DATA_DIR)

    if len(documents) != len(actual_sources):
        raise ValueError(
            f"Found {len(actual_sources)} .txt files but loaded "
            f"{len(documents)} documents. Review the loader and corpus."
        )

    loaded_sources = {document["source"] for document in documents}
    if loaded_sources != actual_sources:
        raise ValueError(
            "Loaded document sources do not match input files. "
            f"Missing from loaded documents: "
            f"{sorted(actual_sources - loaded_sources)}; "
            f"unexpected loaded sources: "
            f"{sorted(loaded_sources - actual_sources)}"
        )

    print(f"Validated input documents: {len(documents)}")
    for document in documents:
        print(
            f"  - {document['source']}: "
            f"{len(document['text'])} characters"
        )

    print("\n=== Build configuration ===")
    print("Chunk size      : 120")
    print("Overlap         : 15")
    print("Embedding model : intfloat/multilingual-e5-small")
    print("Overwrite       : disabled")

    encoder = TextEncoder(
        model_name="intfloat/multilingual-e5-small"
    )

    result = build_index_pipeline(
        data_dir=DATA_DIR,
        index_dir=INDEX_DIR,
        encoder=encoder,
        chunk_size=120,
        overlap=15,
    )

    # Reload saved artifacts and verify source/vector consistency.
    store = IndexStore(INDEX_DIR)
    index, chunks = store.load()

    indexed_sources = {chunk["source"] for chunk in chunks}
    if indexed_sources != actual_sources:
        raise RuntimeError(
            "Saved chunks do not contain exactly the input sources. "
            f"Missing: {sorted(actual_sources - indexed_sources)}; "
            f"unexpected: {sorted(indexed_sources - actual_sources)}"
        )

    if index.size != len(chunks):
        raise RuntimeError(
            f"Vector/chunk mismatch: {index.size} vectors, "
            f"{len(chunks)} chunks."
        )

    print("\n=== V4 Expanded build summary ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"Reloaded vectors: {index.size}")
    print(f"Reloaded chunks : {len(chunks)}")
    print(f"Indexed sources : {len(indexed_sources)}")

    print("\n=== Output artifacts ===")
    for filename in ("index.faiss", "chunks.json"):
        artifact = INDEX_DIR / filename
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(
                f"Missing or empty output artifact: {artifact}"
            )
        print(
            f"PASS | {artifact.relative_to(PROJECT_ROOT)} "
            f"| {artifact.stat().st_size:,} bytes"
        )

    print("\nPASS: V4 Expanded index built and verified.")
    print("Existing V3 and V4 index directories were not targeted.")


if __name__ == "__main__":
    main()
