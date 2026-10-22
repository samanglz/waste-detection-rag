"""Build an independent V4 retrieval index."""

import argparse
from pathlib import Path

from src.embeddings.encoder import TextEncoder
from src.pipeline import build_index_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "retrieval_index_v4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the V4 retrieval index."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing V4 index.",
    )
    args = parser.parse_args()

    existing_files = [
        INDEX_DIR / "index.faiss",
        INDEX_DIR / "chunks.json",
    ]
    if any(path.exists() for path in existing_files) and not args.overwrite:
        raise FileExistsError(
            f"V4 index already exists at {INDEX_DIR}. "
            "Inspect it first, or pass --overwrite to rebuild it."
        )

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Input directory not found: {DATA_DIR}")

    print(f"Input directory : {DATA_DIR}")
    print(f"Output directory: {INDEX_DIR}")
    print("Chunk size      : 120")
    print("Overlap         : 15")
    print("Embedding model : intfloat/multilingual-e5-small")
    print()

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

    print("\n=== V4 build summary ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nSaved files:")
    for filename in ("index.faiss", "chunks.json"):
        path = INDEX_DIR / filename
        print(f"{path.name}: {'OK' if path.is_file() else 'MISSING'}")

    if not all(path.is_file() for path in existing_files):
        raise RuntimeError("V4 index build did not produce all expected files.")


if __name__ == "__main__":
    main()
