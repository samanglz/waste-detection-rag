import json
from pathlib import Path
from typing import Any


def load_text_documents(
    directory: str | Path,
) -> list[dict[str, Any]]:
    """Load non-empty UTF-8 text documents and optional source metadata."""

    data_dir = Path(directory)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    manifest_path = data_dir / "manifest.json"
    manifest: dict[str, Any] = {}

    if manifest_path.is_file():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Invalid ingestion manifest: {manifest_path}"
            ) from exc

        if not isinstance(manifest, dict):
            raise ValueError("Ingestion manifest must be a JSON object")

        for source, metadata in manifest.items():
            if not isinstance(source, str) or not isinstance(metadata, dict):
                raise ValueError(
                    "Manifest entries must map filenames to metadata objects"
                )

    documents = []

    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        metadata = manifest.get(path.name, {})

        documents.append(
            {
                "source": path.name,
                "text": text,
                "metadata": metadata.copy(),
            }
        )

    return documents
