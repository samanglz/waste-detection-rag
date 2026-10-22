
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.source_registry import load_registry, save_registry, upsert_source

RAW_DIR = PROJECT_ROOT / "data" / "raw_v4_expanded"
REGISTRY_PATH = PROJECT_ROOT / "metadata" / "source_registry.json"
MANIFEST_PATH = RAW_DIR / "manifest.json"

SOURCE_IDS = [
    "eea_recycling_quality",
    "eea_separate_collection",
    "eea_circular_plastics",
]

MIN_TEXT_LENGTH = 1500


def extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    for tag in soup([
        "script", "style", "nav", "footer", "header",
        "aside", "noscript", "svg", "form"
    ]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        main = soup

    lines = [
        line.strip()
        for line in main.get_text("\n").splitlines()
        if line.strip()
    ]

    # Remove consecutive duplicate lines while preserving document order.
    clean_lines = []
    previous = None
    for line in lines:
        if line != previous:
            clean_lines.append(line)
        previous = line

    return title, "\n".join(clean_lines).strip()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry(REGISTRY_PATH)
    sources_by_id = {
        item["document_id"]: item for item in registry["sources"]
    }

    missing = [sid for sid in SOURCE_IDS if sid not in sources_by_id]
    if missing:
        raise ValueError(f"Sources missing from registry: {missing}")

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Existing manifest.json must contain a JSON object")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "WasteDetectionRAG/1.0 (educational research)"
    })

    successes = 0
    failures = 0

    for document_id in SOURCE_IDS:
        source = sources_by_id[document_id]
        print(f"\nCollecting: {document_id}")
        print(f"URL: {source['url']}")

        try:
            response = session.get(
                source["url"],
                timeout=30,
                allow_redirects=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                raise ValueError(
                    f"Expected HTML but received Content-Type={content_type!r}"
                )

            page_title, text = extract_main_text(response.text)

            if len(text) < MIN_TEXT_LENGTH:
                raise ValueError(
                    f"Extracted text is too short: {len(text)} characters"
                )

            filename = f"{document_id}.txt"
            output_path = RAW_DIR / filename

            # Keep an existing file intact unless the user explicitly reviews it.
            if output_path.exists():
                raise FileExistsError(
                    f"{output_path} already exists; inspect it before replacing"
                )

            output_path.write_text(text + "\n", encoding="utf-8")
            digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
            retrieved_at = datetime.now(timezone.utc).isoformat()

            source["status"] = "downloaded"
            source["local_path"] = str(output_path.relative_to(PROJECT_ROOT))
            source["sha256"] = digest
            source["retrieved_at"] = retrieved_at
            source["final_url"] = response.url
            source["http_status"] = response.status_code
            source["extracted_chars"] = len(text)
            source["page_title"] = page_title

            manifest[filename] = {
                "document_id": document_id,
                "title": source["title"],
                "publisher": source["publisher"],
                "url": source["url"],
                "source_type": source["source_type"],
                "language": source["language"],
                "topics": source["topics"],
                "sha256": digest,
                "retrieved_at": retrieved_at,
                "extraction_status": "downloaded_needs_review",
            }

            upsert_source(registry, source)
            successes += 1

            print(f"Result: downloaded")
            print(f"HTTP status: {response.status_code}")
            print(f"Extracted characters: {len(text)}")
            print(f"Saved: {output_path}")
            print(f"SHA-256: {digest}")

        except Exception as exc:
            failures += 1
            print(f"Result: FAILED — {type(exc).__name__}: {exc}")

    save_registry(REGISTRY_PATH, registry)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nCollection summary")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Raw directory: {RAW_DIR}")
    print(f"Registry: {REGISTRY_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("Index directories were not modified.")


if __name__ == "__main__":
    main()
