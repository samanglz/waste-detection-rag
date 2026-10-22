
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.source_registry import load_registry

RAW_DIR = PROJECT_ROOT / "data" / "raw_v4_expanded"
STAGING_DIR = PROJECT_ROOT / "data" / "staging_v4_expanded"
REGISTRY_PATH = PROJECT_ROOT / "metadata" / "source_registry.json"

SOURCE_IDS = [
    "eea_recycling_quality",
    "eea_separate_collection",
    "eea_circular_plastics",
]

NOISE_EXACT = {
    "share",
    "image",
    "html",
    "text",
    "ris",
    "bibtex",
    "related content",
    "related publications",
}


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_blocks(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    page_title = (
        soup.title.get_text(" ", strip=True)
        if soup.title else ""
    )

    for tag in soup([
        "script", "style", "nav", "footer", "header",
        "aside", "noscript", "svg", "form"
    ]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        main = soup

    blocks = []
    seen = set()

    for element in main.find_all(
        ["h1", "h2", "h3", "h4", "h5", "p", "li", "blockquote"]
    ):
        text = normalize_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if text.casefold() in NOISE_EXACT:
            continue
        if text in seen:
            continue

        seen.add(text)
        blocks.append(text)

    return page_title, "\n\n".join(blocks)


def main() -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry(REGISTRY_PATH)
    sources = {
        item["document_id"]: item
        for item in registry["sources"]
    }

    session = requests.Session()
    session.headers.update({
        "User-Agent": "WasteDetectionRAG/1.0 (educational research)"
    })

    for document_id in SOURCE_IDS:
        if document_id not in sources:
            print(f"MISSING REGISTRY ENTRY: {document_id}")
            continue

        source = sources[document_id]
        print("\n" + "=" * 80)
        print("Source:", document_id)

        try:
            response = session.get(
                source["url"],
                timeout=30,
                allow_redirects=True,
            )
            response.raise_for_status()

            if "html" not in response.headers.get(
                "Content-Type", ""
            ).lower():
                raise ValueError("Response is not HTML")

            page_title, text = extract_blocks(response.text)

            if len(text) < 1500:
                raise ValueError(
                    f"Extracted text too short: {len(text)} characters"
                )

            output_path = STAGING_DIR / f"{document_id}.txt"
            output_path.write_text(text + "\n", encoding="utf-8")

            print("HTTP:", response.status_code)
            print("Page title:", page_title)
            print("Characters:", len(text))
            print("Words:", len(re.findall(r"\b[\w'-]+\b", text)))
            print("Saved to:", output_path)
            print("\n--- First 10 blocks ---")

            for block in text.split("\n\n")[:10]:
                print("-", block[:300])

            print("\n--- Last 5 blocks ---")
            for block in text.split("\n\n")[-5:]:
                print("-", block[:250])

        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")

    print("\nStaging extraction complete.")
    print("Original raw files and index directories were not modified.")


if __name__ == "__main__":
    main()
