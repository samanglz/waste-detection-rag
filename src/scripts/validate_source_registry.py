import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.source_registry import load_registry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the waste-RAG source registry."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "metadata" / "source_registry.json",
        help="Path to the source registry JSON file.",
    )
    args = parser.parse_args()

    try:
        registry = load_registry(args.registry)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Registry validation failed: {exc}", file=sys.stderr)
        return 1

    sources = registry.get("sources", [])
    statuses = Counter(record["status"] for record in sources)
    publishers = Counter(record["publisher"] for record in sources)
    topics = Counter(
        topic
        for record in sources
        for topic in record["topics"]
    )

    print("Source registry validation: PASSED")
    print(f"Registry: {args.registry}")
    print(f"Total sources: {len(sources)}")
    print("\\nSources by status:")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")

    print("\\nTop publishers:")
    for publisher, count in publishers.most_common(10):
        print(f"  {publisher}: {count}")

    print("\\nTopics:")
    for topic, count in sorted(topics.items()):
        print(f"  {topic}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
