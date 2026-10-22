
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw_v4_expanded"

DOCUMENTS = [
    "eea_recycling_quality.txt",
    "eea_separate_collection.txt",
    "eea_circular_plastics.txt",
]

BOILERPLATE_PATTERNS = [
    r"\bcookie settings\b",
    r"\bprivacy policy\b",
    r"\bskip to main content\b",
    r"\bsubscribe to our newsletter\b",
    r"\ball rights reserved\b",
    r"\baccept all cookies\b",
]


def inspect_document(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = re.findall(r"\b[\w'-]+\b", text)
    replacement_chars = text.count("\ufffd")

    boilerplate_matches = {
        pattern: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern in BOILERPLATE_PATTERNS
    }

    return {
        "file": path.name,
        "characters": len(text),
        "words": len(words),
        "nonempty_lines": len(lines),
        "unique_line_ratio": (
            len(set(lines)) / len(lines) if lines else 0
        ),
        "replacement_characters": replacement_chars,
        "boilerplate_matches": {
            key: value
            for key, value in boilerplate_matches.items()
            if value
        },
        "first_lines": lines[:12],
        "last_lines": lines[-8:],
    }


def main() -> None:
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(RAW_DIR)

    failed = False

    for filename in DOCUMENTS:
        path = RAW_DIR / filename
        print("\n" + "=" * 88)

        if not path.is_file():
            print(f"MISSING: {path}")
            failed = True
            continue

        result = inspect_document(path)

        print(f"File: {result['file']}")
        print(f"Characters: {result['characters']:,}")
        print(f"Words: {result['words']:,}")
        print(f"Non-empty lines: {result['nonempty_lines']:,}")
        print(f"Unique-line ratio: {result['unique_line_ratio']:.3f}")
        print(f"Unicode replacement characters: {result['replacement_characters']}")
        print(f"Potential boilerplate: {result['boilerplate_matches'] or 'none detected'}")

        print("\n--- First lines ---")
        for line in result["first_lines"]:
            print(line[:250])

        print("\n--- Last lines ---")
        for line in result["last_lines"]:
            print(line[:250])

        if result["characters"] < 1500 or result["words"] < 200:
            print("WARNING: document may be incomplete")
            failed = True

        if result["replacement_characters"] > 0:
            print("WARNING: text contains Unicode replacement characters")
            failed = True

    print("\n" + "=" * 88)
    print("Inspection completed. No files or registry records were modified.")
    if failed:
        print("Some files require attention.")
    else:
        print("Basic structural checks passed; manually review the excerpts above.")


if __name__ == "__main__":
    main()
