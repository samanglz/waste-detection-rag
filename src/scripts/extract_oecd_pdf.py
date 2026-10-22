"""Extract and quality-check text from an OECD PDF in staging."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = (
    PROJECT_ROOT
    / "data"
    / "staging_v4_expanded"
    / "oecd_global_plastics_outlook.pdf"
)
DEFAULT_TEXT = (
    PROJECT_ROOT
    / "data"
    / "staging_v4_expanded"
    / "oecd_global_plastics_outlook.txt"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "staging_v4_expanded"
    / "oecd_global_plastics_outlook_extraction_report.json"
)


def normalize_page_text(text: str) -> str:
    """Normalize PDF line wraps and whitespace for retrieval."""
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from an OECD PDF and report extraction quality."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if not args.pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    # Never silently overwrite existing artifacts.
    for output_path in (args.output, args.report):
        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "Inspect it or choose a different output path."
            )

    reader = PdfReader(str(args.pdf))
    page_records = []
    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        normalized = normalize_page_text(raw_text)

        page_records.append(
            {
                "page": page_number,
                "raw_characters": len(raw_text),
                "normalized_characters": len(normalized),
                "word_count": len(normalized.split()),
                "empty": not bool(normalized),
            }
        )

        if normalized:
            extracted_pages.append(normalized)

    if not extracted_pages:
        raise RuntimeError(
            "No extractable text was found in the PDF."
        )

    full_text = "\n\n".join(extracted_pages).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(full_text + "\n", encoding="utf-8")

    nonempty_counts = [
        item["normalized_characters"]
        for item in page_records
        if not item["empty"]
    ]
    low_text_pages = [
        item["page"]
        for item in page_records
        if item["normalized_characters"] < 100
    ]
    empty_pages = [
        item["page"] for item in page_records if item["empty"]
    ]

    report = {
        "document_id": args.pdf.stem,
        "pdf_path": str(args.pdf.relative_to(PROJECT_ROOT)),
        "output_path": str(args.output.relative_to(PROJECT_ROOT)),
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "page_count": len(reader.pages),
        "nonempty_page_count": len(extracted_pages),
        "empty_pages": empty_pages,
        "pages_under_100_characters": low_text_pages,
        "total_characters": len(full_text),
        "total_words": len(full_text.split()),
        "mean_characters_per_nonempty_page": (
            round(sum(nonempty_counts) / len(nonempty_counts), 2)
            if nonempty_counts
            else 0
        ),
        "page_details": page_records,
        "quality_note": (
            "Automated extraction checks only. "
            "Review content and references before corpus inclusion."
        ),
    }

    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== PDF extraction summary ===")
    print(f"Pages                 : {report['page_count']}")
    print(f"Non-empty pages       : {report['nonempty_page_count']}")
    print(f"Empty pages           : {len(empty_pages)}")
    print(f"Pages under 100 chars : {len(low_text_pages)}")
    print(f"Extracted characters  : {report['total_characters']:,}")
    print(f"Extracted words       : {report['total_words']:,}")
    print(f"Text output           : {args.output}")
    print(f"QA report             : {args.report}")

    print("\n=== Sample pages ===")
    for page_number in (1, 2, 10, 50, 100, 150, len(reader.pages)):
        if page_number > len(page_records):
            continue
        detail = page_records[page_number - 1]
        print(
            f"Page {page_number}: "
            f"{detail['normalized_characters']} characters, "
            f"{detail['word_count']} words"
        )

    print("\n=== Extracted text preview ===")
    print(full_text[:2500])

    print("\n=== Automated checks ===")
    print(f"PASS | text file exists: {args.output.is_file()}")
    print(f"PASS | report exists: {args.report.is_file()}")
    print(f"PASS | non-empty extraction: {bool(full_text)}")


if __name__ == "__main__":
    main()
