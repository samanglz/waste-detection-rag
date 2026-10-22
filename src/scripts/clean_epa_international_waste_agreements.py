from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = PROJECT_ROOT / "data" / "staging_v4_expanded"

INPUT_FILE = STAGING_DIR / "epa_international_waste_agreements.txt"
OUTPUT_FILE = STAGING_DIR / "epa_international_waste_agreements.cleaned.txt"
REPORT_FILE = STAGING_DIR / "epa_international_waste_agreements_cleaning_report.json"

EXPECTED_HEADINGS = [
    "The Basel Convention",
    "Basel Convention Requirements",
    "Basel Convention Restrictions",
    "Implications of the Basel Convention for U.S. Importers & Exporters",
    "Organization for Economic Cooperation and Development (OECD) Council Decision",
    "Bilateral Agreements Between the United States and Other Countries",
    "Canada",
    "Mexico",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a cleaned copy and QA report for the EPA waste-agreements text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Input text file; defaults to the staging EPA source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Cleaned output path; must not already exist.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_FILE,
        help="QA report path; must not already exist.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite cleaned output: {output_path}"
        )

    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite QA report: {report_path}"
        )

    original = input_path.read_text(encoding="utf-8")
    cleaned = original.replace("\r\n", "\n").replace("\r", "\n")

    # Apply only a confirmed, conservative spacing correction.
    correction_pattern = re.compile(r"\bDecisionon\b")
    correction_count = len(correction_pattern.findall(cleaned))
    cleaned = correction_pattern.sub("Decision on", cleaned)

    # Normalize trailing whitespace and excessive blank lines only.
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"

    missing_headings = [
        heading for heading in EXPECTED_HEADINGS
        if heading.casefold() not in cleaned.casefold()
    ]

    urls = re.findall(r"https?://[^\s)>\]]+", cleaned)
    html_entities = re.findall(r"&#(?:x[0-9a-fA-F]+|\d+);", cleaned)
    replacement_characters = cleaned.count("\ufffd")

    # This is a warning, not proof that source links were originally present.
    link_warning = (
        "No URLs were found in the copied text. Verify source links against "
        "the official EPA webpage; do not infer missing links."
        if not urls
        else None
    )

    report = {
        "document_id": "epa_international_waste_agreements",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256_text(original),
        "cleaned_sha256": sha256_text(cleaned),
        "input_characters": len(original),
        "cleaned_characters": len(cleaned),
        "input_words": len(original.split()),
        "cleaned_words": len(cleaned.split()),
        "corrections": {
            "Decisionon_to_Decision_on": correction_count,
        },
        "expected_headings_found": [
            heading for heading in EXPECTED_HEADINGS
            if heading.casefold() in cleaned.casefold()
        ],
        "missing_expected_headings": missing_headings,
        "url_count": len(urls),
        "urls_found": urls,
        "html_entity_count": len(html_entities),
        "replacement_character_count": replacement_characters,
        "warnings": [
            warning for warning in [
                link_warning,
                (
                    "Some expected headings were not found; inspect the source text."
                    if missing_headings else None
                ),
                (
                    "The text has not been legally cleared for full-text storage."
                ),
            ]
            if warning
        ],
        "validation_status": "pending_manual_review",
        "license_status": "verify_before_full_text_storage",
    }

    # Write outputs only after all checks have completed.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(cleaned, encoding="utf-8")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Verify output files after writing.
    if output_path.read_text(encoding="utf-8") != cleaned:
        raise RuntimeError("Cleaned output read-back verification failed.")

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    if saved_report["cleaned_sha256"] != sha256_text(
        output_path.read_text(encoding="utf-8")
    ):
        raise RuntimeError("Output hash verification failed.")

    print("Cleaned copy saved:", output_path)
    print("QA report saved:", report_path)
    print("Input characters:", f"{len(original):,}")
    print("Output characters:", f"{len(cleaned):,}")
    print("Words:", f"{len(cleaned.split()):,}")
    print("Decisionon corrections:", correction_count)
    print("URLs found:", len(urls))
    print("Missing expected headings:", missing_headings)
    print("Read-back and hash verification: PASSED")
    print("Validation status: pending_manual_review")
    print("License status: verify_before_full_text_storage")


if __name__ == "__main__":
    main()
