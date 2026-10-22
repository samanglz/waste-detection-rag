from pathlib import Path
import hashlib
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = PROJECT_ROOT / "data" / "staging_v4_expanded"

INPUT_FILE = STAGING_DIR / "epa_international_waste_agreements.txt"
OUTPUT_FILE = STAGING_DIR / "epa_international_waste_agreements.complete.txt"

SECTION_HEADING = "Costa Rica, Malaysia, and the Philippines"

APPENDED_SECTION = """### Costa Rica, Malaysia, and the Philippines

Costa Rica, Malaysia and the Philippines have separate bilateral agreements with the United States. The United States may receive waste for recycling or disposal from Costa Rica, Malaysia, and the Philippines, but may not export waste to these countries.

- [The Agreement Between the Government of America and the Government of Malaysia Concerning the Transboundary Movement of Hazardous Wastes from Malaysia to the United States (1995)](https://www.epa.gov/hwgenerators/agreement-between-government-america-and-government-malaysia-concerning-transboundary)
- [The Agreement on the Transboundary Movement of Hazardous Waste from Costa Rica to the United States (1997)](https://www.epa.gov/hwgenerators/agreement-transboundary-movement-hazardous-waste-costa-rica-united-states-1997)
- [The Agreement Between the Government of the United States of America and the Government of the Republic of the Philippines Concerning the Transboundary Movement of Hazardous Wastes from the Philippines to the United States (2001)](https://www.epa.gov/hwgenerators/agreement-between-government-united-states-america-and-government-republic-philippines)
"""


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_FILE}"
        )

    original = INPUT_FILE.read_text(encoding="utf-8")

    if SECTION_HEADING.casefold() in original.casefold():
        raise ValueError(
            "The Costa Rica/Malaysia/Philippines section already exists "
            "in the input. Inspect manually; refusing to append twice."
        )

    complete = original.rstrip() + "\n\n" + APPENDED_SECTION.strip() + "\n"

    urls = re.findall(r"https?://[^\s)>\]]+", complete)
    required_urls = [
        "https://www.epa.gov/hwgenerators/agreement-between-government-america-and-government-malaysia-concerning-transboundary",
        "https://www.epa.gov/hwgenerators/agreement-transboundary-movement-hazardous-waste-costa-rica-united-states-1997",
        "https://www.epa.gov/hwgenerators/agreement-between-government-united-states-america-and-government-republic-philippines",
    ]

    missing_urls = [url for url in required_urls if url not in complete]
    if missing_urls:
        raise ValueError(f"Required URLs missing: {missing_urls}")

    OUTPUT_FILE.write_text(complete, encoding="utf-8")
    saved = OUTPUT_FILE.read_text(encoding="utf-8")

    if saved != complete:
        raise RuntimeError("Read-back verification failed.")

    print("Complete copy saved:", OUTPUT_FILE)
    print("Original preserved:", INPUT_FILE)
    print("Original SHA-256:", hashlib.sha256(original.encode("utf-8")).hexdigest())
    print("Complete SHA-256:", hashlib.sha256(saved.encode("utf-8")).hexdigest())
    print("Characters:", f"{len(saved):,}")
    print("Words:", f"{len(saved.split()):,}")
    print("URLs found:", len(urls))
    print("Required new URLs present:", len(required_urls) - len(missing_urls), "/ 3")
    print("Read-back verification: PASSED")
    print("Registry update: NOT performed")
    print("Validation status: pending_manual_review")


if __name__ == "__main__":
    main()
