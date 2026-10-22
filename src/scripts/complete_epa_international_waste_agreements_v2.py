from pathlib import Path
import hashlib
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = PROJECT_ROOT / "data" / "staging_v4_expanded"

INPUT_FILE = STAGING_DIR / "epa_international_waste_agreements.txt"
OUTPUT_FILE = STAGING_DIR / "epa_international_waste_agreements.complete.txt"

SECTION_HEADING = "Costa Rica, Malaysia, and the Philippines"

SECTION = """### Costa Rica, Malaysia, and the Philippines

Costa Rica, Malaysia and the Philippines have separate bilateral agreements with the United States. The United States may receive waste for recycling or disposal from Costa Rica, Malaysia, and the Philippines, but may not export waste to these countries.

- [The Agreement Between the Government of America and the Government of Malaysia Concerning the Transboundary Movement of Hazardous Wastes from Malaysia to the United States (1995)](https://www.epa.gov/hwgenerators/agreement-between-government-america-and-government-malaysia-concerning-transboundary)
- [The Agreement on the Transboundary Movement of Hazardous Waste from Costa Rica to the United States (1997)](https://www.epa.gov/hwgenerators/agreement-transboundary-movement-hazardous-waste-costa-rica-united-states-1997)
- [The Agreement Between the Government of the United States of America and the Government of the Republic of the Philippines Concerning the Transboundary Movement of Hazardous Wastes from the Philippines to the United States (2001)](https://www.epa.gov/hwgenerators/agreement-between-government-united-states-america-and-government-republic-philippines)
"""

REQUIRED_URLS = [
    "https://www.epa.gov/hwgenerators/agreement-between-government-america-and-government-malaysia-concerning-transboundary",
    "https://www.epa.gov/hwgenerators/agreement-transboundary-movement-hazardous-waste-costa-rica-united-states-1997",
    "https://www.epa.gov/hwgenerators/agreement-between-government-united-states-america-and-government-republic-philippines",
]


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    if OUTPUT_FILE.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT_FILE}")

    original = INPUT_FILE.read_text(encoding="utf-8")

    # Do not append again if all three URLs already exist.
    missing_urls = [url for url in REQUIRED_URLS if url not in original]
    if not missing_urls:
        raise ValueError(
            "All three agreement URLs already exist in the input. "
            "No changes needed."
        )

    # If any required URL is already present, stop for manual review rather
    # than creating a partially duplicated section.
    present_urls = [url for url in REQUIRED_URLS if url in original]
    if present_urls:
        raise ValueError(
            "Only some agreement URLs are present. Inspect manually before "
            f"appending. Present URLs: {present_urls}"
        )

    # The heading alone is not enough to establish that the section is complete.
    if SECTION_HEADING.casefold() not in original.casefold():
        complete = original.rstrip() + "\n\n" + SECTION.strip() + "\n"
    else:
        # Existing text already contains a mention of this heading/country group.
        # Add the detailed subsection at the end, clearly separated.
        complete = (
            original.rstrip()
            + "\n\n"
            + SECTION.strip()
            + "\n"
        )

    if any(url not in complete for url in REQUIRED_URLS):
        raise RuntimeError("Required agreement URLs missing from output.")

    OUTPUT_FILE.write_text(complete, encoding="utf-8")
    saved = OUTPUT_FILE.read_text(encoding="utf-8")
    if saved != complete:
        raise RuntimeError("Read-back verification failed.")

    found_urls = re.findall(r"https?://[^\s)>\]]+", saved)
    print("Complete copy saved:", OUTPUT_FILE)
    print("Original preserved:", INPUT_FILE)
    print("Original SHA-256:", hashlib.sha256(original.encode("utf-8")).hexdigest())
    print("Complete SHA-256:", hashlib.sha256(saved.encode("utf-8")).hexdigest())
    print("Characters:", f"{len(saved):,}")
    print("Words:", f"{len(saved.split()):,}")
    print("URLs found:", len(found_urls))
    print("Required agreement URLs:", sum(url in saved for url in REQUIRED_URLS), "/ 3")
    print("Read-back verification: PASSED")
    print("Registry update: NOT performed")
    print("Validation status: pending_manual_review")


if __name__ == "__main__":
    main()
