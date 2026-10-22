from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT / "data/staging_v4_expanded/"
    "epa_international_waste_agreements.complete.txt"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data/staging_v4_expanded/"
    "epa_international_waste_agreements.complete_v2.txt"
)

AGREEMENTS = [
    (
        "The Agreement Between the Government of America and the Government "
        "of Malaysia Concerning the Transboundary Movement of Hazardous "
        "Wastes from Malaysia to the United States (1995)",
        "https://www.epa.gov/hwgenerators/"
        "agreement-between-government-america-and-government-malaysia-"
        "concerning-transboundary",
    ),
    (
        "The Agreement on the Transboundary Movement of Hazardous Waste "
        "from Costa Rica to the United States (1997)",
        "https://www.epa.gov/hwgenerators/"
        "agreement-transboundary-movement-hazardous-waste-costa-rica-"
        "united-states-1997",
    ),
    (
        "The Agreement Between the Government of the United States of "
        "America and the Government of the Republic of the Philippines "
        "Concerning the Transboundary Movement of Hazardous Wastes from "
        "the Philippines to the United States (2001)",
        "https://www.epa.gov/hwgenerators/"
        "agreement-between-government-united-states-america-and-"
        "government-republic-philippines",
    ),
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT_FILE}"
        )

    original = INPUT_FILE.read_text(encoding="utf-8")
    fixed_lines = []
    repaired = set()

    for line in original.splitlines():
        matches = [(title, url) for title, url in AGREEMENTS if url in line]

        if len(matches) > 1:
            raise ValueError(f"Multiple agreement URLs on one line: {line}")

        if matches:
            title, url = matches[0]
            if url in repaired:
                raise ValueError(f"Duplicate URL: {url}")
            fixed_lines.append(f"- [{title}]({url})")
            repaired.add(url)
        else:
            fixed_lines.append(line)

    expected_urls = {url for _, url in AGREEMENTS}
    if repaired != expected_urls:
        raise ValueError(
            f"Agreement link lines missing: {expected_urls - repaired}"
        )

    fixed_text = "\n".join(fixed_lines).rstrip() + "\n"

    for title, url in AGREEMENTS:
        if fixed_text.count(f"[{title}]({url})") != 1:
            raise ValueError(f"Markdown link validation failed: {title}")
        if fixed_text.count(url) != 1:
            raise ValueError(f"URL must occur exactly once: {url}")

    OUTPUT_FILE.write_text(fixed_text, encoding="utf-8")

    if OUTPUT_FILE.read_text(encoding="utf-8") != fixed_text:
        raise IOError("Read-back verification failed.")

    print("Output:", OUTPUT_FILE)
    print("Original SHA-256:", sha256(INPUT_FILE))
    print("Fixed SHA-256:", sha256(OUTPUT_FILE))
    print("Characters:", len(fixed_text))
    print("Words:", len(fixed_text.split()))
    print("Agreement links validated: 3 / 3")
    print("Read-back verification: PASSED")
    print("Registry update: NOT performed")
    print("Validation status: pending_manual_review")

if __name__ == "__main__":
    main()
