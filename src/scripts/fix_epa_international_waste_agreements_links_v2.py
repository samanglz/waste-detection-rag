from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/staging_v4_expanded/epa_international_waste_agreements.complete_v2.txt"
OUTPUT = ROOT / "data/staging_v4_expanded/epa_international_waste_agreements.complete_v3.txt"

LINKS = [
    (
        "The Agreement Between the Government of America and the Government of Malaysia Concerning the Transboundary Movement of Hazardous Wastes from Malaysia to the United States (1995)",
        "https://www.epa.gov/hwgenerators/agreement-between-government-america-and-government-malaysia-concerning-transboundary",
    ),
    (
        "The Agreement on the Transboundary Movement of Hazardous Waste from Costa Rica to the United States (1997)",
        "https://www.epa.gov/hwgenerators/agreement-transboundary-movement-hazardous-waste-costa-rica-united-states-1997",
    ),
    (
        "The Agreement Between the Government of the United States of America and the Government of the Republic of the Philippines Concerning the Transboundary Movement of Hazardous Wastes from the Philippines to the United States (2001)",
        "https://www.epa.gov/hwgenerators/agreement-between-government-united-states-america-and-government-republic-philippines",
    ),
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    text = INPUT.read_text(encoding="utf-8")
    lines = text.splitlines()
    output_lines = []
    replaced = set()

    for line in lines:
        matched = [(title, url) for title, url in LINKS if url in line]

        if not matched:
            output_lines.append(line)
            continue

        if len(matched) != 1:
            raise ValueError(f"Ambiguous line: {line}")

        title, url = matched[0]

        if url in replaced:
            raise ValueError(f"Duplicate URL line: {url}")

        # Replace the entire malformed line with standard Markdown.
        output_lines.append(f"- [{title}]({url})")
        replaced.add(url)

    expected = {url for _, url in LINKS}
    if replaced != expected:
        raise ValueError(f"Missing links: {expected - replaced}")

    result = "\n".join(output_lines).rstrip() + "\n"

    # Strict validation: exact standard Markdown, no escaped brackets,
    # and each URL appears once.
    for title, url in LINKS:
        expected_link = f"- [{title}]({url})"
        if result.count(expected_link) != 1:
            raise ValueError(f"Malformed or missing link: {title}")
        if result.count(url) != 1:
            raise ValueError(f"URL count is not one: {url}")

    if r"]\(" in result:
        raise ValueError("Escaped Markdown syntax remains in the output.")

    OUTPUT.write_text(result, encoding="utf-8")

    if OUTPUT.read_text(encoding="utf-8") != result:
        raise IOError("Read-back verification failed.")

    print("Output:", OUTPUT)
    print("Input SHA-256:", sha256(INPUT))
    print("Output SHA-256:", sha256(OUTPUT))
    print("Links repaired and verified:", len(replaced), "/ 3")
    print("Read-back verification: PASSED")
    print("Registry update: NOT performed")
    print("Validation status: pending_manual_review")

if __name__ == "__main__":
    main()
