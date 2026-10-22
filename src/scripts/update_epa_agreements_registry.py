import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "metadata/source_registry.json"
SOURCE = ROOT / "data/staging_v4_expanded/epa_international_waste_agreements.complete_v4.txt"
BACKUP = ROOT / "metadata/source_registry.json.before_epa_agreements_v4.bak"
TEMP = REGISTRY.with_suffix(".json.tmp")

DOC_ID = "epa_international_waste_agreements"
NEW_PATH = "data/staging_v4_expanded/epa_international_waste_agreements.complete_v4.txt"
EXPECTED_LICENSE = "verify_before_full_text_storage"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def find_records(obj):
    found = []
    if isinstance(obj, dict):
        if obj.get("document_id") == DOC_ID:
            found.append(obj)
        for value in obj.values():
            found.extend(find_records(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(find_records(value))
    return found

def main():
    if not REGISTRY.is_file():
        raise FileNotFoundError(REGISTRY)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if BACKUP.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing backup: {BACKUP}"
        )
    if TEMP.exists():
        raise FileExistsError(
            f"Temporary file already exists; inspect manually: {TEMP}"
        )

    # Validate that the completed file contains the three expected links.
    text = SOURCE.read_text(encoding="utf-8")
    required_urls = [
        "https://www.epa.gov/hwgenerators/agreement-between-government-america-and-government-malaysia-concerning-transboundary",
        "https://www.epa.gov/hwgenerators/agreement-transboundary-movement-hazardous-waste-costa-rica-united-states-1997",
        "https://www.epa.gov/hwgenerators/agreement-between-government-united-states-america-and-government-republic-philippines",
    ]
    for url in required_urls:
        if text.count(url) != 1:
            raise ValueError(f"Expected URL exactly once: {url}")

    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    records = find_records(data)
    if len(records) != 1:
        raise ValueError(
            f"Expected exactly one registry record for {DOC_ID}; found {len(records)}"
        )

    record = records[0]
    old_status = record.get("status")
    old_license = record.get("license_status")

    if old_license != EXPECTED_LICENSE:
        raise ValueError(
            "Unexpected license_status; refusing to change registry: "
            f"{old_license!r}"
        )

    # Preserve the registry before changing anything.
    shutil.copy2(REGISTRY, BACKUP)

    record["local_path"] = NEW_PATH
    record["sha256"] = sha256(SOURCE)

    try:
        TEMP.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # Verify temporary JSON and the exact target record before replacement.
        verified = json.loads(TEMP.read_text(encoding="utf-8"))
        verified_records = find_records(verified)
        if len(verified_records) != 1:
            raise ValueError("Temporary registry verification failed.")
        updated = verified_records[0]
        if updated.get("local_path") != NEW_PATH:
            raise ValueError("Temporary local_path verification failed.")
        if updated.get("sha256") != sha256(SOURCE):
            raise ValueError("Temporary SHA-256 verification failed.")
        if updated.get("status") != old_status:
            raise ValueError("Source status unexpectedly changed.")
        if updated.get("license_status") != old_license:
            raise ValueError("License status unexpectedly changed.")

        os.replace(TEMP, REGISTRY)
    except Exception:
        if TEMP.exists():
            TEMP.unlink()
        raise

    final_data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    final_records = find_records(final_data)
    if len(final_records) != 1:
        raise RuntimeError("Final registry record count is invalid.")

    final_record = final_records[0]
    if final_record.get("local_path") != NEW_PATH:
        raise RuntimeError("Final local_path verification failed.")
    if final_record.get("sha256") != sha256(SOURCE):
        raise RuntimeError("Final hash verification failed.")
    if final_record.get("status") != old_status:
        raise RuntimeError("Final source status changed unexpectedly.")
    if final_record.get("license_status") != old_license:
        raise RuntimeError("Final license status changed unexpectedly.")

    print("Registry updated and verified.")
    print("Backup:", BACKUP)
    print("Document ID:", DOC_ID)
    print("local_path:", final_record["local_path"])
    print("sha256:", final_record["sha256"])
    print("status preserved:", final_record.get("status"))
    print("license_status preserved:", final_record.get("license_status"))
    print("Full-text license approval: NOT granted")
    print("Index build: NOT performed")

if __name__ == "__main__":
    main()
