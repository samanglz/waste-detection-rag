from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.source_registry import load_registry, save_registry, upsert_source

REGISTRY_PATH = PROJECT_ROOT / "metadata" / "source_registry.json"

SOURCES = [
    {
        "document_id": "epa_recycling_basics",
        "title": "Recycling Basics and Benefits",
        "publisher": "U.S. Environmental Protection Agency",
        "source_type": "official_guidance",
        "url": "https://www.epa.gov/recycle/recycling-basics-and-benefits",
        "language": "en",
        "topics": ["recycling", "waste_management", "environment"],
        "status": "candidate",
        "license_status": "verify_before_full_text_storage",
    },
    {
        "document_id": "epa_common_recyclables",
        "title": "How Do I Recycle Common Recyclables?",
        "publisher": "U.S. Environmental Protection Agency",
        "source_type": "official_guidance",
        "url": "https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables",
        "language": "en",
        "topics": ["paper", "cardboard", "glass", "metal", "plastic"],
        "status": "candidate",
        "license_status": "verify_before_full_text_storage",
    },
    {
        "document_id": "epa_plastic_best_practices",
        "title": "Best Practices",
        "publisher": "U.S. Environmental Protection Agency",
        "source_type": "official_guidance",
        "url": "https://www.epa.gov/plastics/best-practices",
        "language": "en",
        "topics": ["plastic", "collection", "sorting", "waste_management"],
        "status": "candidate",
        "license_status": "verify_before_full_text_storage",
    },
    {
        "document_id": "eea_recycling_quality",
        "title": "Measuring the Quality of Recycling",
        "publisher": "European Environment Agency",
        "source_type": "official_report",
        "url": "https://www.eea.europa.eu/en/analysis/publications/measuring-the-quality-of-recycling",
        "language": "en",
        "topics": ["recycling_quality", "sorting", "plastic", "evaluation"],
        "status": "candidate",
        "license_status": "verify_before_full_text_storage",
    },
    {
        "document_id": "eea_separate_collection",
        "title": "Economic Instruments and Separate Collection Systems",
        "publisher": "European Environment Agency",
        "source_type": "official_report",
        "url": "https://www.eea.europa.eu/en/analysis/publications/economic-instruments-and-separate-collection",
        "language": "en",
        "topics": ["collection", "paper", "glass", "metal", "plastic"],
        "status": "candidate",
        "license_status": "verify_before_full_text_storage",
    },
    {
        "document_id": "eea_circular_plastics",
        "title": "Pathways towards Circular Plastics in Europe",
        "publisher": "European Environment Agency",
        "source_type": "official_report",
        "url": "https://www.eea.europa.eu/en/analysis/publications/pathways-towards-circular-plastics",
        "language": "en",
        "topics": ["plastic", "circular_economy", "sorting", "recycling"],
        "status": "candidate",
        "license_status": "verify_before_full_text_storage",
    },
    {
        "document_id": "oecd_global_plastics_outlook",
        "title": "Global Plastics Outlook",
        "publisher": "OECD",
        "source_type": "official_report",
        "url": "https://www.oecd.org/en/publications/global-plastics-outlook_de747aef-en.html",
        "language": "en",
        "topics": ["plastic", "waste_management", "recycling", "policy"],
        "status": "candidate",
        "license_status": "verify_before_full_text_storage",
    },
    {
        "document_id": "oecd_recycled_plastics_markets",
        "title": "Improving Markets for Recycled Plastics",
        "publisher": "OECD",
        "source_type": "official_report",
        "url": "https://www.oecd.org/en/publications/improving-markets-for-recycled-plastics_9789264301016-en.html",
        "language": "en",
        "topics": ["plastic", "recycling", "circular_economy", "markets"],
        "status": "candidate",
        "license_status": "verify_before_full_text_storage",
    },
]


def main() -> None:
    registry = load_registry(REGISTRY_PATH)
    before = len(registry["sources"])

    for source in SOURCES:
        upsert_source(registry, source)

    save_registry(REGISTRY_PATH, registry)

    print(f"Registry: {REGISTRY_PATH}")
    print(f"Sources before: {before}")
    print(f"Sources after: {len(registry['sources'])}")
    print(f"Candidates processed: {len(SOURCES)}")


if __name__ == "__main__":
    main()
