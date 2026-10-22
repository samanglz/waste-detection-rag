import json
import ssl
import certifi
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.source_registry import load_registry

REGISTRY = PROJECT_ROOT / "metadata" / "source_registry.json"
REPORT = PROJECT_ROOT / "metadata" / "url_check_report.json"


def check_url(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": "WasteDetectionRAG-SourceChecker/1.0",
            "Range": "bytes=0-2047",
        },
    )
    try:
        with urlopen(
            request,
            timeout=20,
            context=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
            return {
                "http_status": response.status,
                "final_url": response.geturl(),
                "content_type": response.headers.get("Content-Type"),
                "result": (
                    "reachable"
                    if 200 <= response.status < 400
                    else "check_required"
                ),
            }
    except HTTPError as exc:
        return {
            "http_status": exc.code,
            "final_url": exc.geturl(),
            "result": "check_required",
            "error": str(exc),
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "result": "connection_error",
            "error": str(exc),
        }


def main() -> None:
    registry = load_registry(REGISTRY)
    results = []

    for source in registry["sources"]:
        result = {
            "document_id": source["document_id"],
            "title": source["title"],
            "url": source["url"],
        }
        result.update(check_url(source["url"]))
        results.append(result)

        print(
            f'{result["document_id"]}: '
            f'{result["result"]} '
            f'(HTTP {result.get("http_status", "N/A")})'
        )

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(REGISTRY.relative_to(PROJECT_ROOT)),
        "results": results,
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved: {REPORT}")
    print(f"Total checked: {len(results)}")
    print(f"Reachable: {sum(r['result'] == 'reachable' for r in results)}")


if __name__ == "__main__":
    main()
