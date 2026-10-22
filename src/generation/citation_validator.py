import re
from typing import Any


CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def validate_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate citation IDs against the source registry.

    This checks ID existence, not whether sources support factual claims.
    """
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    if not isinstance(sources, list):
        raise TypeError("sources must be a list")

    valid_ids = list(dict.fromkeys(
        str(source.get("id", "")).strip()
        for source in sources
        if isinstance(source, dict) and str(source.get("id", "")).strip()
    ))
    valid_id_set = set(valid_ids)

    cited_ids = list(dict.fromkeys(CITATION_PATTERN.findall(answer)))
    invalid_ids = [
        citation for citation in cited_ids
        if citation not in valid_id_set
    ]
    unused_ids = [
        source_id for source_id in valid_ids
        if source_id not in cited_ids
    ]

    return {
        "is_valid": bool(cited_ids) and not invalid_ids,
        "has_citations": bool(cited_ids),
        "all_citations_exist": not invalid_ids,
        "cited_ids": cited_ids,
        "invalid_ids": invalid_ids,
        "unused_source_ids": unused_ids,
    }
