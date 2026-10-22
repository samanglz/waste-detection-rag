from typing import Any


def build_context(results: list[dict[str, Any]], max_chars: int = 6000) -> dict[str, Any]:
    """Build a bounded context and register only sources represented in it."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    context_parts = []
    sources = []
    source_ids = {}
    used_chars = 0

    for result in results:
        text = result.get("text", "").strip()
        source = result.get("source", "").strip()

        if not text or not source:
            continue

        is_new_source = source not in source_ids
        source_id = (
            f"S{len(source_ids) + 1}"
            if is_new_source
            else source_ids[source]
        )

        block = (
            f"[{source_id}] Source: {source}\\n"
            f"Chunk ID: {result.get('chunk_id', 'unknown')}\\n"
            f"Content: {text}"
        )

        separator = "\\n\\n" if context_parts else ""
        remaining = max_chars - used_chars - len(separator)

        if remaining <= 0:
            break

        if len(block) > remaining:
            prefix = (
                f"[{source_id}] Source: {source}\\n"
                f"Chunk ID: {result.get('chunk_id', 'unknown')}\\n"
                "Content: "
            )

            if remaining <= len(prefix):
                break

            block = prefix + text[: remaining - len(prefix)]

        # Register the source only after confirming that its block will be included.
        if is_new_source:
            source_ids[source] = source_id
            sources.append({"id": source_id, "source": source})

        context_parts.append(block)
        used_chars += len(separator) + len(block)

        if used_chars >= max_chars:
            break

    return {
        "context": "\\n\\n".join(context_parts),
        "sources": sources,
    }
