from typing import Any

import spacy


def _create_sentence_segmenter():
    """Create a lightweight, language-independent sentence segmenter."""
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp


_SENTENCE_SEGMENTER = _create_sentence_segmenter()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy's rule-based sentencizer."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    doc = _SENTENCE_SEGMENTER(normalized)
    return [
        sentence.text.strip()
        for sentence in doc.sents
        if sentence.text.strip()
    ]


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long_sentence(
    sentence: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split an overlong sentence into word windows with overlap."""
    words = sentence.split()
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break

    return chunks


def _overlap_sentences(
    sentences: list[str],
    overlap: int,
    next_sentence_words: int,
    chunk_size: int,
) -> list[str]:
    """Keep the largest suffix of whole sentences that fits both limits."""
    carried = []
    carried_words = 0

    for sentence in reversed(sentences):
        count = _word_count(sentence)

        if carried_words + count > overlap:
            break

        if carried_words + count + next_sentence_words > chunk_size:
            break

        carried.insert(0, sentence)
        carried_words += count

    return carried


def _build_text_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Prefer sentence boundaries and use whole-sentence overlap when possible."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush_current() -> None:
        nonlocal current, current_words

        if current:
            chunks.append(" ".join(current))

        current = []
        current_words = 0

    for sentence in sentences:
        sentence_count = _word_count(sentence)

        # Only split a sentence when it exceeds the entire chunk budget.
        if sentence_count > chunk_size:
            flush_current()
            chunks.extend(
                _split_long_sentence(sentence, chunk_size, overlap)
            )
            continue

        # The next complete sentence would exceed the chunk budget.
        if current and current_words + sentence_count > chunk_size:
            previous = current[:]
            flush_current()

            current = _overlap_sentences(
                sentences=previous,
                overlap=overlap,
                next_sentence_words=sentence_count,
                chunk_size=chunk_size,
            )
            current_words = sum(_word_count(item) for item in current)

        current.append(sentence)
        current_words += sentence_count

    flush_current()

    if any(_word_count(chunk) > chunk_size for chunk in chunks):
        raise RuntimeError("Internal error: chunk exceeds chunk_size")

    return chunks


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 120,
    overlap: int = 15,
) -> list[dict[str, Any]]:
    """Create sentence-aware chunks while preserving the existing API."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: list[dict[str, Any]] = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        if not isinstance(source, str) or not source.strip():
            raise ValueError("Document source must be a non-empty string")

        if not isinstance(text, str):
            raise ValueError(f"Document text must be a string: {source}")

        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Document metadata must be a dictionary: {source}"
            )

        text_chunks = _build_text_chunks(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk_text in enumerate(text_chunks):
            chunks.append({
                "chunk_id": f"{source}::chunk_{index}",
                "source": source,
                "text": chunk_text,
                "metadata": metadata.copy(),
            })

    return chunks
