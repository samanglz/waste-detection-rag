import pytest

from src.chunking.text_chunker import chunk_documents


def test_short_document_creates_one_chunk():
    documents = [{"source": "plastic.txt", "text": "PET can be recycled."}]
    chunks = chunk_documents(documents, chunk_size=10, overlap=2)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "plastic.txt"
    assert chunks[0]["text"] == "PET can be recycled."
    assert chunks[0]["metadata"] == {}


def test_chunks_overlap_for_plain_word_sequence():
    words = [f"word{i}" for i in range(12)]
    documents = [{"source": "long.txt", "text": " ".join(words)}]

    chunks = chunk_documents(documents, chunk_size=5, overlap=2)

    assert len(chunks) == 4
    assert chunks[0]["text"].split()[-2:] == ["word3", "word4"]
    assert chunks[1]["text"].split()[:2] == ["word3", "word4"]


def test_metadata_is_preserved_in_each_chunk():
    metadata = {
        "title": "Example document",
        "source_url": "https://example.org/document",
        "jurisdiction": "general",
    }
    documents = [{
        "source": "example.txt",
        "text": " ".join(f"word{i}" for i in range(12)),
        "metadata": metadata,
    }]

    chunks = chunk_documents(documents, chunk_size=5, overlap=2)

    assert len(chunks) == 4
    assert all(chunk["metadata"] == metadata for chunk in chunks)


def test_chunk_metadata_is_not_shared_mutably_with_document():
    metadata = {"topic": "plastic"}
    documents = [{
        "source": "plastic.txt",
        "text": "PET HDPE recycling",
        "metadata": metadata,
    }]

    chunks = chunk_documents(documents, chunk_size=2, overlap=0)
    chunks[0]["metadata"]["topic"] = "changed"

    assert metadata["topic"] == "plastic"
    assert chunks[1]["metadata"]["topic"] == "plastic"


def test_chunk_ids_are_unique_for_one_document():
    documents = [{
        "source": "long.txt",
        "text": "one two three four five six seven eight",
    }]

    chunks = chunk_documents(documents, chunk_size=3, overlap=1)

    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [
        "long.txt::chunk_0",
        "long.txt::chunk_1",
        "long.txt::chunk_2",
        "long.txt::chunk_3",
    ]


def test_empty_document_text_creates_no_chunks():
    assert chunk_documents(
        [{"source": "empty.txt", "text": "   "}]
    ) == []


def test_rejects_invalid_source():
    with pytest.raises(ValueError, match="source"):
        chunk_documents([{"source": " ", "text": "some text"}])


def test_rejects_non_string_text():
    with pytest.raises(ValueError, match="text must be a string"):
        chunk_documents([{"source": "bad.txt", "text": None}])


def test_rejects_invalid_metadata():
    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        chunk_documents([{
            "source": "bad.txt",
            "text": "some text",
            "metadata": ["invalid"],
        }])


def test_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_documents([], chunk_size=0, overlap=0)


def test_invalid_overlap():
    with pytest.raises(ValueError, match="overlap must be >= 0"):
        chunk_documents([], chunk_size=5, overlap=5)


def test_sentence_boundaries_are_preserved_when_possible():
    documents = [{
        "source": "sentences.txt",
        "text": (
            "First sentence has five words here. "
            "Second sentence also has five words. "
            "Third sentence ends the passage."
        ),
    }]

    chunks = chunk_documents(documents, chunk_size=11, overlap=0)

    assert len(chunks) == 2
    assert chunks[0]["text"].endswith("here.")
    assert chunks[1]["text"].startswith("Second sentence")


def test_no_chunk_exceeds_hard_word_limit():
    documents = [{
        "source": "long-sentence.txt",
        "text": " ".join(f"token{i}" for i in range(25)),
    }]

    chunks = chunk_documents(documents, chunk_size=8, overlap=2)

    assert chunks
    assert all(len(chunk["text"].split()) <= 8 for chunk in chunks)


def test_long_sentence_content_is_preserved_in_order():
    words = [f"token{i}" for i in range(25)]
    chunks = chunk_documents(
        [{"source": "long.txt", "text": " ".join(words)}],
        chunk_size=8,
        overlap=2,
    )

    # Remove duplicated overlap by checking that every original token
    # appears in the emitted sequence in the same order.
    emitted = []
    for chunk in chunks:
        for word in chunk["text"].split():
            if not emitted or word != emitted[-1]:
                emitted.append(word)

    positions = [emitted.index(word) for word in words]
    assert positions == sorted(positions)
    assert all(word in emitted for word in words)


def test_metadata_defaults_to_empty_dictionary():
    chunks = chunk_documents([{
        "source": "plain.txt",
        "text": "Plain text.",
    }])

    assert chunks[0]["metadata"] == {}


def test_persian_sentence_boundaries_are_preserved():
    documents = [{
        "source": "persian.txt",
        "text": (
            "بازیافت کاغذ مهم است. "
            "آیا همه کاغذها بازیافت می‌شوند؟ "
            "خیر، بعضی از آن‌ها پذیرفته نمی‌شوند!"
        ),
    }]

    chunks = chunk_documents(
        documents,
        chunk_size=30,
        overlap=0,
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == documents[0]["text"]


def test_mixed_persian_and_english_sentence_segmentation():
    documents = [{
        "source": "mixed.txt",
        "text": (
            "بازیافت کاغذ مهم است. "
            "Paper recycling reduces waste. "
            "آیا همه مواد قابل بازیافت هستند؟"
        ),
    }]

    chunks = chunk_documents(
        documents,
        chunk_size=8,
        overlap=0,
    )

    assert chunks
    assert all(
        len(chunk["text"].split()) <= 8
        for chunk in chunks
    )

    combined_text = " ".join(chunk["text"] for chunk in chunks)
    assert "بازیافت کاغذ مهم است." in combined_text
    assert "Paper recycling reduces waste." in combined_text
    assert "آیا همه مواد قابل بازیافت هستند؟" in combined_text
