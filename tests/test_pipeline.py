import numpy as np
import pytest

from src.indexing.index_store import IndexStore
from src.pipeline import build_index_pipeline


class FakeEncoder:
    """Deterministic encoder for pipeline tests."""

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [[float(i + 1), 1.0] for i in range(len(texts))],
            dtype=np.float32,
        )


def test_build_index_pipeline(tmp_path):
    data_dir = tmp_path / "raw"
    index_dir = tmp_path / "index"
    data_dir.mkdir()

    (data_dir / "plastic.txt").write_text(
        "Plastic can be sorted and recycled into new products.",
        encoding="utf-8",
    )
    (data_dir / "paper.txt").write_text(
        "Paper recycling saves resources and reduces waste.",
        encoding="utf-8",
    )

    summary = build_index_pipeline(
        data_dir=data_dir,
        index_dir=index_dir,
        encoder=FakeEncoder(),
        chunk_size=5,
        overlap=1,
    )

    assert summary["documents"] == 2
    assert summary["chunks"] > 0
    assert summary["vectors"] == summary["chunks"]

    index, chunks = IndexStore(index_dir).load()

    assert index.size == summary["vectors"]
    assert len(chunks) == summary["chunks"]
    assert {chunk["source"] for chunk in chunks} == {
        "plastic.txt",
        "paper.txt",
    }


def test_pipeline_rejects_empty_directory(tmp_path):
    data_dir = tmp_path / "empty"
    data_dir.mkdir()

    with pytest.raises(ValueError, match="No non-empty text documents"):
        build_index_pipeline(
            data_dir=data_dir,
            index_dir=tmp_path / "index",
            encoder=FakeEncoder(),
        )
