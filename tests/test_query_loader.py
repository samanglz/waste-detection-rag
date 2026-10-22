import json

import pytest

from src.evaluation.query_loader import load_evaluation_queries


def test_loads_valid_queries(tmp_path):
    queries = [
        {
            "query": "plastic recycling",
            "relevant_sources": ["plastic.txt"],
        }
    ]
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(queries), encoding="utf-8")

    result = load_evaluation_queries(path)

    assert result == queries


def test_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_evaluation_queries(tmp_path / "missing.json")


def test_rejects_invalid_json(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid evaluation query JSON"):
        load_evaluation_queries(path)


@pytest.mark.parametrize("content", [[], {}, None, "invalid"])
def test_rejects_non_list_or_empty_queries(tmp_path, content):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty list"):
        load_evaluation_queries(path)


@pytest.mark.parametrize(
    "item",
    [
        {"query": "", "relevant_sources": ["plastic.txt"]},
        {"query": "plastic", "relevant_sources": []},
        {"query": "plastic", "relevant_sources": "plastic.txt"},
        {"query": "plastic", "relevant_sources": [""]},
        {
            "query": "plastic",
            "relevant_sources": ["plastic.txt", "plastic.txt"],
        },
        "invalid",
    ],
)
def test_rejects_invalid_query_items(tmp_path, item):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps([item]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_evaluation_queries(path)
