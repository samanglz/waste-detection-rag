
import pytest

from src.retrieval.reciprocal_rank_fusion import (
    reciprocal_rank_fusion,
)


def test_fuses_two_rankings():
    rankings = [
        [0, 1, 2],
        [1, 0, 3],
    ]

    ranking, scores = reciprocal_rank_fusion(rankings)

    assert ranking == [0, 1, 2, 3]
    assert scores[0] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 63)
    assert scores[3] == pytest.approx(1 / 63)


def test_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[0, 1]], k=0)


def test_empty_rankings():
    ranking, scores = reciprocal_rank_fusion([])

    assert ranking == []
    assert scores == {}
