from __future__ import annotations

import pytest

from uncertainty_reasoning.uncertainty.semantic import (
    BidirectionalEntailmentClusterer,
    CanonicalAnswerClusterer,
    empirical_semantic_entropy,
)


def test_canonical_clusterer_groups_symbolically_equal_numbers() -> None:
    result = empirical_semantic_entropy(["1/2", "0.5", "2"])
    assert result.cluster_ids == (0, 0, 1)
    assert result.probabilities == pytest.approx((2 / 3, 1 / 3))
    assert result.normalized_entropy == pytest.approx(0.5793801643)


def test_canonical_clusterer_does_not_use_gold_answer() -> None:
    clusterer = CanonicalAnswerClusterer()
    assert clusterer.cluster(["A", "a", "B"]) == (0, 0, 1)


def test_bidirectional_entailment_requires_both_directions() -> None:
    relations = {
        ("Paris", "The capital is Paris"),
        ("The capital is Paris", "Paris"),
        ("France", "The capital is Paris"),
    }
    clusterer = BidirectionalEntailmentClusterer(lambda left, right: (left, right) in relations)
    assert clusterer.cluster(["Paris", "The capital is Paris", "France"]) == (0, 0, 1)
