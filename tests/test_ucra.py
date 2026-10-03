import pytest

from uncertainty_reasoning.policies.ucra import (
    UCRAPolicy,
    correlation_weights,
    decide_helper_acquisition,
)
from uncertainty_reasoning.types import Rollout
from uncertainty_reasoning.uncertainty.semantic import BidirectionalEntailmentClusterer


def rollout(answer: str, status: str = "valid") -> Rollout:
    return Rollout(answer=answer, token_count=10, metadata={"status": status})


def policy() -> UCRAPolicy:
    return UCRAPolicy(
        target_model="target",
        helper_models=("helper-1", "helper-2"),
        reliabilities={"target": 0.7, "helper-1": 0.8, "helper-2": 0.9},
        evidence_weights={"target": 1.0, "helper-1": 1.0, "helper-2": 1.0},
    )


def test_unanimous_warm_start_does_not_query_second_helper() -> None:
    result = policy().run(
        [rollout("A"), rollout("A")], rollout("A"), rollout("B")
    )
    assert result.answer == "a"
    assert not result.queried_helper_2
    assert result.warm_start_uncertainty == pytest.approx(0.0)
    assert result.warm_start_clusters == (0, 0, 0)
    assert len(result.used) == 3


def test_disagreement_queries_second_helper_and_can_reverse_target_majority() -> None:
    result = policy().run(
        [rollout("A"), rollout("A")], rollout("B"), rollout("B")
    )
    assert result.answer == "b"
    assert result.queried_helper_2
    assert result.warm_start_uncertainty == pytest.approx(0.5793801643)
    assert result.warm_start_clusters == (0, 0, 1)
    assert len(result.used) == 4


def test_invalid_warm_answer_queries_second_helper() -> None:
    result = policy().run(
        [rollout("A"), rollout("A", "truncated")], rollout("A"), rollout("B")
    )
    assert result.queried_helper_2
    assert result.warm_start_uncertainty == 1.0
    assert result.warm_start_clusters == ()


def test_three_distinct_clusters_have_maximum_uncertainty() -> None:
    decision = decide_helper_acquisition([rollout("A"), rollout("B"), rollout("C")])
    assert decision.query_helper_2
    assert decision.clustering is not None
    assert decision.clustering.normalized_entropy == pytest.approx(1.0)


def test_online_api_requires_helper_only_after_gate_requests_it() -> None:
    selected = policy()
    stop = selected.decide([rollout("A"), rollout("A")], rollout("A"))
    assert not stop.query_helper_2
    assert selected.finalize([rollout("A"), rollout("A")], rollout("A")).answer == "a"

    query = selected.decide([rollout("A"), rollout("A")], rollout("B"))
    assert query.query_helper_2
    with pytest.raises(ValueError, match="requested helper-2"):
        selected.finalize([rollout("A"), rollout("A")], rollout("B"))


def test_custom_semantic_clusterer_is_used_by_gate_and_fusion() -> None:
    equivalents = {("Paris", "The capital is Paris"), ("The capital is Paris", "Paris")}
    selected = UCRAPolicy(
        target_model="target",
        helper_models=("helper-1", "helper-2"),
        reliabilities={"target": 0.7, "helper-1": 0.8, "helper-2": 0.9},
        evidence_weights={"target": 1.0, "helper-1": 1.0, "helper-2": 1.0},
        clusterer=BidirectionalEntailmentClusterer(
            lambda left, right: left == right or (left, right) in equivalents
        ),
    )
    result = selected.finalize(
        [rollout("Paris"), rollout("The capital is Paris")], rollout("Paris")
    )
    assert not result.queried_helper_2
    assert result.answer == "Paris"


def test_correlation_weights_are_positive_and_normalized() -> None:
    weights = correlation_weights(
        ["target", "helper-1", "helper-2"],
        {
            "helper-1|target": 0.7,
            "helper-2|target": 0.2,
            "helper-1|helper-2": 0.1,
        },
    )
    assert all(value > 0 for value in weights.values())
    assert sum(weights.values()) == pytest.approx(3.0)
    assert weights["helper-2"] > weights["helper-1"]
