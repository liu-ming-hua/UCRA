"""Online acquisition and source-aware fusion for UCRA."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from uncertainty_reasoning.types import Rollout
from uncertainty_reasoning.uncertainty.semantic import (
    AnswerClusterer,
    CanonicalAnswerClusterer,
    SemanticClustering,
    empirical_semantic_entropy,
)


def valid_answer(rollout: Rollout) -> str | None:
    """Return a valid answer without consulting a gold label."""

    if not rollout.answer or rollout.metadata.get("status", "valid") != "valid":
        return None
    return rollout.answer.strip()


@dataclass(frozen=True)
class UCRAResult:
    answer: str | None
    posterior: float
    used: tuple[tuple[str, Rollout], ...]
    queried_helper_2: bool
    warm_start_uncertainty: float
    warm_start_clusters: tuple[int, ...]


@dataclass(frozen=True)
class AcquisitionDecision:
    """The observable UCRA gate decision before answer fusion."""

    query_helper_2: bool
    clustering: SemanticClustering | None
    reason: str


def decide_helper_acquisition(
    warm_start: list[Rollout],
    *,
    threshold: float = 0.5,
    clusterer: AnswerClusterer | None = None,
) -> AcquisitionDecision:
    """Apply UCRA's semantic-entropy gate to three warm-start responses.

    A response marked invalid or truncated is treated as maximally uncertain.
    Otherwise answers are clustered *without gold labels*, and helper-2 is
    queried exactly when normalized empirical semantic entropy exceeds
    ``threshold``. The reference protocol uses three responses and
    ``threshold=0.5``.
    """

    if len(warm_start) != 3:
        raise ValueError("UCRA requires exactly three warm-start responses")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    answers = [valid_answer(rollout) for rollout in warm_start]
    if any(answer is None for answer in answers):
        return AcquisitionDecision(True, None, "invalid warm-start response")
    valid = [answer for answer in answers if answer is not None]
    clustering = empirical_semantic_entropy(
        valid, clusterer or CanonicalAnswerClusterer()
    )
    return AcquisitionDecision(
        query_helper_2=clustering.normalized_entropy > threshold,
        clustering=clustering,
        reason=(
            "semantic entropy above threshold"
            if clustering.normalized_entropy > threshold
            else "semantic entropy at or below threshold"
        ),
    )


def reliability_source_posterior(
    target: list[Rollout],
    helpers: list[tuple[str, Rollout]],
    *,
    target_model: str,
    reliabilities: Mapping[str, float],
    evidence_weights: Mapping[str, float] | None = None,
    clusterer: AnswerClusterer | None = None,
) -> tuple[str | None, float]:
    """Return the MAP answer and normalized composite-likelihood mass."""

    groups: list[tuple[str, list[Rollout]]] = [(target_model, target)]
    groups.extend((model, [rollout]) for model, rollout in helpers)
    selected_clusterer = clusterer or CanonicalAnswerClusterer()
    raw_answers: list[str] = []
    group_answer_indexes: list[tuple[str, list[int]]] = []
    for model, rollouts in groups:
        indexes: list[int] = []
        for rollout in rollouts:
            answer = valid_answer(rollout)
            if answer is None:
                continue
            indexes.append(len(raw_answers))
            raw_answers.append(answer)
        if indexes:
            group_answer_indexes.append((model, indexes))

    if not raw_answers:
        return None, 0.0
    assignments = selected_clusterer.cluster(raw_answers)
    order = list(dict.fromkeys(assignments))
    representatives: dict[int, str] = {}
    for answer, cluster_id in zip(raw_answers, assignments, strict=True):
        if cluster_id not in representatives:
            representatives[cluster_id] = (
                selected_clusterer.canonicalize(answer)
                if isinstance(selected_clusterer, CanonicalAnswerClusterer)
                else answer
            )

    distributions: list[tuple[str, dict[int, float]]] = []
    for model, indexes in group_answer_indexes:
        counts: defaultdict[int, int] = defaultdict(int)
        for index in indexes:
            counts[assignments[index]] += 1
        total = sum(counts.values())
        if total:
            distributions.append(
                (model, {answer: count / total for answer, count in counts.items()})
            )
    if len(order) <= 1:
        return representatives[order[0]], 1.0

    candidate_count = len(order)
    scores = {answer: 0.0 for answer in order}
    for model, distribution in distributions:
        reliability = min(0.99, max(0.01, reliabilities[model]))
        evidence_weight = 1.0 if evidence_weights is None else evidence_weights[model]
        log_correct = math.log(reliability)
        log_wrong = math.log((1.0 - reliability) / (candidate_count - 1))
        for candidate in order:
            scores[candidate] += evidence_weight * sum(
                probability * (log_correct if answer == candidate else log_wrong)
                for answer, probability in distribution.items()
            )
    winner = max(order, key=scores.__getitem__)
    maximum = scores[winner]
    normalizer = sum(math.exp(score - maximum) for score in scores.values())
    return representatives[winner], 1.0 / normalizer


def solve_linear(matrix: list[list[float]], values: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    size = len(values)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        if abs(scale) < 1e-10:
            return [1.0] * size
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * reference
                for current, reference in zip(augmented[row], augmented[column], strict=True)
            ]
    return [row[-1] for row in augmented]


def correlation_weights(models: list[str], correlations: Mapping[str, float]) -> dict[str, float]:
    """Non-negative generalized-least-squares weights from a correlation matrix."""

    matrix = []
    for left_index, left_model in enumerate(models):
        row = []
        for right_index, right_model in enumerate(models):
            if left_index == right_index:
                row.append(1.05)
            else:
                row.append(correlations["|".join(sorted((left_model, right_model)))])
        matrix.append(row)
    raw = [max(0.05, value) for value in solve_linear(matrix, [1.0] * len(models))]
    scale = len(raw) / sum(raw)
    return {model: value * scale for model, value in zip(models, raw, strict=True)}


class UCRAPolicy:
    """Reusable online UCRA with frozen acquisition and fusion parameters."""

    def __init__(
        self,
        *,
        target_model: str,
        helper_models: tuple[str, str],
        reliabilities: Mapping[str, float],
        evidence_weights: Mapping[str, float],
        entropy_threshold: float = 0.5,
        clusterer: AnswerClusterer | None = None,
    ) -> None:
        self.target_model = target_model
        self.helper_models = helper_models
        self.reliabilities = reliabilities
        self.evidence_weights = evidence_weights
        self.entropy_threshold = entropy_threshold
        self.clusterer = clusterer or CanonicalAnswerClusterer()

    def decide(self, target: list[Rollout], helper_1: Rollout) -> AcquisitionDecision:
        """Decide whether helper-2 must be queried; this performs no query."""

        if len(target) < 2:
            raise ValueError("UCRA requires at least two target rollouts")
        return decide_helper_acquisition(
            [target[0], target[1], helper_1],
            threshold=self.entropy_threshold,
            clusterer=self.clusterer,
        )

    def finalize(
        self,
        target: list[Rollout],
        helper_1: Rollout,
        helper_2: Rollout | None = None,
    ) -> UCRAResult:
        """Fuse acquired evidence after applying the online acquisition decision.

        Call :meth:`decide` first in an online system. If it requests helper-2,
        acquire that response and pass it here. Cached replay may call
        :meth:`run_cached` instead.
        """

        decision = self.decide(target, helper_1)
        if decision.query_helper_2 and helper_2 is None:
            raise ValueError("the entropy gate requested helper-2, but no response was supplied")
        used = [("target", target[0]), ("target", target[1]), ("helper_1", helper_1)]
        if decision.query_helper_2:
            assert helper_2 is not None
            used.append(("helper_2", helper_2))
        selected_target = [rollout for source, rollout in used if source == "target"]
        selected_helpers = [
            (
                self.helper_models[0] if source == "helper_1" else self.helper_models[1],
                rollout,
            )
            for source, rollout in used
            if source != "target"
        ]
        answer, posterior = reliability_source_posterior(
            selected_target,
            selected_helpers,
            target_model=self.target_model,
            reliabilities=self.reliabilities,
            evidence_weights=self.evidence_weights,
            clusterer=self.clusterer,
        )
        return UCRAResult(
            answer=answer,
            posterior=posterior,
            used=tuple(used),
            queried_helper_2=decision.query_helper_2,
            warm_start_uncertainty=(
                decision.clustering.normalized_entropy if decision.clustering else 1.0
            ),
            warm_start_clusters=(
                decision.clustering.cluster_ids if decision.clustering else ()
            ),
        )

    def run_cached(
        self, target: list[Rollout], helper_1: Rollout, helper_2: Rollout
    ) -> UCRAResult:
        """Replay UCRA from an existing pool that already contains helper-2."""

        return self.finalize(target, helper_1, helper_2)

    def run(
        self, target: list[Rollout], helper_1: Rollout, helper_2: Rollout | None = None
    ) -> UCRAResult:
        """Convenience alias for :meth:`finalize`."""

        return self.finalize(target, helper_1, helper_2)
