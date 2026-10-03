"""Semantic answer clustering and empirical semantic entropy.

The reference UCRA configuration uses :class:`CanonicalAnswerClusterer`: benchmark
answers are parsed and canonicalized, and equal canonical answers form a
semantic class. This is exact for the evaluated multiple-choice labels and for
the deterministic numeric/symbolic equivalences handled by the answer adapter.

The module also exposes :class:`BidirectionalEntailmentClusterer` so the same
policy can be used with free-form answers. It requires a caller-supplied NLI
predicate and is not enabled by the default configuration.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from uncertainty_reasoning.evaluation.answers import normalize_answer


class AnswerClusterer(Protocol):
    """Assign equivalent valid answers to the same integer cluster."""

    def cluster(self, answers: Sequence[str]) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class SemanticClustering:
    """Auditable output of answer clustering and entropy calculation."""

    cluster_ids: tuple[int, ...]
    counts: tuple[int, ...]
    probabilities: tuple[float, ...]
    entropy: float
    normalized_entropy: float


@dataclass(frozen=True)
class CanonicalAnswerClusterer:
    """Cluster answers with a deterministic benchmark answer adapter.

    ``canonicalize`` may be replaced by a dataset-specific parser. The default
    implements the numeric/symbolic normalization used in the released
    reference configuration. It never inspects a gold label.
    """

    canonicalize: Callable[[str], str] = normalize_answer

    def cluster(self, answers: Sequence[str]) -> tuple[int, ...]:
        cluster_for_key: dict[str, int] = {}
        assignments: list[int] = []
        for answer in answers:
            key = self.canonicalize(answer)
            if key not in cluster_for_key:
                cluster_for_key[key] = len(cluster_for_key)
            assignments.append(cluster_for_key[key])
        return tuple(assignments)


@dataclass(frozen=True)
class BidirectionalEntailmentClusterer:
    """Reference clusterer for free-form answers using mutual entailment.

    ``entails(premise, hypothesis)`` should return a Boolean decision. Two
    answers are connected only when entailment holds in both directions. The
    connected components of that undirected graph are returned as clusters.
    This adapter is supplied for free-form extensions and requires an external
    entailment model.
    """

    entails: Callable[[str, str], bool]

    def cluster(self, answers: Sequence[str]) -> tuple[int, ...]:
        parents = list(range(len(answers)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for left in range(len(answers)):
            for right in range(left + 1, len(answers)):
                if self.entails(answers[left], answers[right]) and self.entails(
                    answers[right], answers[left]
                ):
                    union(left, right)

        cluster_for_root: dict[int, int] = {}
        assignments = []
        for index in range(len(answers)):
            root = find(index)
            if root not in cluster_for_root:
                cluster_for_root[root] = len(cluster_for_root)
            assignments.append(cluster_for_root[root])
        return tuple(assignments)


def semantic_equivalence_key(answer: str) -> str:
    """Backward-compatible canonical key used by older baseline code."""

    return normalize_answer(answer)


def cluster_entropy(cluster_ids: Sequence[int]) -> SemanticClustering:
    """Compute Shannon entropy over empirical semantic-cluster frequencies.

    UCRA normalizes by ``log(n)`` for ``n`` valid responses, placing the value
    in ``[0, 1]``. With three responses, patterns ``AAA``, ``AAB``, and ``ABC``
    yield approximately ``0``, ``0.579``, and ``1``. The frozen threshold
    ``tau=0.5`` therefore stops only for unanimity.
    """

    total = len(cluster_ids)
    if total == 0:
        return SemanticClustering((), (), (), 0.0, 1.0)
    ordered_counts = Counter(cluster_ids)
    counts = tuple(ordered_counts.values())
    probabilities = tuple(count / total for count in counts)
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized = entropy / math.log(total) if total > 1 else 0.0
    return SemanticClustering(
        cluster_ids=tuple(cluster_ids),
        counts=counts,
        probabilities=probabilities,
        entropy=entropy,
        normalized_entropy=normalized,
    )


def empirical_semantic_entropy(
    answers: Sequence[str], clusterer: AnswerClusterer | None = None
) -> SemanticClustering:
    """Cluster valid answers and return their empirical semantic entropy."""

    selected = clusterer or CanonicalAnswerClusterer()
    return cluster_entropy(selected.cluster(answers))


def normalized_empirical_answer_entropy(answers: Sequence[str | None]) -> float:
    """Compatibility wrapper; invalid evidence triggers maximum uncertainty."""

    if not answers or any(answer is None for answer in answers):
        return 1.0
    valid = [answer for answer in answers if answer is not None]
    return empirical_semantic_entropy(valid).normalized_entropy
