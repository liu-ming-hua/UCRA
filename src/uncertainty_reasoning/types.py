from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Rollout:
    answer: str
    text: str = ""
    token_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    stop: bool
    allocations: tuple[int, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReasoningState:
    observed: tuple[Rollout, ...]
    spent_samples: int
    max_samples: int

    @property
    def remaining(self) -> int:
        return max(0, self.max_samples - self.spent_samples)
