"""Run UCRA on two tiny cached examples without a GPU or model download."""

from __future__ import annotations

import json
from pathlib import Path

from uncertainty_reasoning.policies.ucra import UCRAPolicy
from uncertainty_reasoning.types import Rollout

ROOT = Path(__file__).resolve().parents[1]


def rollout(answer: str) -> Rollout:
    return Rollout(answer=answer, token_count=10, metadata={"status": "valid"})


def main() -> None:
    parameters = json.loads(
        (ROOT / "configs/ucra_example_parameters.json").read_text(encoding="utf-8")
    )
    target_model = "qwen3-8b"
    helpers = tuple(parameters["helpers"][target_model])
    policy = UCRAPolicy(
        target_model=target_model,
        helper_models=helpers,
        reliabilities=parameters["reliabilities"],
        evidence_weights=parameters["evidence_weights"][target_model],
        entropy_threshold=parameters["entropy_threshold"],
    )
    examples = json.loads(
        (ROOT / "examples/data/toy_multi_model_pool.json").read_text(encoding="utf-8")
    )
    for example in examples:
        result = policy.run_cached(
            [rollout(answer) for answer in example["target"]],
            rollout(example["helper_1"]),
            rollout(example["helper_2"]),
        )
        print(
            f"{example['id']}: answer={result.answer} "
            f"entropy={result.warm_start_uncertainty:.3f} "
            f"clusters={result.warm_start_clusters} "
            f"helper_2={result.queried_helper_2} calls={len(result.used)}"
        )


if __name__ == "__main__":
    main()
