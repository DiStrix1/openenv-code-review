from typing import Any, Dict, List


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _correct_fixes(history: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for step in history
        if step.get("fix_applied", False) and step.get("fixed_issue_source") == "ground_truth"
    )


def _false_positives(history: List[Dict[str, Any]]) -> int:
    return sum(1 for step in history if step.get("false_positive", False))


def _unnecessary_actions(history: List[Dict[str, Any]]) -> int:
    return sum(1 for step in history if step.get("unnecessary", False))


def compute_score(
    history: List[Dict[str, Any]],
    remaining_issues: int,
    step_count: int,
    done: bool,
    max_steps: int,
) -> float:
    reward = 0.0
    reward += 2.0 * float(_correct_fixes(history))
    reward -= 2.0 * float(_false_positives(history))
    reward -= 1.0 * float(_unnecessary_actions(history))

    # Step efficiency: stronger visible penalty.
    reward -= 0.5 * float(step_count)

    # Completion and early finish bonuses.
    if done and remaining_issues == 0:
        reward += 3.0
    if done and step_count < max_steps:
        reward += (max_steps - step_count) * 0.5

    return float(_clamp(reward, -5.0, 10.0))

