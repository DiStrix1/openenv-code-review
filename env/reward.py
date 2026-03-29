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


def _llm_alignment_signal(history: List[Dict[str, Any]]) -> float:
    llm_scores: List[float] = []
    for step in history:
        value = step.get("llm_score")
        if value is None:
            continue
        llm_scores.append(float(value))
    if not llm_scores:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    total = len(llm_scores)
    for idx, score in enumerate(llm_scores, start=1):
        weight = idx / float(total)
        weighted_sum += weight * (score - 0.5)
        weight_total += weight
    if weight_total == 0.0:
        return 0.0
    return weighted_sum / weight_total


def compute_score(
    history: List[Dict[str, Any]],
    remaining_issues: int,
    total_issues: int,
    step_count: int,
    done: bool,
    max_steps: int,
) -> float:
    reward = 0.0
    reward += 1.2 * float(_correct_fixes(history))
    reward -= 1.8 * float(_false_positives(history))
    reward -= 0.8 * float(_unnecessary_actions(history))

    # Step efficiency.
    reward -= 0.35 * float(step_count)

    # LLM alignment signal contributes directly to reward.
    reward += 1.5 * float(_llm_alignment_signal(history))

    total = max(int(total_issues), 1)
    remaining_fraction = float(remaining_issues) / float(total)
    completion_fraction = 1.0 - remaining_fraction
    reward += 3.0 * (completion_fraction ** 2)

    # Completion and early finish bonuses.
    if done and remaining_issues == 0:
        reward += 2.0
    if done and step_count < max_steps:
        reward += (max_steps - step_count) * 0.3

    return float(_clamp(reward, -5.0, 10.0))
