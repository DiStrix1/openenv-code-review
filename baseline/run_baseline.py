import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.environment import CodeReviewEnv
from env.grader import grade


def _has_unresolved(issues: List[Dict[str, Any]]) -> bool:
    return any(not issue.get("resolved", False) for issue in issues)


def _unresolved(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [issue for issue in issues if not issue.get("resolved", False)]


def smart_agent(state: Dict[str, Any]) -> str:
    issues = state.get("issues", [])
    history = state.get("history", [])
    unresolved = _unresolved(issues)
    unresolved_ground_truth = [issue for issue in unresolved if issue.get("source") == "ground_truth"]
    unresolved_introduced = [issue for issue in unresolved if issue.get("source") == "introduced_bug"]

    if not _has_unresolved(issues):
        return "ignore"

    last_action = history[-1]["action"] if history else None
    flagged_unresolved = [issue for issue in unresolved if issue.get("flagged", False)]
    flagged_ground_truth = [issue for issue in flagged_unresolved if issue.get("source") == "ground_truth"]
    high_ground_truth = [
        issue
        for issue in unresolved_ground_truth
        if issue.get("severity") == "high"
    ]
    perf_ground_truth = [
        issue
        for issue in unresolved_ground_truth
        if issue.get("type") == "performance"
    ]
    high_introduced = [
        issue
        for issue in unresolved_introduced
        if issue.get("severity") == "high"
    ]

    if flagged_ground_truth:
        candidate = "suggest_fix"
    elif high_ground_truth:
        candidate = "flag_issue"
    elif perf_ground_truth:
        candidate = "optimize_code"
    elif unresolved_ground_truth:
        candidate = "flag_issue"
    elif flagged_unresolved:
        candidate = "suggest_fix"
    elif high_introduced:
        candidate = "flag_issue"
    else:
        candidate = "suggest_fix"

    if candidate != last_action:
        return candidate

    fallback_priority = ["suggest_fix", "flag_issue", "optimize_code", "ignore"]
    for action in fallback_priority:
        if action == last_action:
            continue
        if action == "ignore" and _has_unresolved(issues):
            continue
        if action == "optimize_code" and not perf_ground_truth:
            continue
        return action

    return "ignore"


def run_episode(task_name: str, max_steps: int = 12) -> Dict[str, Any]:
    env = CodeReviewEnv(task=task_name, max_steps=max_steps)
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = smart_agent(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    final_score = grade(state)
    unresolved_ground_truth = [
        issue
        for issue in state["issues"]
        if issue.get("source") == "ground_truth" and not issue.get("resolved", False)
    ]
    unresolved_total = [issue for issue in env.issues if not issue.get("resolved", False)]
    return {
        "task": task_name,
        "steps": state["step_count"],
        "total_reward": float(total_reward),
        "final_grade": final_score,
        "remaining_issues": len(unresolved_ground_truth),
        "remaining_total_issues": len(unresolved_total),
    }


def main() -> None:
    for task_name in ("easy", "medium", "hard"):
        result = run_episode(task_name)
        print(
            f"task={result['task']} "
            f"steps={result['steps']} "
            f"reward={result['total_reward']:.2f} "
            f"grade={result['final_grade']:.3f} "
            f"remaining_gt={result['remaining_issues']} "
            f"remaining_total={result['remaining_total_issues']}"
        )


if __name__ == "__main__":
    main()
