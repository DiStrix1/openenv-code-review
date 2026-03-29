import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.environment import CodeReviewEnv
from env.grader import grade


_SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 3}


def _unresolved(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [issue for issue in issues if not issue.get("resolved", False)]


def _issue_priority(issue: Dict[str, Any]) -> int:
    severity = _SEVERITY_WEIGHT.get(issue.get("severity", "low"), 1)
    source_bonus = 2 if issue.get("source") == "ground_truth" else 0
    flagged_bonus = 1 if issue.get("flagged", False) else 0
    return severity + source_bonus + flagged_bonus


def _issue_history_stats(history: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for step in history:
        issue_id = step.get("target_issue_id")
        if not issue_id:
            continue
        if issue_id not in stats:
            stats[issue_id] = {
                "attempts": 0,
                "suggest_attempts": 0,
                "failed": 0,
                "succeeded": 0,
            }
        issue_stats = stats[issue_id]
        issue_stats["attempts"] += 1
        if step.get("action") == "suggest_fix":
            issue_stats["suggest_attempts"] += 1
        if step.get("fix_failed", False):
            issue_stats["failed"] += 1
        if step.get("fix_applied", False):
            issue_stats["succeeded"] += 1
    return stats


def _recent_diagnostics(history: List[Dict[str, Any]], window: int = 3) -> Dict[str, int]:
    recent = history[-window:]
    llm_values = [float(step.get("llm_score", 0.5)) for step in recent]
    avg_llm_recent = sum(llm_values) / float(len(llm_values)) if llm_values else 0.5
    return {
        "negative_rewards": sum(1 for step in recent if float(step.get("reward", 0.0)) < 0.0),
        "introduced_bugs": sum(1 for step in recent if step.get("introduced_bug_id") is not None),
        "failed_fixes": sum(1 for step in recent if step.get("fix_failed", False)),
        "low_llm_scores": sum(1 for step in recent if float(step.get("llm_score", 0.5)) < 0.4),
        "avg_llm_recent": int(round(avg_llm_recent * 100)),
    }


def _pick_target_issue(
    issues: List[Dict[str, Any]],
    stats: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    ranked = sorted(
        issues,
        key=lambda issue: (
            _issue_priority(issue),
            -stats.get(issue.get("id", ""), {}).get("failed", 0),
            issue.get("id", ""),
        ),
        reverse=True,
    )
    return ranked[0]


def _fallback_action(
    preferred: str,
    unresolved: List[Dict[str, Any]],
    stabilize_mode: bool,
) -> str:
    has_perf = any(issue.get("type") == "performance" for issue in unresolved)
    if preferred == "suggest_fix":
        return "flag_issue"
    if preferred == "flag_issue":
        if has_perf and not stabilize_mode:
            return "optimize_code"
        return "suggest_fix"
    if preferred == "optimize_code":
        return "suggest_fix"
    return "ignore"


def smart_agent(state: Dict[str, Any]) -> str:
    issues = state.get("issues", [])
    history = state.get("history", [])
    step_count = int(state.get("step_count", 0))
    max_steps = int(state.get("max_steps", 10))
    steps_left = max(max_steps - step_count, 0)

    unresolved = _unresolved(issues)
    if not unresolved:
        return "ignore"

    unresolved_ground_truth = [issue for issue in unresolved if issue.get("source") == "ground_truth"]
    review_pool = unresolved_ground_truth if unresolved_ground_truth else unresolved
    flagged_pool = [issue for issue in review_pool if issue.get("flagged", False)]
    unflagged_pool = [issue for issue in review_pool if not issue.get("flagged", False)]
    high_pool = [issue for issue in review_pool if issue.get("severity") == "high"]
    perf_pool = [issue for issue in review_pool if issue.get("type") == "performance"]

    stats = _issue_history_stats(history)
    diagnostics = _recent_diagnostics(history, window=3)

    closeout_mode = steps_left <= 2
    stabilize_mode = (
        diagnostics["introduced_bugs"] >= 2
        or diagnostics["negative_rewards"] >= 2
        or diagnostics["low_llm_scores"] >= 2
        or diagnostics["avg_llm_recent"] <= 38
    )
    confident_mode = diagnostics["avg_llm_recent"] >= 72

    last_action = history[-1]["action"] if history else None
    target = _pick_target_issue(review_pool, stats)
    target_stats = stats.get(target.get("id", ""), {})
    target_failed = target_stats.get("failed", 0)
    target_suggest_attempts = target_stats.get("suggest_attempts", 0)
    target_flagged = bool(target.get("flagged", False))

    forced = False
    if closeout_mode and flagged_pool:
        candidate = "suggest_fix"
        forced = True
    elif closeout_mode and high_pool:
        candidate = "flag_issue"
        forced = True
    elif target_flagged:
        if target_failed >= 2 and not closeout_mode:
            candidate = "flag_issue"
        else:
            candidate = "suggest_fix"
    elif target.get("type") == "performance":
        if stabilize_mode and flagged_pool:
            candidate = "suggest_fix"
        elif target_suggest_attempts == 0 and not closeout_mode and confident_mode:
            candidate = "optimize_code"
        else:
            candidate = "flag_issue"
    elif unflagged_pool:
        candidate = "flag_issue"
    elif flagged_pool:
        candidate = "suggest_fix"
    else:
        candidate = "suggest_fix"

    if stabilize_mode and candidate == "optimize_code":
        candidate = "suggest_fix" if flagged_pool else "flag_issue"
    if not confident_mode and candidate == "optimize_code":
        candidate = "flag_issue"

    if candidate == "optimize_code" and not perf_pool:
        candidate = "flag_issue"

    if candidate != last_action or forced:
        return candidate

    fallback = _fallback_action(candidate, review_pool, stabilize_mode)
    if fallback == "optimize_code" and not perf_pool:
        fallback = "flag_issue"
    return fallback


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
    llm_scores = [float(step.get("llm_score", 0.0)) for step in state["history"]]
    avg_llm_score = sum(llm_scores) / float(len(llm_scores)) if llm_scores else 0.0
    negative_steps = sum(1 for step in state["history"] if float(step.get("reward", 0.0)) < 0.0)
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
        "avg_llm_score": float(avg_llm_score),
        "negative_steps": negative_steps,
        "llm_mode": state.get("llm_evaluation_mode", "heuristic"),
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
            f"remaining_total={result['remaining_total_issues']} "
            f"avg_llm={result['avg_llm_score']:.3f}"
        )


if __name__ == "__main__":
    main()
