from typing import Any, Dict, List, Optional

ACTIONS = [
    "flag_issue",
    "suggest_fix",
    "optimize_code",
    "ignore",
]

_SEVERITY_WEIGHT = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def validate_action(action: str) -> None:
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}. Valid actions: {ACTIONS}")


def _highest_priority_issue(issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not issues:
        return None
    return sorted(
        issues,
        key=lambda issue: (
            _SEVERITY_WEIGHT.get(issue.get("severity", "low"), 1),
            issue.get("id", ""),
        ),
        reverse=True,
    )[0]


def _unresolved_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [issue for issue in issues if not issue.get("resolved", False)]


def _actionable_issues(
    candidates: List[Dict[str, Any]],
    all_issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    unresolved_ids = {
        issue.get("id")
        for issue in all_issues
        if not issue.get("resolved", False)
    }
    actionable: List[Dict[str, Any]] = []
    for issue in candidates:
        requires = issue.get("requires", [])
        if all(req_id not in unresolved_ids for req_id in requires):
            actionable.append(issue)
    return actionable


def apply_action(
    action: str,
    issues: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    validate_action(action)

    unresolved = _unresolved_issues(issues)
    unresolved_ground_truth = [issue for issue in unresolved if issue.get("source") == "ground_truth"]
    unresolved_introduced = [issue for issue in unresolved if issue.get("source") == "introduced_bug"]
    repeated = bool(history) and history[-1].get("action") == action
    previously_detected_targets = {
        step.get("target_issue_id")
        for step in history
        if step.get("correct_detection", False) and step.get("target_issue_id") is not None
    }

    effect: Dict[str, Any] = {
        "action": action,
        "target_issue_id": None,
        "target_source": None,
        "correct_detection": False,
        "high_severity": False,
        "false_positive": False,
        "repeated": repeated,
        "unnecessary": False,
        "repeat_target": False,
    }

    if action == "flag_issue":
        target_pool = unresolved_ground_truth if unresolved_ground_truth else unresolved_introduced
        actionable_pool = _actionable_issues(target_pool, unresolved)
        target = _highest_priority_issue(actionable_pool if actionable_pool else target_pool)
        if target is None:
            effect["false_positive"] = True
        else:
            target["flagged"] = True
            effect["target_issue_id"] = target["id"]
            effect["target_source"] = target.get("source")
            if target["id"] in previously_detected_targets:
                effect["repeat_target"] = True
            else:
                effect["correct_detection"] = True
                effect["high_severity"] = target.get("severity") == "high"
        return effect

    if action == "suggest_fix":
        flagged = [issue for issue in unresolved if issue.get("flagged", False)]
        flagged_ground_truth = [issue for issue in flagged if issue.get("source") == "ground_truth"]
        target_pool: List[Dict[str, Any]]
        if flagged_ground_truth:
            target_pool = flagged_ground_truth
        elif flagged:
            target_pool = flagged
        elif unresolved_ground_truth:
            target_pool = unresolved_ground_truth
        else:
            target_pool = unresolved
        actionable_pool = _actionable_issues(target_pool, unresolved)
        target = _highest_priority_issue(actionable_pool if actionable_pool else target_pool)
        if target is None:
            effect["false_positive"] = True
        else:
            effect["target_issue_id"] = target["id"]
            effect["target_source"] = target.get("source")
            effect["correct_detection"] = True
            effect["high_severity"] = target.get("severity") == "high"
            previous_suggest_fix_attempt = any(
                step.get("action") == "suggest_fix"
                and step.get("target_issue_id") == target["id"]
                for step in history
            )
            if previous_suggest_fix_attempt:
                effect["repeat_target"] = True
        return effect

    if action == "optimize_code":
        perf_issues = [issue for issue in unresolved if issue.get("type") == "performance"]
        perf_ground_truth = [issue for issue in perf_issues if issue.get("source") == "ground_truth"]
        target_pool = perf_ground_truth if perf_ground_truth else perf_issues
        actionable_pool = _actionable_issues(target_pool, unresolved)
        target = _highest_priority_issue(actionable_pool if actionable_pool else target_pool)
        if target is None:
            effect["unnecessary"] = True
        else:
            effect["target_issue_id"] = target["id"]
            effect["target_source"] = target.get("source")
            if target["id"] in previously_detected_targets:
                effect["repeat_target"] = True
            else:
                effect["correct_detection"] = True
                effect["high_severity"] = target.get("severity") == "high"
        return effect

    if action == "ignore":
        if unresolved:
            effect["unnecessary"] = True
        return effect

    return effect
