from typing import Any, Dict, List


def grade(state: Dict[str, Any]) -> float:
    issues: List[Dict[str, Any]] = state.get("issues", [])
    total = int(state.get("total_issues", 0))
    if total <= 0:
        return 0.0

    remaining = len(issues)
    if remaining == 0:
        return 1.0

    return max(0.0, (total - remaining) / float(total))

