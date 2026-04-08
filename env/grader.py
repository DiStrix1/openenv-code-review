from env.models import PullRequest

_SCORE_EPSILON = 0.01


def _strict_unit_interval(value: float) -> float:
    return max(_SCORE_EPSILON, min(value, 1.0 - _SCORE_EPSILON))


def grade(pr: PullRequest) -> float:
    score = 0.0
    total_weight = 0.0

    for issue in pr.issues:
        weight = 1.0

        if issue.severity == "high":
            weight = 2.0
        elif issue.severity == "medium":
            weight = 1.5

        total_weight += weight

        if issue.resolved:
            score += weight

    if total_weight == 0:
        return 0.5

    return _strict_unit_interval(score / total_weight)
