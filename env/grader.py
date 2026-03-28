from env.models import PullRequest
def grade(pr: PullRequest):
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
        return 0.0

    return round(score / total_weight, 2)
