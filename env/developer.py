import random
from env.models import PullRequest, Issue


class DeveloperSimulator:

    def __init__(self, profile=None):
        profile = profile or {}
        # Task difficulty affects only issue-fix probability.
        self.difficulty_factor = profile.get(
            "difficulty_factor",
            profile.get("fix_multiplier", 1.0),
        )

        # Keep other simulator behavior consistent across tasks.
        self.feedback_boost = 0.2
        self.unprompted_factor = 0.5
        self.max_fix_prob = 0.6
        self.bug_introduce_prob = 0.3
        self.bug_severity_weights = [0.5, 0.3, 0.2]

    def apply_changes(self, pr: PullRequest):
        """
        Simulates how a developer responds after the agent takes an action.
        """
        self._fix_issues(pr)
        self._maybe_introduce_bug(pr)

    def _fix_issues(self, pr: PullRequest):
        for issue in pr.issues:
            if issue.resolved:
                continue

            commented = f"ISSUE_{issue.id}" in pr.comments

        # base probability by severity
            if issue.severity == "high":
                base_prob = 0.3
            elif issue.severity == "medium":
                base_prob = 0.2
            else:
                base_prob = 0.1

        # boost if agent flagged it
            if commented:
                base_prob += self.feedback_boost

        # Small independent fixing chance to avoid deadlock.
            if not commented:
                base_prob *= self.unprompted_factor

            base_prob *= self.difficulty_factor

        # cap probability
            base_prob = min(base_prob, self.max_fix_prob)

            if random.random() < base_prob:
                issue.resolved = True

    def _maybe_introduce_bug(self, pr: PullRequest):
        """
        Simulate developer accidentally introducing new bugs.
        """
        if random.random() < self.bug_introduce_prob:
            new_id = len(pr.issues) + 1

            pr.issues.append(
                Issue(
                    id=new_id,
                    description="New bug introduced during fix",
                    severity=random.choices(
                        ["low", "medium", "high"],
                        weights=self.bug_severity_weights,
                        k=1,
                    )[0],
                    resolved=False
                )
            )