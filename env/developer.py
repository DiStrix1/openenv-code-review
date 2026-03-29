import copy
import random
from typing import Any, Dict, List, Tuple

random.seed(42)


class DeveloperSimulator:
    def __init__(self, difficulty: str) -> None:
        self.difficulty = difficulty
        self._rng = random.Random(42)
        self._introduced_issue_index = 0
        self._config = {
            "easy": {"fix_prob": 0.82, "new_bug_prob": 0.04},
            "medium": {"fix_prob": 0.50, "new_bug_prob": 0.12},
            "hard": {"fix_prob": 0.45, "new_bug_prob": 0.25},
        }
        if difficulty not in self._config:
            raise ValueError(f"Unsupported difficulty: {difficulty}")

    def reset(self) -> None:
        self._rng = random.Random(42)
        self._introduced_issue_index = 0

    def _find_issue(self, issues: List[Dict[str, Any]], issue_id: str) -> Dict[str, Any]:
        for issue in issues:
            if issue.get("id") == issue_id:
                return issue
        return {}

    def _resolve_issue(
        self,
        issue: Dict[str, Any],
        code: str,
        step_count: int,
    ) -> str:
        issue["resolved"] = True
        issue["flagged"] = True
        return f"{code}\n# developer_fix_step_{step_count}_{issue['id']}"

    def _introduce_new_bug(self, issues: List[Dict[str, Any]], step_count: int) -> Dict[str, Any]:
        self._introduced_issue_index += 1
        severity_table = {
            "easy": ["low", "medium", "medium"],
            "medium": ["medium", "medium", "high"],
            "hard": ["medium", "medium", "high"],
        }
        issue_type_table = {
            "easy": ["style", "logic", "performance"],
            "medium": ["logic", "performance", "security"],
            "hard": ["security", "logic", "performance"],
        }
        severity = severity_table[self.difficulty][self._rng.randint(0, 2)]
        issue_type = issue_type_table[self.difficulty][self._rng.randint(0, 2)]
        new_issue = {
            "id": f"N{self._introduced_issue_index}",
            "description": f"Regression introduced at step {step_count}",
            "severity": severity,
            "type": issue_type,
            "resolved": False,
            "flagged": False,
            "source": "introduced_bug",
        }
        issues.append(new_issue)
        return new_issue

    def respond(
        self,
        action: str,
        code: str,
        issues: List[Dict[str, Any]],
        action_effect: Dict[str, Any],
        step_count: int,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        updated_issues = copy.deepcopy(issues)
        result: Dict[str, Any] = {
            "fix_applied": False,
            "fix_failed": False,
            "fixed_issue_id": None,
            "fixed_issue_source": None,
            "introduced_bug_id": None,
        }
        config = self._config[self.difficulty]
        target_id = action_effect.get("target_issue_id")

        if target_id:
            target_issue = self._find_issue(updated_issues, target_id)
        else:
            target_issue = {}

        if target_issue and not target_issue.get("resolved", False):
            fix_prob = config["fix_prob"]
            if action == "flag_issue":
                fix_prob *= 0.65
            elif action == "optimize_code":
                if target_issue.get("type") == "performance":
                    fix_prob *= 0.75
                else:
                    fix_prob *= 0.35
            elif action == "suggest_fix":
                fix_prob *= 1.05
            if action_effect.get("repeat_target", False):
                if self.difficulty == "easy":
                    fix_prob *= 0.85
                elif self.difficulty == "medium":
                    fix_prob *= 0.65
                else:
                    fix_prob *= 0.60

            if target_issue.get("severity") == "high":
                fix_prob -= 0.05
            if target_issue.get("source") == "ground_truth":
                fix_prob += 0.03
            elif target_issue.get("source") == "introduced_bug":
                fix_prob -= 0.05
            if self.difficulty == "hard":
                fix_prob -= 0.02
            fatigue_penalty = max(step_count - 4, 0) * 0.02
            fix_prob -= fatigue_penalty
            fix_prob = min(max(fix_prob, 0.05), 0.95)

            if self._rng.random() < fix_prob:
                code = self._resolve_issue(target_issue, code, step_count)
                result["fix_applied"] = True
                result["fixed_issue_id"] = target_issue["id"]
                result["fixed_issue_source"] = target_issue.get("source")
            else:
                result["fix_failed"] = True

        new_bug_prob = config["new_bug_prob"]
        if action == "suggest_fix":
            new_bug_prob *= 0.75
        elif action == "flag_issue":
            new_bug_prob *= 0.90
        elif action == "optimize_code":
            new_bug_prob *= 1.10
        elif action == "ignore":
            new_bug_prob *= 1.15

        if self.difficulty == "hard":
            new_bug_prob += 0.02
            new_bug_prob += 0.008 * max(step_count - 1, 0)

        unresolved_introduced = sum(
            1
            for issue in updated_issues
            if issue.get("source") == "introduced_bug" and not issue.get("resolved", False)
        )
        max_unresolved_introduced = {
            "easy": 1,
            "medium": 2,
            "hard": 2,
        }[self.difficulty]
        can_introduce = unresolved_introduced < max_unresolved_introduced

        if can_introduce and self._rng.random() < min(new_bug_prob, 0.95):
            new_issue = self._introduce_new_bug(updated_issues, step_count)
            code = f"{code}\n# regression_note_step_{step_count}_{new_issue['id']}"
            result["introduced_bug_id"] = new_issue["id"]

        return code, updated_issues, result
