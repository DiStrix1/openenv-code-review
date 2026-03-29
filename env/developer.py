import copy
import random
from typing import Any, Dict, List, Optional, Tuple

random.seed(42)


class DeveloperSimulator:
    def __init__(self, difficulty: str) -> None:
        self.difficulty = difficulty
        self._rng = random.Random(42)
        self._introduced_issue_index = 0
        self._config = {
            "easy": {"fix_prob": 0.78, "new_bug_prob": 0.05, "followup_prob": 0.03},
            "medium": {"fix_prob": 0.52, "new_bug_prob": 0.14, "followup_prob": 0.08},
            "hard": {"fix_prob": 0.40, "new_bug_prob": 0.20, "followup_prob": 0.10},
        }
        self._bug_library = {
            "easy": [
                {
                    "description": "Refactor introduced minor naming inconsistency.",
                    "severity": "low",
                    "type": "style",
                    "context_tags": ["readability"],
                },
                {
                    "description": "Boundary check skipped for empty input.",
                    "severity": "medium",
                    "type": "logic",
                    "context_tags": ["edge_case"],
                },
                {
                    "description": "Repeated loop adds avoidable overhead.",
                    "severity": "medium",
                    "type": "performance",
                    "context_tags": ["loop"],
                },
            ],
            "medium": [
                {
                    "description": "Fix omitted null-guard in secondary path.",
                    "severity": "medium",
                    "type": "logic",
                    "context_tags": ["fallback_path"],
                },
                {
                    "description": "Validation weakened while handling malformed data.",
                    "severity": "high",
                    "type": "security",
                    "context_tags": ["validation"],
                },
                {
                    "description": "Caching patch causes stale lookup behavior.",
                    "severity": "medium",
                    "type": "performance",
                    "context_tags": ["cache"],
                },
                {
                    "description": "Error pathway leaks partial internal details.",
                    "severity": "high",
                    "type": "security",
                    "context_tags": ["logging"],
                },
            ],
            "hard": [
                {
                    "description": "Hotfix bypassed input constraints in one branch.",
                    "severity": "high",
                    "type": "security",
                    "context_tags": ["branch_divergence", "validation"],
                },
                {
                    "description": "Refactor introduced subtle state-carryover bug.",
                    "severity": "high",
                    "type": "logic",
                    "context_tags": ["shared_state"],
                },
                {
                    "description": "Optimization attempt created contention under load.",
                    "severity": "medium",
                    "type": "performance",
                    "context_tags": ["contention", "hot_path"],
                },
                {
                    "description": "Conditional fallback leaks sensitive debugging details.",
                    "severity": "high",
                    "type": "security",
                    "context_tags": ["logging", "fallback_path"],
                },
                {
                    "description": "Silent conversion fallback masks data corruption.",
                    "severity": "medium",
                    "type": "logic",
                    "context_tags": ["data_integrity"],
                },
            ],
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

    def _context_counts(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        unresolved = [issue for issue in issues if not issue.get("resolved", False)]
        return {
            "unresolved_total": len(unresolved),
            "unresolved_high_security": sum(
                1
                for issue in unresolved
                if issue.get("type") == "security" and issue.get("severity") == "high"
            ),
            "unresolved_performance": sum(
                1 for issue in unresolved if issue.get("type") == "performance"
            ),
            "unresolved_introduced": sum(
                1 for issue in unresolved if issue.get("source") == "introduced_bug"
            ),
        }

    def _select_bug_template(self, preferred_type: Optional[str]) -> Dict[str, Any]:
        candidates = self._bug_library[self.difficulty]
        if preferred_type is not None:
            preferred = [item for item in candidates if item.get("type") == preferred_type]
            if preferred:
                candidates = preferred
        return copy.deepcopy(candidates[self._rng.randint(0, len(candidates) - 1)])

    def _introduce_new_bug(
        self,
        issues: List[Dict[str, Any]],
        step_count: int,
        origin: str,
        preferred_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._introduced_issue_index += 1
        template = self._select_bug_template(preferred_type=preferred_type)
        new_issue = {
            "id": f"N{self._introduced_issue_index}",
            "description": template["description"],
            "severity": template["severity"],
            "type": template["type"],
            "resolved": False,
            "flagged": False,
            "source": "introduced_bug",
            "origin": origin,
            "context_tags": template.get("context_tags", []),
            "introduced_step": step_count,
        }
        issues.append(new_issue)
        return new_issue

    def _compute_fix_probability(
        self,
        action: str,
        target_issue: Dict[str, Any],
        action_effect: Dict[str, Any],
        step_count: int,
        issues: List[Dict[str, Any]],
        llm_guidance: float,
    ) -> float:
        context = self._context_counts(issues)
        config = self._config[self.difficulty]
        fix_prob = float(config["fix_prob"])

        if action == "flag_issue":
            fix_prob *= 0.62
        elif action == "optimize_code":
            if target_issue.get("type") == "performance":
                fix_prob *= 0.74
            else:
                fix_prob *= 0.32
        elif action == "suggest_fix":
            fix_prob *= 1.05

        if target_issue.get("flagged", False):
            fix_prob += 0.07
        if action_effect.get("repeat_target", False):
            fix_prob *= {
                "easy": 0.88,
                "medium": 0.72,
                "hard": 0.62,
            }[self.difficulty]

        if target_issue.get("severity") == "high":
            fix_prob -= 0.05
        if target_issue.get("source") == "ground_truth":
            fix_prob += 0.03
        if target_issue.get("source") == "introduced_bug":
            fix_prob -= 0.06
        if target_issue.get("misleading_pattern"):
            fix_prob -= 0.05
        if target_issue.get("noisy_signal"):
            fix_prob -= 0.04
        if target_issue.get("requires"):
            unresolved_ids = {
                issue.get("id")
                for issue in issues
                if not issue.get("resolved", False)
            }
            if any(req in unresolved_ids for req in target_issue["requires"]):
                fix_prob -= 0.15

        fix_prob -= 0.02 * context["unresolved_introduced"]
        fix_prob -= 0.015 * max(step_count - 4, 0)
        if context["unresolved_high_security"] >= 2 and target_issue.get("type") != "security":
            fix_prob -= 0.04
        if self.difficulty == "hard":
            fix_prob -= 0.07

        # LLM guidance influences execution quality.
        llm_shift = (float(llm_guidance) - 0.5) * 0.10
        fix_prob += llm_shift

        return min(max(fix_prob, 0.05), 0.95)

    def _compute_new_bug_probability(
        self,
        action: str,
        action_effect: Dict[str, Any],
        step_count: int,
        issues: List[Dict[str, Any]],
        llm_guidance: float,
    ) -> float:
        context = self._context_counts(issues)
        new_bug_prob = float(self._config[self.difficulty]["new_bug_prob"])

        if action == "suggest_fix":
            new_bug_prob *= 0.78
        elif action == "flag_issue":
            new_bug_prob *= 0.88
        elif action == "optimize_code":
            new_bug_prob *= 1.18
        elif action == "ignore":
            new_bug_prob *= 1.25

        if action_effect.get("false_positive", False):
            new_bug_prob += 0.04
        if action_effect.get("unnecessary", False):
            new_bug_prob += 0.05
        if action_effect.get("repeat_target", False):
            new_bug_prob += 0.03

        new_bug_prob += 0.006 * min(context["unresolved_total"], 8)
        new_bug_prob += 0.007 * max(step_count - 1, 0)

        if self.difficulty == "hard":
            new_bug_prob += 0.005
            if context["unresolved_introduced"] >= 2:
                new_bug_prob *= 0.78
            if step_count >= 7:
                new_bug_prob *= 0.90

        # Better LLM guidance reduces regression probability.
        llm_shift = (float(llm_guidance) - 0.5) * 0.08
        new_bug_prob -= llm_shift

        return min(max(new_bug_prob, 0.0), 0.95)

    def respond(
        self,
        action: str,
        code: str,
        issues: List[Dict[str, Any]],
        action_effect: Dict[str, Any],
        step_count: int,
        llm_guidance: float = 0.5,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        updated_issues = copy.deepcopy(issues)
        result: Dict[str, Any] = {
            "fix_applied": False,
            "fix_failed": False,
            "fixed_issue_id": None,
            "fixed_issue_source": None,
            "introduced_bug_id": None,
            "followup_bug_id": None,
            "severity_escalated_issue_id": None,
            "llm_guidance": round(float(llm_guidance), 3),
        }
        target_id = action_effect.get("target_issue_id")

        if target_id:
            target_issue = self._find_issue(updated_issues, target_id)
        else:
            target_issue = {}

        if target_issue and not target_issue.get("resolved", False):
            fix_prob = self._compute_fix_probability(
                action=action,
                target_issue=target_issue,
                action_effect=action_effect,
                step_count=step_count,
                issues=updated_issues,
                llm_guidance=llm_guidance,
            )

            if self._rng.random() < fix_prob:
                code = self._resolve_issue(target_issue, code, step_count)
                result["fix_applied"] = True
                result["fixed_issue_id"] = target_issue["id"]
                result["fixed_issue_source"] = target_issue.get("source")

                followup_prob = self._config[self.difficulty]["followup_prob"]
                if action == "optimize_code":
                    followup_prob *= 1.20
                if target_issue.get("type") == "security":
                    followup_prob *= 1.10
                if target_issue.get("source") == "introduced_bug":
                    followup_prob *= 1.15
                if action_effect.get("repeat_target", False):
                    followup_prob *= 1.10

                unresolved_introduced = sum(
                    1
                    for issue in updated_issues
                    if issue.get("source") == "introduced_bug" and not issue.get("resolved", False)
                )
                max_unresolved_introduced = {
                    "easy": 1,
                    "medium": 2,
                    "hard": 1,
                }[self.difficulty]
                can_add_followup = unresolved_introduced < max_unresolved_introduced

                if can_add_followup and self._rng.random() < min(followup_prob, 0.85):
                    preferred_followup_type = {
                        "security": "logic",
                        "performance": "security",
                        "logic": "performance",
                        "style": "logic",
                    }.get(target_issue.get("type"), None)
                    followup_issue = self._introduce_new_bug(
                        updated_issues,
                        step_count=step_count,
                        origin="followup_after_fix",
                        preferred_type=preferred_followup_type,
                    )
                    code = f"{code}\n# followup_regression_step_{step_count}_{followup_issue['id']}"
                    result["followup_bug_id"] = followup_issue["id"]
            else:
                result["fix_failed"] = True
                if (
                    self.difficulty in {"medium", "hard"}
                    and target_issue.get("severity") == "medium"
                    and self._rng.random() < 0.25
                ):
                    target_issue["severity"] = "high"
                    result["severity_escalated_issue_id"] = target_issue["id"]

        new_bug_prob = self._compute_new_bug_probability(
            action=action,
            action_effect=action_effect,
            step_count=step_count,
            issues=updated_issues,
            llm_guidance=llm_guidance,
        )
        if result["fix_failed"]:
            new_bug_prob += 0.06
        if result["fix_applied"]:
            new_bug_prob -= 0.04

        unresolved_introduced = sum(
            1
            for issue in updated_issues
            if issue.get("source") == "introduced_bug" and not issue.get("resolved", False)
        )
        max_unresolved_introduced = {
            "easy": 1,
            "medium": 2,
            "hard": 1,
        }[self.difficulty]
        can_introduce = unresolved_introduced < max_unresolved_introduced

        if can_introduce and self._rng.random() < min(new_bug_prob, 0.95):
            preferred_type = None
            if target_issue:
                preferred_type = {
                    "security": "logic",
                    "logic": "security",
                    "performance": "performance",
                    "style": "logic",
                }.get(target_issue.get("type"), None)
            new_issue = self._introduce_new_bug(
                updated_issues,
                step_count=step_count,
                origin="developer_regression",
                preferred_type=preferred_type,
            )
            code = f"{code}\n# regression_note_step_{step_count}_{new_issue['id']}"
            result["introduced_bug_id"] = new_issue["id"]

        return code, updated_issues, result
