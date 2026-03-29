import copy
from typing import Any, Dict, List, Tuple, Union

from .actions import ACTIONS, apply_action
from .developer import DeveloperSimulator
from .reward import compute_score
from .tasks import BaseTask, EasyTask, HardTask, MediumTask


class CodeReviewEnv:
    def __init__(self, task: Union[str, BaseTask], max_steps: int = 12) -> None:
        self.task = self._resolve_task(task)
        self.max_steps = self._resolve_max_steps(max_steps, self.task.difficulty)
        self.step_count = 0
        self.done = False
        self.history: List[Dict[str, Any]] = []

        initial = self.task.initial_state()
        self.code = initial["code"]
        self.issues = initial["issues"]
        self.total_issues = sum(
            1 for issue in self.issues if issue.get("source") == "ground_truth"
        )

        self.developer = DeveloperSimulator(self.task.difficulty)

    def _resolve_task(self, task: Union[str, BaseTask]) -> BaseTask:
        if isinstance(task, BaseTask):
            return task
        if task == "easy":
            return EasyTask()
        if task == "medium":
            return MediumTask()
        if task == "hard":
            return HardTask()
        raise ValueError("task must be one of: 'easy', 'medium', 'hard', or a BaseTask instance")

    def _resolve_max_steps(self, max_steps: int, difficulty: str) -> int:
        difficulty_caps = {
            "easy": 12,
            "medium": 9,
            "hard": 10,
        }
        fallback = difficulty_caps[difficulty]
        requested = max_steps if max_steps > 0 else fallback
        return min(requested, difficulty_caps[difficulty])

    def _unresolved_ground_truth(self) -> List[Dict[str, Any]]:
        return [
            issue
            for issue in self.issues
            if issue.get("source") == "ground_truth" and not issue.get("resolved", False)
        ]

    def _all_ground_truth_resolved(self) -> bool:
        return len(self._unresolved_ground_truth()) == 0

    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self.done = False
        self.history = []
        initial = self.task.initial_state()
        self.code = initial["code"]
        self.issues = initial["issues"]
        self.total_issues = sum(
            1 for issue in self.issues if issue.get("source") == "ground_truth"
        )
        self.developer.reset()
        return self.state()

    def state(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "issues": copy.deepcopy(self._unresolved_ground_truth()),
            "history": copy.deepcopy(self.history),
            "step_count": self.step_count,
            "total_issues": self.total_issues,
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Expected one of {ACTIONS}")

        if self.done:
            return self.state(), 0.0, True, {"reason": "episode_already_done"}

        score_before = compute_score(
            history=self.history,
            remaining_issues=len(self._unresolved_ground_truth()),
            step_count=self.step_count,
            done=False,
            max_steps=self.max_steps,
        )

        action_effect = apply_action(action, self.issues, self.history)

        self.step_count += 1
        self.code, self.issues, developer_result = self.developer.respond(
            action=action,
            code=self.code,
            issues=self.issues,
            action_effect=action_effect,
            step_count=self.step_count,
        )

        history_entry = {
            "step": self.step_count,
            "action": action,
            "target_issue_id": action_effect.get("target_issue_id"),
            "target_source": action_effect.get("target_source"),
            "correct_detection": action_effect.get("correct_detection", False),
            "high_severity": action_effect.get("high_severity", False),
            "false_positive": action_effect.get("false_positive", False),
            "repeated": action_effect.get("repeated", False),
            "unnecessary": action_effect.get("unnecessary", False),
            "repeat_target": action_effect.get("repeat_target", False),
            "fix_applied": developer_result.get("fix_applied", False),
            "fixed_issue_id": developer_result.get("fixed_issue_id"),
            "fixed_issue_source": developer_result.get("fixed_issue_source"),
            "fix_failed": developer_result.get("fix_failed", False),
            "introduced_bug_id": developer_result.get("introduced_bug_id"),
        }
        self.history.append(history_entry)

        terminated_by_steps = self.step_count >= self.max_steps
        terminated_by_resolution = self._all_ground_truth_resolved()
        self.done = terminated_by_steps or terminated_by_resolution

        score_after = compute_score(
            history=self.history,
            remaining_issues=len(self._unresolved_ground_truth()),
            step_count=self.step_count,
            done=self.done,
            max_steps=self.max_steps,
        )
        reward = float(score_after - score_before)
        history_entry["reward"] = reward

        unresolved_total = [
            issue for issue in self.issues if not issue.get("resolved", False)
        ]

        info = {
            "difficulty": self.task.difficulty,
            "max_steps": self.max_steps,
            "unresolved_ground_truth_count": len(self._unresolved_ground_truth()),
            "unresolved_total_count": len(unresolved_total),
            "terminated_by_steps": terminated_by_steps,
            "terminated_by_resolution": terminated_by_resolution,
            "developer_result": developer_result,
            "score_before": score_before,
            "score_after": score_after,
        }
        return self.state(), reward, self.done, info
