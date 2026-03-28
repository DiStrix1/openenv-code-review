from pathlib import Path

import yaml

from env.models import PullRequest, Observation, Action, Reward
from env.tasks import BaseTask
from env.developer import DeveloperSimulator


class CodeReviewEnv:
    def __init__(self, task: BaseTask):
        self.task = task
        self.config = self._load_config()
        self._validate_config()
        self._validate_task_config(task)
        self.developer = DeveloperSimulator(profile=task.get_developer_profile())
        self.pr = None

    def _load_config(self):
        config_path = Path(__file__).resolve().parents[1] / "openenv.yaml"
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _validate_task_config(self, task: BaseTask):
        configured_tasks = self.config.get("tasks", [])
        if getattr(task, "name", "") not in configured_tasks:
            raise ValueError(f"Task '{task.name}' is not declared in openenv.yaml")

    def _validate_config(self):
        if "reward" not in self.config:
            raise ValueError("openenv.yaml is missing required 'reward' section")

    # Start a new episode.
    def reset(self):
        self.pr = self.task.create_pr()
        return self._get_observation()

    # Return full internal state (not for agent).
    def state(self):
        return self.pr

    # What agent sees (no ground-truth leakage).
    def _get_observation(self):
        return Observation(
            diff=self.pr.diff,
            comments=self.pr.comments,
            iteration=self.pr.iteration
        )

    # Core interaction loop.
    def step(self, action: Action):
        reward = 0.0
        done = False

        # =========================
        # 1. APPLY AGENT ACTION
        # =========================
        if action.action_type == "comment":

            if action.issue_id is None:
                reward -= 0.3

            else:
                # Add comment for developer interaction.
                self.pr.comments.append(f"ISSUE_{action.issue_id}")

                # Penalize repeated comments.
                if self.pr.comments.count(f"ISSUE_{action.issue_id}") > 1:
                    reward -= 0.2

                issue = next(
                    (i for i in self.pr.issues if i.id == action.issue_id),
                    None
                )

                if issue:
                    if not issue.resolved:
                        reward += 0.3

                        # severity bonus
                        if issue.severity == "high":
                            reward += 0.2
                        elif issue.severity == "medium":
                            reward += 0.1
                    else:
                        # commenting already resolved issue
                        reward -= 0.2
                else:
                    # false positive
                    reward -= 0.5

        elif action.action_type == "approve":
            unresolved = [i for i in self.pr.issues if not i.resolved]

            if len(unresolved) == 0:
                reward += 1.0
                done = True
            else:
                reward -= 1.0

        elif action.action_type == "request_changes":
            reward += 0.2

        else:
            reward -= 0.2

        # =========================
        # 2. EFFICIENCY PENALTY
        # =========================
        reward -= 0.05

        # =========================
        # 3. ENVIRONMENT TRANSITION
        # =========================
        self.developer.apply_changes(self.pr)

        # update iteration
        self.pr.iteration += 1

        # =========================
        # 4. Critical bug penalty.
        # =========================
        for issue in self.pr.issues:
            if not issue.resolved and issue.severity == "high":
                reward -= 0.5

        # =========================
        # 5. TERMINATION CONDITION
        # =========================
        if self.pr.iteration >= 5:
            done = True

        # =========================
        # 6. RETURN
        # =========================
        return (
            self._get_observation(),
            Reward(value=reward, reason="step"),
            done,
            {}
        )