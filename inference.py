from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Body, FastAPI

from env.environment import CodeReviewEnv
from env.grader import grade
from env.models import CodeReviewAction, CodeReviewObservation, CodeReviewState

try:
    from openenv.core.env_server import create_app
    from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata

    OPENENV_AVAILABLE = True
except Exception:
    OPENENV_AVAILABLE = False

    class Environment:
        pass

    class EnvironmentMetadata:
        def __init__(
            self,
            name: str,
            description: str,
            version: str = "1.0.0",
            author: Optional[str] = None,
            documentation_url: Optional[str] = None,
        ) -> None:
            self.name = name
            self.description = description
            self.version = version
            self.author = author
            self.documentation_url = documentation_url


class CodeReviewEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._task_name = "medium"
        self._requested_max_steps = 10
        self._episode_id = str(uuid4())
        self._env = CodeReviewEnv(task=self._task_name, max_steps=self._requested_max_steps)
        initial_state = self._env.reset()
        self._state = self._build_state(initial_state)

    def _build_state(self, state: Dict[str, Any]) -> CodeReviewState:
        history = state["history"]
        last_llm_score = float(history[-1].get("llm_score", 0.0)) if history else 0.0
        return CodeReviewState(
            episode_id=self._episode_id,
            step_count=state["step_count"],
            code=state["code"],
            issues=state["issues"],
            history=history,
            total_issues=state.get("total_issues", 0),
            difficulty=self._task_name,
            max_steps=self._env.max_steps,
            llm_evaluation_mode=state.get("llm_evaluation_mode", "heuristic"),
            last_llm_score=last_llm_score,
        )

    def _build_observation(
        self,
        state: Dict[str, Any],
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]],
        event: str,
    ) -> CodeReviewObservation:
        metadata = {
            "event": event,
            "difficulty": self._task_name,
            "max_steps": self._env.max_steps,
            "grade": grade(state),
            "info": info or {},
        }
        return CodeReviewObservation(
            code=state["code"],
            issues=state["issues"],
            history=state["history"],
            step_count=state["step_count"],
            reward=float(reward),
            done=done,
            metadata=metadata,
        )

    def _validate_task(self, task: str) -> str:
        valid = {"easy", "medium", "hard"}
        if task not in valid:
            raise ValueError(f"Invalid task '{task}'. Expected one of {sorted(valid)}.")
        return task

    def _resolve_max_steps(self, max_steps: Optional[int]) -> int:
        if max_steps is None:
            return self._requested_max_steps
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than 0.")
        return int(max_steps)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        max_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        _ = seed
        _ = kwargs
        if task is not None:
            self._task_name = self._validate_task(task)
        self._requested_max_steps = self._resolve_max_steps(max_steps)
        self._episode_id = episode_id or str(uuid4())
        self._env = CodeReviewEnv(task=self._task_name, max_steps=self._requested_max_steps)
        state = self._env.reset()
        self._state = self._build_state(state)
        return self._build_observation(state, reward=0.0, done=False, info={}, event="reset")

    def step(
        self,
        action: CodeReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        _ = timeout_s
        _ = kwargs
        next_state, reward, done, info = self._env.step(action.action)
        self._state = self._build_state(next_state)
        return self._build_observation(
            next_state,
            reward=float(reward),
            done=done,
            info=info,
            event="step",
        )

    @property
    def state(self) -> CodeReviewState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="CodeReviewEnvironment",
            description=(
                "Multi-step code review simulation with deterministic rewards and "
                "integrated step-level LLM evaluation signals."
            ),
            version="0.1.0",
            author="OpenEnv Hackathon Submission",
            documentation_url="https://huggingface.co/spaces",
        )


if OPENENV_AVAILABLE:
    app = create_app(
        CodeReviewEnvironment,
        CodeReviewAction,
        CodeReviewObservation,
        env_name="code_review_env",
    )
else:
    app = FastAPI(title="Code Review Environment API", version="0.1.0")
    _env_instance = CodeReviewEnvironment()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "healthy", "openenv_available": False}

    @app.post("/reset")
    def reset_route(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        observation = _env_instance.reset(**payload)
        obs_data = observation.model_dump() if hasattr(observation, "model_dump") else dict(observation)
        return {
            "observation": obs_data,
            "reward": observation.reward,
            "done": observation.done,
        }

    @app.post("/step")
    def step_route(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        action_payload = payload.get("action", payload)
        action = CodeReviewAction(**action_payload)
        observation = _env_instance.step(action=action)
        obs_data = observation.model_dump() if hasattr(observation, "model_dump") else dict(observation)
        return {
            "observation": obs_data,
            "reward": observation.reward,
            "done": observation.done,
        }

    @app.get("/state")
    def state_route() -> Dict[str, Any]:
        state = _env_instance.state
        return state.model_dump() if hasattr(state, "model_dump") else dict(state)


@app.get("/")
def root_route() -> Dict[str, Any]:
    return {
        "service": "code_review_env",
        "status": "running",
        "openenv_available": OPENENV_AVAILABLE,
        "docs": "/docs",
    }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
