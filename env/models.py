from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except Exception:
    from pydantic import BaseModel, Field

    class Action(BaseModel):
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        done: bool = False
        reward: float | None = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0


class CodeReviewAction(Action):
    action: Literal["flag_issue", "suggest_fix", "optimize_code", "ignore"]


class CodeReviewObservation(Observation):
    code: str
    issues: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    step_count: int


class CodeReviewState(State):
    code: str = ""
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    total_issues: int = 0
    difficulty: str = "medium"
    max_steps: int = 10
