from pydantic import BaseModel, Field
from typing import List, Optional
class Issue(BaseModel):
    id: int
    description: str
    severity: str #low/medium/high
    resolved: bool = False
class PullRequest(BaseModel):
    diff: str
    issues: List[Issue]
    comments: List[str] = Field(default_factory=list)
    iteration: int = 0
class Observation(BaseModel):
    diff: str
    comments: List[str]
    iteration: int
class Action(BaseModel):
    action_type: str #comment/approve/request_changes
    issue_id: Optional[int] = None
    severity: Optional[str] = None
class Reward(BaseModel):
    value: float
    reason: str