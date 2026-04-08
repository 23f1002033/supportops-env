from pydantic import BaseModel, Field
from typing import List, Optional


class SupportAction(BaseModel):
    """
    What the agent sends to the environment.
    """
    response: str = Field(..., min_length=1, max_length=500)


class SupportObservation(BaseModel):
    """
    What the agent sees at each step.
    """
    user_message: str
    sentiment: float = Field(..., gt=-1, lt=1)
    resolved: bool
    step_count: int
    urgency: float = Field(0, gt=0, lt=1)


class SupportState(BaseModel):
    """
    Internal environment state (NOT visible to agent).
    """
    conversation: List[str] = []
    step_count: int = 0

    patience: float = Field(..., gt=0, lt=1)
    trust: float = Field(..., gt=0, lt=1)
    churn_risk: float = Field(..., gt=0, lt=1)

    task_name: str
    difficulty: str = "easy"
    expected_intent: str
    resolved: bool = False
    resolution_type: Optional[str] = None


class StepResult(BaseModel):
    """
    Structured return (useful for API / debugging).
    """
    observation: SupportObservation
    reward: float
    done: bool
    info: dict


class GradeResult(BaseModel):
    """
    Structured grading output with breakdown.
    """
    score: float = Field(..., gt=0, lt=1)
    resolution_score: float
    efficiency_score: float
    trust_score: float
    patience_score: float
    churn_score: float
    breakdown: dict