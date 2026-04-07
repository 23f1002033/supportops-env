from pydantic import BaseModel, Field
from typing import List


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
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    resolved: bool
    step_count: int 


class SupportState(BaseModel):
    """
    Internal environment state (NOT visible to agent).
    """
    conversation: List[str] = []
    step_count: int = 0

    patience: float = Field(..., ge=0.0, le=1.0)
    trust: float = Field(..., ge=0.0, le=1.0)
    churn_risk: float = Field(..., ge=0.0, le=1.0)

    task_name: str
    expected_intent: str

class StepResult(BaseModel):
    """
    Structured return (useful for API / debugging).
    """
    observation: SupportObservation
    reward: float
    done: bool
    info: dict