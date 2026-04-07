from typing import Dict, Tuple
from env.models import SupportAction, SupportObservation, SupportState
import json
import os


class SupportEnv:
    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self._state: SupportState | None = None
        self._current_task: str = "easy"

    def load_task(self, task_name):
        path = os.path.join("env", "tasks", f"{task_name}.json")
        with open(path, "r") as f:
            return json.load(f)

    def reset(self, task: str = "easy") -> SupportObservation:
        self._current_task = task
        task_data = self.load_task(task)

        self._state = SupportState(
            conversation=[],
            step_count=0,
            patience=1.0,
            trust=0.4,
            churn_risk=0.3,
            task_name=task_data["task_name"],
            expected_intent=task_data["expected_intent"]
        )

        return SupportObservation(
            user_message=task_data["initial_message"],
            sentiment=-0.5,
            resolved=False,
            step_count=0
        )

    def step(self, action: SupportAction) -> Tuple[SupportObservation, float, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        self._state.conversation.append(action.response)

        text = action.response.lower()
        reward = 0.0

        if "sorry" in text or "apologize" in text:
            reward += 0.2
            self._state.trust = min(1.0, self._state.trust + 0.1)

        if "refund" in text:
            reward += 0.2

        if "process" in text or "initiated" in text:
            reward += 0.2

        if "help" in text:
            reward += 0.1

        if len(self._state.conversation) >= 2 and self._state.conversation[-1] == self._state.conversation[-2]:
            reward -= 0.3

        if "provide" in text and "details" in text:
            reward -= 0.4

        if self._state.step_count > 4:
            self._state.patience = max(0.0, self._state.patience - 0.2)
            reward -= 0.1

        self._state.churn_risk = max(0.0, min(1.0, 1.0 - self._state.trust))
        resolved = False

        if (
            "refund" in text
            and ("process" in text or "initiated" in text)
            and self._state.trust > 0.5
        ):
            resolved = True
            reward += 0.4

        done = (
            resolved
            or self._state.step_count >= self.max_steps
            or self._state.patience <= 0.0
        )


        if resolved:
            next_msg = "Thanks, that helps."
            sentiment = 0.3

        elif self._state.patience <= 0.0:
            next_msg = "I’m done. Escalating this issue."
            sentiment = -1.0
            reward -= 0.3

        else:
            next_msg = "I am still waiting..."
            sentiment = -0.6 if self._state.trust < 0.5 else -0.2

        reward = max(0.0, min(1.0, reward))

        obs = SupportObservation(
            user_message=next_msg,
            sentiment=sentiment,
            resolved=resolved,
            step_count=self._state.step_count
        )

        info = {
            "task": self._state.task_name,
            "step_count": self._state.step_count,
            "patience": self._state.patience,
            "trust": self._state.trust,
            "churn_risk": self._state.churn_risk,
        }

        return obs, reward, done, info

    def state(self) -> SupportState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state