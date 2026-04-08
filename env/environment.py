from typing import Dict, Tuple, List
from env.models import SupportAction, SupportObservation, SupportState
import json
import os
import random


class SupportEnv:
    """
    SupportOps-RL: A realistic customer support simulation environment.

    Simulates multi-step customer-agent interactions with hidden behavioral
    dynamics (trust, patience, churn risk) and task-specific reward shaping.
    """

    def __init__(self, max_steps: int = 10):
        self.default_max_steps = max_steps
        self.max_steps: int = max_steps
        self._state: SupportState | None = None
        self._current_task: str = "easy"
        self._task_data: dict = {}
        self._follow_up_index: int = 0

    def load_task(self, task_name: str) -> dict:
        """Load task definition from JSON file."""
        path = os.path.join(os.path.dirname(__file__), "tasks", f"{task_name}.json")
        with open(path, "r") as f:
            return json.load(f)

    def reset(self, task: str = "easy") -> SupportObservation:
        """
        Reset the environment for a new episode.

        Args:
            task: Task difficulty — "easy", "medium", or "hard"

        Returns:
            Initial observation for the agent
        """
        self._current_task = task
        self._task_data = self.load_task(task)
        self._follow_up_index = 0

        # Use task-specific max_steps if defined, else fall back to default
        self.max_steps = self._task_data.get("max_steps", self.default_max_steps)

        self._state = SupportState(
            conversation=[],
            step_count=0,
            patience=self._task_data.get("initial_patience", 1.0),
            trust=self._task_data.get("initial_trust", 0.4),
            churn_risk=self._task_data.get("initial_churn_risk", 0.3),
            task_name=self._task_data["task_name"],
            difficulty=self._task_data.get("difficulty", "easy"),
            expected_intent=self._task_data["expected_intent"],
            resolved=False,
            resolution_type=None,
        )

        # Initial sentiment based on difficulty
        initial_sentiment = {
            "easy": -0.2,
            "medium": -0.5,
            "hard": -0.9,
        }.get(self._task_data.get("difficulty", "easy"), -0.5)

        return SupportObservation(
            user_message=self._task_data["initial_message"],
            sentiment=initial_sentiment,
            resolved=False,
            step_count=0,
            urgency=self._compute_urgency(),
        )

    def _compute_urgency(self) -> float:
        """Derive urgency from hidden state (exposed to agent as a signal)."""
        if self._state is None:
            return 0.0
        # Urgency rises as patience drops and churn risk increases
        urgency = (1.0 - self._state.patience) * 0.5 + self._state.churn_risk * 0.5
        return round(max(0.01, min(0.99, urgency)), 3)

    def _compute_reward(self, text: str) -> float:
        """
        Compute reward based on agent response and current task context.
        Returns raw reward (can be negative).
        """
        assert self._state is not None
        reward = 0.0
        intent = self._state.expected_intent
        difficulty = self._state.difficulty

        # ─── Universal Positive Signals ───
        # Empathy / apology
        empathy_words = ["sorry", "apologize", "apologies", "understand your frustration",
                         "completely understand", "sincerely"]
        empathy_count = sum(1 for w in empathy_words if w in text)
        if empathy_count > 0:
            empathy_reward = min(0.25, empathy_count * 0.1)
            reward += empathy_reward
            self._state.trust = min(1.0, self._state.trust + 0.08 * empathy_count)

        # Offering help
        if any(w in text for w in ["help", "assist", "support", "here for you"]):
            reward += 0.05

        # ─── Task-Specific Positive Signals ───
        if intent == "tracking":
            # Easy: reward tracking-related info
            tracking_words = ["track", "status", "shipped", "deliver", "dispatch",
                              "transit", "arrival", "estimated", "order number",
                              "shipping", "package"]
            tracking_hits = sum(1 for w in tracking_words if w in text)
            if tracking_hits > 0:
                reward += min(0.35, tracking_hits * 0.1)
            # Penalize offering refund when user just wants tracking
            if "refund" in text:
                reward -= 0.15

        elif intent == "refund":
            # Medium: reward refund processing
            if "refund" in text:
                reward += 0.15
            if any(w in text for w in ["process", "initiate", "credit", "reimburse"]):
                reward += 0.15
            if any(w in text for w in ["return", "replacement"]):
                reward += 0.05

        elif intent == "refund_escalation":
            # Hard: heavy reward for emotional de-escalation + action
            if empathy_count > 0:
                reward += 0.15  # Extra empathy bonus for hard tasks
            if any(w in text for w in ["priority", "escalate", "manager", "senior",
                                        "immediate", "personally"]):
                reward += 0.15
            if "refund" in text:
                reward += 0.10
            if any(w in text for w in ["process", "initiate", "credit", "reimburse"]):
                reward += 0.10

        # ─── Universal Negative Signals ───
        # Repetition penalty
        if (len(self._state.conversation) >= 2 and
                self._state.conversation[-1] == self._state.conversation[-2]):
            reward -= 0.3

        # Asking for unnecessary details
        unnecessary = ["provide", "could you", "can you send", "please share",
                        "verify your", "confirm your"]
        if any(phrase in text for phrase in unnecessary):
            reward -= 0.2

        # Generic/low-quality response
        generic = ["i understand", "thank you for reaching out", "let me check"]
        generic_matches = sum(1 for g in generic if g in text)
        if generic_matches >= 2 and len(text.split()) < 30:
            reward -= 0.15

        # Efficiency penalty for late steps
        if self._state.step_count > 3:
            reward -= 0.05 * (self._state.step_count - 3)

        return reward

    def _check_resolution(self, text: str) -> bool:
        """
        Check if the response resolves the issue based on task type.
        """
        assert self._state is not None
        intent = self._state.expected_intent
        resolution_keywords = self._task_data.get("resolution_keywords", [])

        # Count how many resolution keywords appear
        keyword_hits = sum(1 for kw in resolution_keywords if kw in text)

        if intent == "tracking":
            # Easy: resolved if agent provides tracking-related info
            # Needs at least 2 relevant keywords + some trust
            if keyword_hits >= 2 and self._state.trust > 0.3:
                self._state.resolution_type = "tracking_provided"
                return True

        elif intent == "refund":
            # Medium: resolved if agent processes refund + shows empathy
            has_refund = any(w in text for w in ["refund", "credit", "reimburse"])
            has_action = any(w in text for w in ["process", "initiate", "approved",
                                                   "completed"])
            if has_refund and has_action and self._state.trust > 0.35:
                self._state.resolution_type = "refund_processed"
                return True

        elif intent == "refund_escalation":
            # Hard: resolved if agent de-escalates + processes refund + trust recovered
            has_empathy = any(w in text for w in ["sorry", "apologize", "apologies",
                                                    "understand"])
            has_refund = any(w in text for w in ["refund", "credit", "reimburse"])
            has_action = any(w in text for w in ["process", "initiate", "priority",
                                                   "escalate", "immediately"])
            if has_empathy and has_refund and has_action and self._state.trust > 0.25:
                self._state.resolution_type = "escalation_resolved"
                return True

        return False

    def _get_user_response(self, resolved: bool) -> Tuple[str, float]:
        """
        Generate dynamic user response based on hidden state.
        Returns (message, sentiment).
        """
        assert self._state is not None

        if resolved:
            # Positive resolution responses based on trust level
            if self._state.trust > 0.7:
                msg = "Thank you so much! That's exactly what I needed. Really appreciate your help."
                sentiment = 0.8
            elif self._state.trust > 0.5:
                msg = "Okay, thanks. That helps."
                sentiment = 0.4
            else:
                msg = "Fine. I hope this actually gets resolved this time."
                sentiment = 0.1
            return msg, sentiment

        if self._state.patience <= 0.0:
            # Patience depleted
            escalation_msgs = [
                "I'm done. I'm escalating this to your manager.",
                "Forget it. I'm filing a formal complaint.",
                "This is the worst support experience I've ever had. I'm leaving.",
            ]
            return random.choice(escalation_msgs), -1.0

        # Mid-conversation: pick from follow-up messages based on state
        follow_ups = self._task_data.get("follow_up_messages", ["I'm still waiting..."])

        if self._follow_up_index < len(follow_ups):
            msg = follow_ups[self._follow_up_index]
            self._follow_up_index += 1
        else:
            # Fallback messages based on patience
            if self._state.patience < 0.3:
                msg = "I really need this resolved NOW."
            elif self._state.patience < 0.6:
                msg = "Can you please hurry? I've been waiting."
            else:
                msg = "I'm still waiting for an update."

        # Sentiment tracks the hidden trust/patience blend
        sentiment = round(
            (self._state.trust * 0.6 + self._state.patience * 0.4) - 0.5,
            2
        )
        sentiment = max(-0.99, min(0.99, sentiment))

        return msg, sentiment

    def _update_hidden_state(self, text: str) -> None:
        """Update trust, patience, and churn risk based on agent response."""
        assert self._state is not None

        decay_rate = self._task_data.get("patience_decay_rate", 0.15)

        # Patience decays each step, faster for harder tasks
        base_decay = decay_rate * (1.0 + 0.1 * self._state.step_count)
        self._state.patience = max(0.01, self._state.patience - base_decay)

        # Good responses slow patience decay (partially recover)
        if any(w in text for w in ["sorry", "apologize", "understand"]):
            self._state.patience = min(0.99, self._state.patience + 0.05)

        # Action words boost trust
        if any(w in text for w in ["process", "initiate", "approved", "immediately"]):
            self._state.trust = min(0.99, self._state.trust + 0.1)

        # Churn risk is inverse-correlated with trust and patience
        self._state.churn_risk = round(
            max(0.01, min(0.99, 1.0 - (self._state.trust * 0.6 + self._state.patience * 0.4))),
            3
        )

    def step(self, action: SupportAction) -> Tuple[SupportObservation, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Agent's response action

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        self._state.conversation.append(action.response)

        text = action.response.lower()

        # 1. Compute reward
        reward = self._compute_reward(text)

        # 2. Check resolution
        resolved = self._check_resolution(text)

        # 3. Update hidden state dynamics
        self._update_hidden_state(text)

        # 4. Resolution bonus
        if resolved:
            self._state.resolved = True
            efficiency_bonus = max(0.0, 0.3 * (1.0 - self._state.step_count / self.max_steps))
            reward += 0.4 + efficiency_bonus

        # 5. Determine if episode is done
        done = (
            resolved
            or self._state.step_count >= self.max_steps
            or self._state.patience <= 0.0
        )

        # 6. Patience depletion penalty
        if not resolved and self._state.patience <= 0.0:
            reward -= 0.3

        # 7. Clamp reward strictly to (0.0, 1.0) to pass phase 2 validation
        reward = round(max(0.01, min(0.99, reward)), 3)

        # 8. Generate user response
        next_msg, sentiment = self._get_user_response(resolved)

        obs = SupportObservation(
            user_message=next_msg,
            sentiment=round(sentiment, 2),
            resolved=resolved,
            step_count=self._state.step_count,
            urgency=self._compute_urgency(),
        )

        info = {
            "task": self._state.task_name,
            "difficulty": self._state.difficulty,
            "step_count": self._state.step_count,
            "max_steps": self.max_steps,
            "patience": round(self._state.patience, 3),
            "trust": round(self._state.trust, 3),
            "churn_risk": round(self._state.churn_risk, 3),
            "resolution_type": self._state.resolution_type,
        }

        return obs, reward, done, info

    def state(self) -> SupportState:
        """Return the current hidden state (for debugging/grading)."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state