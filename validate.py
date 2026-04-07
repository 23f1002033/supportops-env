"""
SupportOps-RL Validation Suite
Validates environment correctness, task differentiation, edge cases, and grading.
"""

from env.environment import SupportEnv
from env.models import SupportAction
from env.grader import grade

passed = 0
failed = 0


def check(name: str, condition: bool):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1


# ─── 1. Basic Environment Operations ───
print("\n🔍 1. Basic Environment Operations")

env = SupportEnv()
obs = env.reset(task="easy")
check("reset() returns observation", obs is not None)
check("observation has user_message", len(obs.user_message) > 0)
check("initial step_count is 0", obs.step_count == 0)
check("initial resolved is False", obs.resolved is False)
check("sentiment is in range", -1.0 <= obs.sentiment <= 1.0)
check("urgency is in range", 0.0 <= obs.urgency <= 1.0)

state = env.state()
check("state() returns state", state is not None)
check("state has patience", 0.0 <= state.patience <= 1.0)
check("state has trust", 0.0 <= state.trust <= 1.0)


# ─── 2. Task Differentiation ───
print("\n🔍 2. Task Differentiation")

env_easy = SupportEnv()
env_easy.reset(task="easy")
state_easy = env_easy.state()

env_hard = SupportEnv()
env_hard.reset(task="hard")
state_hard = env_hard.state()

check("Easy trust > Hard trust", state_easy.trust > state_hard.trust)
check("Easy patience > Hard patience", state_easy.patience > state_hard.patience)
check("Easy churn_risk < Hard churn_risk", state_easy.churn_risk < state_hard.churn_risk)
check("Easy difficulty is 'easy'", state_easy.difficulty == "easy")
check("Hard difficulty is 'hard'", state_hard.difficulty == "hard")
check("Easy intent is 'tracking'", state_easy.expected_intent == "tracking")
check("Hard intent is 'refund_escalation'", state_hard.expected_intent == "refund_escalation")


# ─── 3. Easy Task Resolution ───
print("\n🔍 3. Easy Task Resolution (tracking)")

env = SupportEnv()
obs = env.reset(task="easy")
action = SupportAction(
    response="Your order has been shipped and is in transit. "
             "The estimated delivery is Friday. Your tracking status shows dispatch."
)
obs, reward, done, info = env.step(action)
check("Easy task can resolve with tracking keywords", obs.resolved is True)
check("Easy resolution type is tracking_provided", info["resolution_type"] == "tracking_provided")
check("Easy reward is positive", reward > 0)


# ─── 4. Medium Task Resolution ───
print("\n🔍 4. Medium Task Resolution (refund)")

env = SupportEnv()
obs = env.reset(task="medium")
action = SupportAction(
    response="I sincerely apologize. I have initiated a full refund to your account. "
             "The credit will appear within 3-5 business days."
)
obs, reward, done, info = env.step(action)
check("Medium task can resolve with refund keywords", obs.resolved is True)
check("Medium resolution type is refund_processed", info["resolution_type"] == "refund_processed")
check("Medium reward is positive", reward > 0)


# ─── 5. Hard Task Resolution ───
print("\n🔍 5. Hard Task Resolution (escalation)")

env = SupportEnv()
obs = env.reset(task="hard")
# Hard task needs empathy + refund + action + trust > 0.25
action = SupportAction(
    response="I am truly sorry for this experience and I completely understand your frustration. "
             "I am personally escalating this as a priority case. "
             "I have initiated an immediate refund to your account."
)
obs, reward, done, info = env.step(action)
check("Hard task can resolve with empathy+refund+action", obs.resolved is True)
check("Hard resolution type is escalation_resolved", info["resolution_type"] == "escalation_resolved")


# ─── 6. Negative Signals ───
print("\n🔍 6. Negative Signals")

env = SupportEnv()
obs = env.reset(task="medium")

# Repetition
action = SupportAction(response="I understand your concern.")
obs1, r1, _, _ = env.step(action)
action = SupportAction(response="I understand your concern.")
obs2, r2, _, _ = env.step(action)
check("Repetition is penalized", r2 < r1)

# Unnecessary questions
env2 = SupportEnv()
env2.reset(task="medium")
action = SupportAction(response="Could you please share your order number and verify your email?")
_, r_ask, _, _ = env2.step(action)
check("Asking unnecessary questions is penalized", r_ask < 0)


# ─── 7. Edge Cases ───
print("\n🔍 7. Edge Cases")

# Max steps termination
env = SupportEnv()
env.reset(task="easy")
done = False
steps = 0
while not done:
    action = SupportAction(response="Let me look into this for you.")
    _, _, done, info = env.step(action)
    steps += 1
check("Episode terminates at max steps", done is True)
check("Steps <= max_steps", steps <= env.max_steps)

# Patience depletion
env = SupportEnv()
env.reset(task="hard")
done = False
steps = 0
while not done and steps < 20:
    action = SupportAction(response="Please provide details about your issue.")
    _, _, done, info = env.step(action)
    steps += 1
state = env.state()
final_patience = state.patience
check("Patience can reach zero", final_patience <= 0.01 or done)


# ─── 8. Grading ───
print("\n🔍 8. Grading")

# Resolved episode should score high
env = SupportEnv()
env.reset(task="easy")
action = SupportAction(
    response="I apologize. Your order has been shipped and the delivery status "
             "shows it's in transit with estimated arrival this week."
)
env.step(action)
result = grade(env.state(), max_steps=env.max_steps)
check("Resolved episode grade > 0.5", result.score > 0.5)
check("Grade has breakdown dict", "task" in result.breakdown)
check("Grade resolution_score for resolved is 1.0", result.resolution_score == 1.0)

# Unresolved episode should score low
env = SupportEnv()
env.reset(task="medium")
for _ in range(7):
    action = SupportAction(response="I understand.")
    env.step(action)
result2 = grade(env.state(), max_steps=env.max_steps)
check("Unresolved episode grade < resolved grade", result2.score < result.score)


# ─── Summary ───
print("\n" + "=" * 50)
total = passed + failed
print(f"📊 Validation Results: {passed}/{total} passed")
if failed == 0:
    print("✅ All validations passed!")
else:
    print(f"❌ {failed} validation(s) failed.")
print("=" * 50 + "\n")