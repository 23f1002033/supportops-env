from env.environment import SupportEnv
from env.models import SupportAction
from env.grader import grade
env = SupportEnv(max_steps=6)

obs = env.reset(task="hard")
print("Initial observation:", obs)

done = False
step = 0

while not done:
    step += 1
    if step == 1:
        action = SupportAction(response="I am sorry for the issue. I will process your refund now.")
    else:
        action = SupportAction(response="Your refund is initiated and I will assist you further.")

    obs, reward, done, info = env.step(action)
    print(f"Step {step} | reward={reward:.2f} | done={done} | obs={obs}")
    print("Info:", info)

print("\nFinal hidden state:", env.state())
print("\nFinal Score:", grade(env.state()))