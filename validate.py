from env.environment import SupportEnv
from env.models import SupportAction

env = SupportEnv()

for task in ["easy", "medium", "hard"]:
    print(f"\n--- Testing {task} ---")

    obs = env.reset(task=task)
    done = False

    while not done:
        action = SupportAction(response="I will process your refund now.")
        obs, reward, done, info = env.step(action)

        print("Reward:", reward, "| Done:", done)

print("\nValidation complete ✅")