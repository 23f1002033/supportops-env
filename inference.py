from dotenv import load_dotenv
import os
from openai import OpenAI
from env.environment import SupportEnv
from env.models import SupportAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

env = SupportEnv()


def log_start(task):
    print(f"[START] task={task} env=supportops model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def run_task(task_name):
    log_start(task_name)

    obs = env.reset(task=task_name)

    done = False
    step = 0
    rewards = []
    success = False

    try:
        while not done and step < 8:
            step += 1

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a customer support agent. "
                            "Resolve the issue in 1-2 steps maximum. "
                            "Avoid asking unnecessary questions. "
                            "Take action immediately and solve the problem."
                        )
                    },
                    {"role": "user", "content": obs.user_message}
                ],
                max_tokens=100,
                temperature=0.3
            )

            action_text = (response.choices[0].message.content or "").strip()

            if not action_text:
                action_text = "I will resolve your issue immediately."

            action = SupportAction(response=action_text)
            obs, reward, done, _ = env.step(action)

            rewards.append(reward)

            log_step(step, action_text, reward, done)

        success = done

    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)
        success = False

    finally:
        log_end(success, step, rewards)


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)