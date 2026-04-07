from dotenv import load_dotenv
import os
import sys
import time
from openai import OpenAI
from env.environment import SupportEnv
from env.models import SupportAction
from env.grader import grade

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required. Set it in .env file.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ─── Task-Specific System Prompts ───

SYSTEM_PROMPTS = {
    "easy": (
        "You are a helpful customer support agent for an e-commerce company. "
        "A customer is asking about their order status. "
        "Provide specific tracking information, estimated delivery dates, and shipping status. "
        "Be concise and provide actionable information immediately. "
        "Do NOT ask unnecessary questions — act on the information you have."
    ),
    "medium": (
        "You are a customer support agent handling a refund request. "
        "The customer received a damaged product and wants a refund. "
        "Acknowledge the damage, apologize sincerely, and process the refund immediately. "
        "Be empathetic but efficient. Do NOT ask the customer to verify details unnecessarily. "
        "Take action: initiate the refund and confirm it to the customer."
    ),
    "hard": (
        "You are a senior customer support agent handling an angry, frustrated customer. "
        "This customer has had repeated bad experiences and is threatening to leave and complain publicly. "
        "Your top priority is emotional de-escalation: acknowledge their feelings, apologize sincerely, "
        "and take immediate concrete action (process refund, offer priority handling, escalate if needed). "
        "Do NOT use scripted or generic responses. Be genuine, empathetic, and action-oriented. "
        "Show the customer they are valued and that you personally care about resolving this."
    ),
}


# ─── Structured Logging (EXACT format required by evaluation) ───

def log_start(task: str) -> None:
    """[START] line — exact format required by evaluation."""
    print(f"[START] task={task} env=supportops model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    """[STEP] line — exact format required by evaluation."""
    error_str = "null" if error is None else str(error)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list) -> None:
    """[END] line — exact format required by evaluation."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def call_llm(messages: list, max_retries: int = 3) -> str:
    """Call the LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=150,
                temperature=0.3,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise
    return "I will resolve your issue immediately."


def run_task(task_name: str) -> dict:
    """
    Run a single task episode and return results.
    """
    env = SupportEnv()
    log_start(task_name)

    obs = env.reset(task=task_name)

    # Build conversation history for context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS.get(task_name, SYSTEM_PROMPTS["easy"])},
        {"role": "user", "content": obs.user_message},
    ]

    done = False
    step = 0
    rewards = []
    success = False

    try:
        while not done and step < env.max_steps:
            step += 1

            # Call LLM with full conversation history
            action_text = call_llm(messages)
            if len(action_text) > 500:
                action_text = action_text[:497] + "..."

            # Add agent response to conversation history
            messages.append({"role": "assistant", "content": action_text})

            action = SupportAction(response=action_text)
            obs, reward, done, info = env.step(action)

            rewards.append(reward)
            log_step(step, action_text, reward, done)

            # Add user follow-up to conversation history (if not done)
            if not done:
                context_msg = (
                    f"{obs.user_message}\n\n"
                    f"[System note: Customer sentiment={obs.sentiment:+.2f}, "
                    f"urgency={obs.urgency:.2f}, step {obs.step_count}/{env.max_steps}]"
                )
                messages.append({"role": "user", "content": context_msg})

        success = obs.resolved if done else False

    except Exception as e:
        log_step(step, "ERROR", 0.0, True, error=str(e))
        success = False

    log_end(success, step, rewards)

    return {
        "task": task_name,
        "success": success,
        "steps": step,
        "rewards": rewards,
    }


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    results = []

    for task in tasks:
        result = run_task(task)
        results.append(result)