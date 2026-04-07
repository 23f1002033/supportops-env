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


def log_start(task: str) -> None:
    print(f"\n{'='*60}")
    print(f"[START] task={task} env=supportops model={MODEL_NAME}")
    print(f"{'='*60}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, info: dict) -> None:
    truncated = action[:80] + "..." if len(action) > 80 else action
    print(
        f"  [STEP] step={step} reward={reward:+.3f} done={str(done).lower()} "
        f"trust={info.get('trust', '?'):.3f} patience={info.get('patience', '?'):.3f} "
        f"churn={info.get('churn_risk', '?'):.3f}",
        flush=True,
    )
    print(f"         action=\"{truncated}\"", flush=True)


def log_end(success: bool, resolved: bool, steps: int, rewards: list, grade_result) -> None:
    rewards_str = ",".join(f"{r:+.3f}" for r in rewards)
    total = sum(rewards)
    print(f"\n  [END] resolved={str(resolved).lower()} steps={steps} total_reward={total:+.3f}")
    print(f"        rewards=[{rewards_str}]")
    print(f"        grade={grade_result.score:.4f} | "
          f"resolution={grade_result.resolution_score:.2f} "
          f"efficiency={grade_result.efficiency_score:.2f} "
          f"trust={grade_result.trust_score:.2f} "
          f"patience={grade_result.patience_score:.2f} "
          f"churn={grade_result.churn_score:.2f}")
    print(f"{'='*60}\n", flush=True)


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
                print(f"  [RETRY] Attempt {attempt+1} failed: {e}. Retrying in {wait}s...", flush=True)
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
    resolved = False

    try:
        while not done and step < env.max_steps:
            step += 1

            # Call LLM with full conversation history
            action_text = call_llm(messages)

            # Add agent response to conversation history
            messages.append({"role": "assistant", "content": action_text})

            action = SupportAction(response=action_text)
            obs, reward, done, info = env.step(action)

            rewards.append(reward)
            resolved = obs.resolved
            log_step(step, action_text, reward, done, info)

            # Add user follow-up to conversation history (if not done)
            if not done:
                # Include observation context for next turn
                context_msg = (
                    f"{obs.user_message}\n\n"
                    f"[System note: Customer sentiment={obs.sentiment:+.2f}, "
                    f"urgency={obs.urgency:.2f}, step {obs.step_count}/{env.max_steps}]"
                )
                messages.append({"role": "user", "content": context_msg})

    except Exception as e:
        print(f"  [ERROR] {str(e)}", flush=True)

    # Grade the episode
    grade_result = grade(env.state(), max_steps=env.max_steps)
    log_end(done, resolved, step, rewards, grade_result)

    return {
        "task": task_name,
        "resolved": resolved,
        "steps": step,
        "total_reward": sum(rewards),
        "grade": grade_result.score,
        "grade_breakdown": grade_result.breakdown,
    }


def print_summary(results: list) -> None:
    """Print a summary table of results across all tasks."""
    print("\n" + "=" * 60)
    print("📊 SUMMARY — SupportOps-RL Evaluation")
    print("=" * 60)
    print(f"{'Task':<10} {'Resolved':<10} {'Steps':<8} {'Reward':<10} {'Grade':<8}")
    print("-" * 46)

    total_grade = 0
    for r in results:
        resolved_str = "✅" if r["resolved"] else "❌"
        print(
            f"{r['task']:<10} {resolved_str:<10} {r['steps']:<8} "
            f"{r['total_reward']:+.3f}     {r['grade']:.4f}"
        )
        total_grade += r["grade"]

    avg_grade = total_grade / len(results) if results else 0
    print("-" * 46)
    print(f"{'AVG':<10} {'':10} {'':8} {'':10} {avg_grade:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    results = []

    for task in tasks:
        result = run_task(task)
        results.append(result)

    print_summary(results)