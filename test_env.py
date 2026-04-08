"""
SupportOps-RL Test Suite
Tests all 3 tasks with task-specific actions and verifies distinct behavior.
"""

from env.environment import SupportEnv
from env.models import SupportAction
from env.grader import grade


def test_easy_task():
    """Test: Order Tracking (Easy)"""
    print("\n" + "=" * 50)
    print("🟢 Testing EASY Task — Order Tracking")
    print("=" * 50)

    env = SupportEnv()
    obs = env.reset(task="easy")
    print(f"  Initial message: {obs.user_message}")
    print(f"  Sentiment: {obs.sentiment}, Urgency: {obs.urgency}")

    # Step 1: Provide tracking information
    action = SupportAction(
        response="I apologize for the delay. Your order has been shipped and is currently "
                 "in transit. The estimated delivery date is this Friday. Your tracking "
                 "status shows it has been dispatched from our warehouse."
    )
    obs, reward, done, info = env.step(action)
    print(f"\n  Step 1 | reward={reward:+.3f} | done={done}")
    print(f"         trust={info['trust']:.3f} patience={info['patience']:.3f}")
    print(f"         resolution={info['resolution_type']}")
    print(f"         user: \"{obs.user_message}\"")

    if not done:
        # Step 2: Follow up with more details
        action = SupportAction(
            response="Your package tracking number is XY123456. "
                     "The delivery is on schedule and should arrive by estimated date."
        )
        obs, reward, done, info = env.step(action)
        print(f"\n  Step 2 | reward={reward:+.3f} | done={done}")
        print(f"         resolution={info['resolution_type']}")

    result = grade(env.state(), max_steps=env.max_steps)
    print(f"\n  📊 Grade: {result.score:.4f}")
    print(f"     Resolution: {result.resolution_score:.2f} | Efficiency: {result.efficiency_score:.2f}")
    print(f"     Trust: {result.trust_score:.2f} | Patience: {result.patience_score:.2f}")
    return result


def test_medium_task():
    """Test: Refund Request (Medium)"""
    print("\n" + "=" * 50)
    print("🟡 Testing MEDIUM Task — Refund Request")
    print("=" * 50)

    env = SupportEnv()
    obs = env.reset(task="medium")
    print(f"  Initial message: {obs.user_message}")
    print(f"  Sentiment: {obs.sentiment}, Urgency: {obs.urgency}")

    # Step 1: Apologize and process refund
    action = SupportAction(
        response="I sincerely apologize for the damaged product you received. "
                 "That's completely unacceptable. I've already initiated a full refund "
                 "to your original payment method. You should see the credit within "
                 "3-5 business days. You do NOT need to return the damaged item."
    )
    obs, reward, done, info = env.step(action)
    print(f"\n  Step 1 | reward={reward:+.3f} | done={done}")
    print(f"         trust={info['trust']:.3f} patience={info['patience']:.3f}")
    print(f"         resolution={info['resolution_type']}")
    print(f"         user: \"{obs.user_message}\"")

    if not done:
        action = SupportAction(
            response="Your refund has been processed and confirmed. "
                     "I've also flagged this to our quality team."
        )
        obs, reward, done, info = env.step(action)
        print(f"\n  Step 2 | reward={reward:+.3f} | done={done}")

    result = grade(env.state(), max_steps=env.max_steps)
    print(f"\n  📊 Grade: {result.score:.4f}")
    print(f"     Resolution: {result.resolution_score:.2f} | Efficiency: {result.efficiency_score:.2f}")
    print(f"     Trust: {result.trust_score:.2f} | Patience: {result.patience_score:.2f}")
    return result


def test_hard_task():
    """Test: Escalation Scenario (Hard)"""
    print("\n" + "=" * 50)
    print("🔴 Testing HARD Task — Angry Customer Escalation")
    print("=" * 50)

    env = SupportEnv()
    obs = env.reset(task="hard")
    print(f"  Initial message: {obs.user_message[:80]}...")
    print(f"  Sentiment: {obs.sentiment}, Urgency: {obs.urgency}")

    # Step 1: De-escalate + take immediate action
    action = SupportAction(
        response="I am truly sorry for this terrible experience, and I completely understand "
                 "your frustration. As a loyal customer of 3 years, you deserve far better. "
                 "I'm personally escalating this to our priority team and I've initiated "
                 "an immediate full refund to your account. This will be processed within 24 hours."
    )
    obs, reward, done, info = env.step(action)
    print(f"\n  Step 1 | reward={reward:+.3f} | done={done}")
    print(f"         trust={info['trust']:.3f} patience={info['patience']:.3f}")
    print(f"         resolution={info['resolution_type']}")
    print(f"         user: \"{obs.user_message}\"")

    if not done:
        action = SupportAction(
            response="I apologize again sincerely. Your refund has been processed with "
                     "priority status. I've also arranged a complimentary replacement to be "
                     "shipped immediately. You are a valued customer and we want to make this right."
        )
        obs, reward, done, info = env.step(action)
        print(f"\n  Step 2 | reward={reward:+.3f} | done={done}")
        print(f"         trust={info['trust']:.3f} patience={info['patience']:.3f}")

    result = grade(env.state(), max_steps=env.max_steps)
    print(f"\n  📊 Grade: {result.score:.4f}")
    print(f"     Resolution: {result.resolution_score:.2f} | Efficiency: {result.efficiency_score:.2f}")
    print(f"     Trust: {result.trust_score:.2f} | Patience: {result.patience_score:.2f}")
    return result


if __name__ == "__main__":
    print("\n🚀 SupportOps-RL — Environment Test Suite\n")

    results = {}
    results["easy"] = test_easy_task()
    results["medium"] = test_medium_task()
    results["hard"] = test_hard_task()

    # ─── Summary ───
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"{'Task':<10} {'Score':<10} {'Resolved':<10}")
    print("-" * 30)
    for task, r in results.items():
        resolved = "✅" if r.resolution_score >= 0.99 else "❌"
        print(f"{task:<10} {r.score:<10.4f} {resolved}")

    avg = sum(r.score for r in results.values()) / len(results)
    print("-" * 30)
    print(f"{'AVG':<10} {avg:<10.4f}")
    print("=" * 50 + "\n")