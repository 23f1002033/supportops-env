from env.models import SupportState, GradeResult


def grade(state: SupportState, max_steps: int = 10) -> GradeResult:
    """
    Compute final episode score (0.0 → 1.0) with detailed breakdown.

    Scoring dimensions:
      - Resolution (35%): Did the agent actually solve the problem?
      - Efficiency (20%): How quickly was it resolved?
      - Trust (20%): Final trust level achieved
      - Patience (15%): How much patience remained?
      - Churn Risk (10%): Was churn risk minimized?

    Hard tasks get a bonus multiplier for trust recovery since starting trust is very low.

    Args:
        state: The final hidden state of the environment
        max_steps: Maximum allowed steps for efficiency calculation

    Returns:
        GradeResult with overall score and per-dimension breakdown
    """

    # ─── Resolution Score ───
    if state.resolved:
        resolution_score = 1.0
    elif state.trust > 0.6:
        resolution_score = 0.3  # Partial credit: good rapport but no resolution
    else:
        resolution_score = 0.0

    # ─── Efficiency Score ───
    if state.resolved:
        # Reward fewer steps: 1.0 at step 1, decaying linearly
        efficiency_score = max(0.0, 1.0 - (state.step_count - 1) / max(1, max_steps - 1))
    elif state.step_count < max_steps:
        efficiency_score = 0.2  # Some credit for not exhausting steps
    else:
        efficiency_score = 0.0

    # ─── Trust Score ───
    trust_score = state.trust

    # Bonus for trust recovery in hard tasks (started at 0.15)
    if state.difficulty == "hard":
        trust_delta = max(0.0, state.trust - 0.15)  # How much trust was gained
        trust_score = min(1.0, state.trust + trust_delta * 0.5)

    # ─── Patience Score ───
    patience_score = state.patience

    # ─── Churn Score ───
    churn_score = 1.0 - state.churn_risk

    # ─── Weighted Final Score ───
    weights = {
        "resolution": 0.35,
        "efficiency": 0.20,
        "trust": 0.20,
        "patience": 0.15,
        "churn": 0.10,
    }

    final_score = (
        weights["resolution"] * resolution_score
        + weights["efficiency"] * efficiency_score
        + weights["trust"] * trust_score
        + weights["patience"] * patience_score
        + weights["churn"] * churn_score
    )
    # Clamp to (0, 1) strictly as required by validator
    final_score = round(max(0.001, min(0.999, final_score)), 4)

    return GradeResult(
        score=final_score,
        resolution_score=round(resolution_score, 3),
        efficiency_score=round(efficiency_score, 3),
        trust_score=round(trust_score, 3),
        patience_score=round(patience_score, 3),
        churn_score=round(churn_score, 3),
        breakdown={
            "weights": weights,
            "task": state.task_name,
            "difficulty": state.difficulty,
            "resolved": state.resolved,
            "resolution_type": state.resolution_type,
            "steps_taken": state.step_count,
            "final_trust": round(state.trust, 3),
            "final_patience": round(state.patience, 3),
            "final_churn_risk": round(state.churn_risk, 3),
        }
    )