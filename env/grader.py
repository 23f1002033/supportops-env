def grade(state):
    """
    Final score (0.0 → 1.0)
    """

    score = 0.0

    score += 0.4 * state.trust

    score += 0.3 * state.patience

    score += 0.3 * (1 - state.churn_risk)

    return max(0.0, min(1.0, score))