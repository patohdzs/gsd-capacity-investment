import numpy as np
from scipy.stats import rv_continuous, norm


def compute_cutoffs(cs_value: np.ndarray, cost: np.ndarray) -> tuple[np.ndarray]:
    # Get number of actions
    n_actions = len(cost)

    # Create lower and higher cutoffs for each action
    low_eps = np.zeros(n_actions)
    high_eps = np.zeros(n_actions)

    # Create array actions played with zero probability
    zero_prob = np.zeros(n_actions, dtype=bool)

    # Check that costs are increasing in actions
    if not np.all(np.diff(cost) > 0):
        raise ValueError("Costs must be strictly increasing.")

    last_l = 0
    for i in range(n_actions):

        # Compute lower cutoff
        if i == n_actions - 1:
            low_eps[i] = -np.inf
        else:
            low_eps[i] = (cs_value[i + 1] - cs_value[i]) / (cost[i + 1] - cost[i])

        # Assign higher cutoff as the lower cutoff of preceding action
        if i == 0:
            high_eps[i] = np.inf
        else:
            high_eps[i] = low_eps[last_l]

        last_l = i
        while low_eps[last_l] >= high_eps[last_l]:

            # Mark as zero probability
            zero_prob[last_l] = True
            if last_l == 0:
                break

            # Check preceding cutoff
            last_l -= 1
            if zero_prob[last_l]:
                continue

            # Recompute lower cutoff
            if i == n_actions - 1:
                low_eps[last_l] = -np.inf
            else:
                low_eps[last_l] = (cs_value[i + 1] - cs_value[last_l]) / (
                    (cost[i + 1] - cost[last_l])
                )

    return high_eps, low_eps, zero_prob


def compute_ccps(
    high_eps: np.ndarray,
    low_eps: np.ndarray,
    zero_prob: np.ndarray,
    eps_dist: rv_continuous = norm,
) -> np.ndarray:
    # Create array for conditional choice probabilities
    ccps = np.zeros_like(high_eps)

    # Compute CCP's
    cdf_high = eps_dist.cdf(high_eps[~zero_prob])
    cdf_low = eps_dist.cdf(low_eps[~zero_prob])
    ccps[~zero_prob] = np.maximum(cdf_high - cdf_low, 0)
    return ccps


def compute_ex_ante_value(
    cs_values: np.ndarray,
    costs: np.ndarray,
    high_eps: np.ndarray,
    low_eps: np.ndarray,
    zero_prob: np.ndarray,
    eps_dist: rv_continuous = norm,
) -> float:
    # Compute CCPs
    ccps = compute_ccps(high_eps, low_eps, zero_prob, eps_dist)

    # Compute expected shock between two cutoffs
    mideps = np.zeros_like(ccps)
    mideps[~zero_prob] = (
        eps_dist.pdf(low_eps[~zero_prob]) - eps_dist.pdf(high_eps[~zero_prob])
    ) / ccps[~zero_prob]

    # Compute ex-ante value
    ex_ante_value = np.sum(
        ccps[~zero_prob]
        * (cs_values[~zero_prob] - mideps[~zero_prob] * costs[~zero_prob])
    )
    return ex_ante_value
