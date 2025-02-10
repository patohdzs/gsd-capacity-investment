import numpy as np
from scipy.stats import norm

# Approximation for a very large value in std. normal CDF
GAUSSCDF1 = 5


def compute_cutoffs(cs_value, cost):
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
            low_eps[i] = -GAUSSCDF1
        else:
            low_eps[i] = (cs_value[i] - cs_value[i + 1]) / ((cost[i] - cost[i + 1]))

        # Assign higher cutoff as the lower cutoff of preceding action
        if i == 0:
            high_eps[i] = GAUSSCDF1
        else:
            high_eps[i] = low_eps[last_l]

        last_l = i
        while low_eps[last_l] > high_eps[last_l]:

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
                low_eps[last_l] = -GAUSSCDF1
            else:
                low_eps[last_l] = (cs_value[last_l] - cs_value[i + 1]) / (
                    (cost[last_l] - cost[i + 1])
                )

    return high_eps, low_eps, zero_prob


def compute_ccps(high_eps, low_eps, zero_prob):
    # Create array for conditional choice probabilities
    ccps = np.zeros_like(zero_prob)

    # Compute CCP's
    cdf_high = norm.cdf(high_eps[~zero_prob])
    cdf_low = norm.cdf(low_eps[~zero_prob])
    ccps[~zero_prob] = np.maximum(cdf_high - cdf_low, 0)
    return ccps


def compute_ex_ante_value(cs_values, costs, high_eps, low_eps, ccps):
    # Compute expected shock between two cutoffs
    mideps = np.zeros_like(ccps)
    mideps[~(ccps == 0)] = (
        norm.pdf(low_eps[~(ccps == 0)]) - norm.pdf(high_eps[~(ccps == 0)])
    ) / ccps[~(ccps == 0)]

    # Compute ex-ante value
    ex_ante_value = np.sum(
        ccps[~(ccps == 0)] * (cs_values[~(ccps == 0)] - mideps * costs[~(ccps == 0)])
    )
    return ex_ante_value
