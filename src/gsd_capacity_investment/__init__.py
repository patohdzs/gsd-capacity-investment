import numpy as np
from scipy.stats import norm


def compute_ex_ante_value(sigma, cs_value, cost, n_actions):
    # Create lower and higher cutoffs for each action
    low_eps = np.zeros(n_actions)
    high_eps = np.zeros(n_actions)

    # Create array actions played with zero probability
    zero_prob = np.zeros(n_actions, dtype=bool)

    # Create array for conditional choice probabilities
    ccps = np.zeros(n_actions)

    # Approximation for a very large value in std. normal CDF
    GAUSSCDF1 = 5

    # Check that costs are increasing in actions
    for i in range(n_actions):
        if i > 0 and cost[i] <= cost[i - 1]:
            raise ValueError("Investment cost values must be strictly increasing.")

    last_l = 0
    for i in range(n_actions):

        # Compute lower cutoff
        if i == n_actions - 1:
            low_eps[i] = -GAUSSCDF1
        else:
            low_eps[i] = (cs_value[i] - cs_value[i + 1]) / (
                (cost[i] - cost[i + 1]) * sigma
            )

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
                    (cost[last_l] - cost[i + 1]) * sigma
                )

    # Compute ex-ante value and CCP's
    ex_ante_value = 0
    for i in range(n_actions):
        # Skip actions played with zero probability
        if zero_prob[i]:
            continue

        # Compute CCP's
        ccps[i] = norm.cdf(high_eps[i]) - norm.cdf(low_eps[i])
        if ccps[i] < 0 and ccps[i] > -1e-8:
            ccps[i] = 0

        # Compute ex-ante value
        if ccps[i] > 0:
            mideps = (norm.pdf(low_eps[i]) - norm.pdf(high_eps[i])) / ccps[i]
            ex_ante_value += ccps[i] * (cs_value[i] - sigma * mideps * cost[i])

    return ex_ante_value, ccps
