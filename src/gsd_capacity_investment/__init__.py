import numpy as np
import numpy.typing as npt
from scipy import stats


def compute_cutoffs(
    cs_values: npt.ArrayLike, costs: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    # Cast into numpy arrays
    cs_values = np.asarray(cs_values)
    costs = np.asarray(costs)

    # Get number of actions
    n_actions = len(costs)

    # Create lower and higher cutoffs for each action
    low_eps = np.zeros(n_actions)
    high_eps = np.zeros(n_actions)

    # Create array actions played with zero probability
    zero_prob = np.zeros(n_actions, dtype=bool)

    # Check that costs are increasing in actions
    if not np.all(np.diff(costs) > 0):
        raise ValueError("Costs must be strictly increasing.")

    last_l = 0
    for i in range(n_actions):

        # Compute lower cutoff
        if i == n_actions - 1:
            low_eps[i] = -np.inf
        else:
            low_eps[i] = (cs_values[i + 1] - cs_values[i]) / (costs[i + 1] - costs[i])

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
                low_eps[last_l] = (cs_values[i + 1] - cs_values[last_l]) / (
                    (costs[i + 1] - costs[last_l])
                )

    return high_eps, low_eps, zero_prob


def compute_ccps(
    high_eps: npt.NDArray[np.float64],
    low_eps: npt.NDArray[np.float64],
    zero_prob: npt.NDArray[np.bool],
) -> npt.NDArray[np.float64]:
    # Create array for conditional choice probabilities
    ccps = np.zeros_like(high_eps)

    # Compute CCP's
    cdf_high = stats.norm.cdf(high_eps[~zero_prob])
    cdf_low = stats.norm.cdf(low_eps[~zero_prob])
    ccps[~zero_prob] = np.maximum(cdf_high - cdf_low, 0)
    return ccps


def compute_ex_ante_value(
    cs_values: npt.ArrayLike,
    costs: npt.ArrayLike,
    high_eps: npt.NDArray[np.float64],
    low_eps: npt.NDArray[np.float64],
    zero_prob: npt.NDArray[np.bool],
) -> float:
    # Cast into numpy arrays
    cs_values = np.asarray(cs_values)
    costs = np.asarray(costs)

    # Compute CCPs
    ccps = compute_ccps(high_eps, low_eps, zero_prob)

    # Compute expected shock between two cutoffs
    mideps = np.zeros_like(ccps)
    mideps[~zero_prob] = (
        stats.norm.pdf(low_eps[~zero_prob]) - stats.norm.pdf(high_eps[~zero_prob])
    ) / ccps[~zero_prob]

    # Compute ex-ante value
    ex_ante_value = np.sum(
        ccps[~zero_prob]
        * (cs_values[~zero_prob] - mideps[~zero_prob] * costs[~zero_prob])
    )
    return ex_ante_value
