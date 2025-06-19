import numpy as np
from scipy.stats import ks_2samp
from typing import Dict

# modified version of SALib's ResultDict for compatibility
# This is a self-contained version of the PAWN analysis without dependencies on SALib.util.
# reference: https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/pawn.html

class ResultDict(dict):
    """Mimics the behavior of SALib's ResultDict, mainly for printing as DataFrame."""
    def to_df(self):
        import pandas as pd
        df_data = {k: v for k, v in self.items() if k != "names" and k != "KS"}  # exclude 2D arrays
        df = pd.DataFrame(df_data, index=self["names"])
        return df


def modified_analyze(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    S: int = 10,
    print_to_console: bool = False,
    seed: int = None,
):
    """Self-contained version of PAWN analysis without SALib.util dependencies."""

    if seed:
        np.random.seed(seed)

    D = problem["num_vars"]
    var_names = problem["names"]
    groups = problem.get("groups", None)

    results = np.full((D, 6), np.nan)
    temp_pawn = np.full((S, D), np.nan)

    step = 1 / S
    for d_i in range(D):
        seq = np.arange(0, 1 + step, step)
        X_di = X[:, d_i]
        X_q = np.nanquantile(X_di, seq)

        for s in range(S):
            Y_sel = Y[(X_di >= X_q[s]) & (X_di < X_q[s + 1])]
            if len(Y_sel) == 0:
                continue

            ks = ks_2samp(Y_sel, Y)
            temp_pawn[s, d_i] = ks.statistic

        p_ind = temp_pawn[:, d_i]
        mins = np.nanmin(p_ind)
        mean = np.nanmean(p_ind)
        med = np.nanmedian(p_ind)
        maxs = np.nanmax(p_ind)
        stdev = np.nanstd(p_ind)
        cv = stdev / mean if mean != 0 else np.nan
        results[d_i, :] = [mins, mean, med, maxs, cv, stdev]

    # If groups are provided
    if groups:
        group_names = list(set(groups))
        group_names.sort()
        n_groups = len(group_names)
        group_results = np.full((n_groups, results.shape[1]), np.nan)

        groups = np.array(groups)
        for i, g in enumerate(group_names):
            group_results[i, :] = np.nanmean(results[groups == g], axis=0)
        results = group_results
        var_names = group_names

    Si = ResultDict({
        "minimum": results[:, 0],
        "mean":    results[:, 1],
        "median":  results[:, 2],
        "maximum": results[:, 3],
        "CV":      results[:, 4],
        "stdev":   results[:, 5],
        "KS":      temp_pawn.T,  # shape: (D, S)
        "names":   var_names
    })

    if print_to_console:
        print(Si.to_df())

    return Si