"""
Modules contains the functions needed to perform dft parameterisation.
"""

import numpy as np
import statsmodels.api as sm


def bootstrap(
    data, R=10_000, CI=95, onlyCoefs=True, model="noB88X", bootstrap_off=False
):
    """
    E = (S-X + VWN-C) + aDX * (HF-X - LSDA-X) + aX * B88-X + aC * LYP-C
    Model: "original", "noB88X", "noB88X_scX"
    """

    coefs_dct = {}

    for _ in range(R):
        if bootstrap_off:
            bs_df = data.sample(n=data.shape[0], replace=False)
        else:
            bs_df = data.sample(n=data.shape[0], replace=True)
        bs_df["Y"] = bs_df["EREF"] - bs_df["ENTVJ"] - bs_df["S-X"] - bs_df["VWN-C"]
        Y = np.array(bs_df["Y"])

        if model == "original":
            X1 = np.array(bs_df["HF-X"] - bs_df["S-X"])
            X2 = np.array(bs_df["B88-X"])
            X3 = np.array(bs_df["LYP-C"])
            X = np.vstack((X1, X2, X3)).T
            model = sm.OLS(Y, X).fit()
            (aDX, aX, aC) = model.params
            fit_coefs = {"aDX": aDX, "aX": aX, "aC": aC}

        if model == "noB88X":
            X1 = np.array(bs_df["HF-X"] - bs_df["S-X"])
            X2 = np.array(bs_df["LYP-C"])
            X = np.vstack((X1, X2)).T
            model = sm.OLS(Y, X).fit()
            (aDX, aC) = model.params
            fit_coefs = {"aDX": aDX, "aC": aC}

        if model == "noB88X_scX":
            X1 = np.array(bs_df["HF-X"])
            X2 = np.array(bs_df["S-X"])
            X3 = np.array(bs_df["LYP-C"])
            X = np.vstack((X1, X2, X3)).T

            model = sm.OLS(Y, X).fit()
            (aX0, aX1, aC) = model.params
            fit_coefs = {"aX0": aX0, "aX1": aX1, "aC": aC}

        for key, val in fit_coefs.items():
            if key not in list(coefs_dct.keys()):
                coefs_dct[key] = []
                coefs_dct[key].append(val)
            else:
                coefs_dct[key].append(val)

    r = (100 - CI) / 2
    ci_lb = r
    ci_ub = CI + r

    result_dct = {}

    if coefs_dct == True:
        for key, val in coefs_dct.items():
            result_dct[key] = {"value": float(), "ci_low": float(), "ci_up": float()}
            result_dct[key]["value"] = round(np.mean(val), 3)
            result_dct[key]["ci_low"] = round(np.percentile(val, [ci_lb, ci_ub])[0], 3)
            result_dct[key]["ci_up"] = round(np.percentile(val, [ci_lb, ci_ub])[1], 3)

    else:
        for key, val in coefs_dct.items():
            result_dct[key] = round(np.mean(val), 3)

    return result_dct


def error_stats(err, sig=1):
    err = np.array(err)
    stats_dict = {}
    # Mean
    stats_dict["Mean"] = np.round(np.mean(err), sig)
    # Median Absolute Deviation
    stats_dict["MAD"] = np.round(np.median(np.median(err)), sig)
    # Mean Absolute Error
    stats_dict["MAE"] = np.round(np.median(abs(err - np.median(err))), sig)
    # Root Mean Square Deviation
    stats_dict["RMSD"] = np.round(np.sqrt(np.mean(np.power(err, 2))), sig)
    # Standard Deviation
    stats_dict["SD"] = np.round(np.std(err), sig)
    # Minimum Error
    stats_dict["MIN"] = np.round(np.min(abs(err)), sig)
    # Maximum Error
    stats_dict["MAX"] = np.round(np.max(abs(err)), sig)
    # Lower Range Error
    stats_dict["LOW"] = np.round(np.min(err), sig)
    # Upper Range Error
    stats_dict["UP"] = np.round(np.max(err), sig)

    return stats_dict


def test_coef(df, coef_dct):

    lookup = {
        "aDX": df["HF-X"] - df["S-X"],
        "aX": df["B88-X"],
        "aC": df["LYP-C"],
        "aX0": df["HF-X"],
        "aX1": df["S-X"],
    }

    scaled_dft = df["ENTVJ"] + df["S-X"] + df["VWN-C"]

    for coef, value in coef_dct.items():
        scaled_dft += value * lookup[coef]

    error = df["EREF"] - scaled_dft
    error_kcalmol = error / 4.184
    error_dct = error_stats(error_kcalmol, sig=3)
    return error_dct


def test_idx_parameterisation(
    X,
    Y,
    train_idxs,
    test_idxs,
    df=None,
    R=1,
    metric="RMSD",
    model="noB88X",
    bootstrap_off=False,
    *args,
    **kwargs,
):
    """
    Tests the BYLP DFT Parameterisation given a set of testing indices and a set of training indices.
    """
    train_data = df.iloc[train_idxs]
    test_data = df.iloc[test_idxs]
    return test_coef(
        test_data,
        bootstrap(
            data=train_data,
            R=R,
            model=model,
            bootstrap_off=bootstrap_off,
            *args,
            **kwargs,
        ),
    )[metric]


def test_parameterisation(
    train_idxs,
    test_idxs,
    df,
    queries=None,
    seed_size=30,
    R=100,
    metric="RMSD",
    model="noB88X",
    bootstrap_off=True,
):
    "Simple function to repeatedly parameterise the DFT method with the ever growing active learning dataset."
    results = []
    test_df = df.iloc[test_idxs]
    if queries is None:
        queries = len(train_idxs) - seed_size
    for no_queries in range(queries + 1):
        train_data = df.iloc[train_idxs[: seed_size + no_queries]]
        curr_results = test_coef(
            test_df, bootstrap(train_data, R, model=model, bootstrap_off=bootstrap_off)
        )

        results.append(curr_results[metric])
    return results
