#!/usr/bin/env python3
import numpy as np, h5py, math, logging
import matplotlib.pyplot as plt           
import pandas as pd                       
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from common.feature_map import FeatureMappedLassoThermalIsing

# config
L_SIZES       = [8,16,32,64,128]
BETA_VAL      = 3
DB_ROOT       = Path("db/alpha_1_5/ising_open_beta/3") #alpha_1_5
DISTANCE      = 4
TARGET_RMSE   = 0.15
N_TEST        = 40
N_TRAIN_MAX   = 260
REPEATS_DEF   = 5   
REPEATS_SPEC  = {128: 1}  # overrides, for L=128 use 3 repeats or 1 for alpha=1.5
H_EVAL        = math.e
ALPHA         = 1.5   #change for alpha=3
RS_GRID       = [5, 10, 20, 40]
GAMMAS_GRID   = [.4, .5, .6, .65, .7, .75]
ALPHAS_GRID   = [2**-k for k in (8,7,6,5)]
ENABLE_DIAGS  = True  # change for some checks in load_pool



logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

def load_pool(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Load and augment dataset for chain length L."""
    with h5py.File(DB_ROOT / f"L{L}.h5", "r") as h5:
        J = h5["J"][:]; E = h5["Energy"][:,0]
    X = np.hstack([J, np.full((len(J),1), BETA_VAL, dtype=J.dtype)])
    if ENABLE_DIAGS:
        logging.info(f"Loaded {len(E)} energies (min={E.min():.3f}, max={E.max():.3f}, mean={E.mean():.3f})")
        q1, q3 = np.percentile(E, [25, 75])
        iqr = q3 - q1
        n_out = ((E < q1 - 1.5*iqr) | (E > q3 + 1.5*iqr)).sum()
        logging.info(f"Flagged outliers: {n_out} / {len(E)}")
    return X, E

@ignore_warnings(category=ConvergenceWarning)
def run_one_size(L: int) -> tuple[float, float, list[int]]:
    """Run experiment for one system size L, returning mean/std of N_tr."""
    X, y = load_pool(L)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=N_TEST, random_state=0)

    # determine repeats based on size
    repeats = REPEATS_SPEC.get(L, REPEATS_DEF)
    logging.info("Running L=%d with %d repeats", L, repeats)

    # log spread of target
    test_range = y_test.max() - y_test.min()
    test_std   = y_test.std(ddof=0) # population sigma
    logging.info("L=%d  test-set range = %.3f   std = %.3f", L, test_range, test_std)

    omegas = np.random.default_rng(0).standard_normal((max(RS_GRID), 2 * (2 * DISTANCE + 1) + 1))
    proto = FeatureMappedLassoThermalIsing(False, L, DISTANCE, ALPHAS_GRID, omegas, H_EVAL)
    grid_params = {"R": RS_GRID, "gamma": GAMMAS_GRID}

    counts = []
    for rep in range(repeats):
        rng_rep = np.random.default_rng(rep)
        idxs = rng_rep.permutation(len(X_train)) if repeats > 1 else np.arange(len(X_train))
        X_seq, y_seq = X_train[idxs], y_train[idxs]

        X_tr, y_tr = [], []
        for n_seen, (x, y_val) in enumerate(zip(X_seq, y_seq), start=1):
            X_tr.append(x); y_tr.append(y_val)
            if n_seen < 5:
                continue
            grid = GridSearchCV(proto, grid_params, cv=5,scoring='neg_root_mean_squared_error', n_jobs=-1)
            grid.fit(np.vstack(X_tr), np.asarray(y_tr))
            rmse = np.sqrt(((y_test - grid.predict(X_test))**2).mean())
            if rmse <= TARGET_RMSE or n_seen >= N_TRAIN_MAX:
                logging.info("L=%d  rep=%d  N_tr=%d  RMSE=%.3f",
                    L, rep + 1, n_seen, rmse
                )
                counts.append(n_seen)
                if rmse > TARGET_RMSE:
                    logging.warning(f"Rep {rep+1} exhausted budget without meeting RMSE")
                break
    return np.mean(counts), np.std(counts), counts

if __name__ == "__main__":
    results, reps = {}, {}
    for L in L_SIZES:
        mu, sigma, rep_counts = run_one_size(L)
        logging.info("L=%d  mean_N_tr=%.1f  std=%.1f", L, mu, sigma)
        results[L] = (mu, sigma)
        reps[L] = rep_counts


    # organize data
    L_arr = np.array(sorted(results.keys()))
    means = np.array([results[L][0] for L in L_arr], dtype=float)
    stds  = np.array([results[L][1] for L in L_arr], dtype=float)

    # save data to csv
    pd.DataFrame({"L": L_arr, "mean_N": means, "std_N": stds}).to_csv(
        f"sample_complexity_ising_summary_Alpha{ALPHA:.1f}_Beta{BETA_VAL}.csv", index=False
    )
    pd.DataFrame.from_dict(reps, orient="index").T.to_csv(
        f"sample_complexity_ising_reps_Alpha{ALPHA:.1f}_Beta{BETA_VAL}.csv", index=False
    )


    # FIGURE
    plt.figure(figsize=(6.5, 4.25))

    # Green scatter of all repetitions
    for L in L_arr:
        plt.scatter([L] * len(reps[L]), reps[L],marker="o", s=18, c="g", alpha=0.7, label="_nolegend_")

    # Blue line through (L, mean_N)
    plt.plot(L_arr, means, "-b", lw=2, marker="o", label="Average behavior")
    plt.errorbar(L_arr, means, yerr=stds, fmt="none", elinewidth=1.2, capsize=3,alpha=0.7, color="b", label="_nolegend_")

    # Choose fit you want
    # "power"  "log+inv"  "log"  "linear+inv"
    # FIT_KIND = "linear+inv"  
    FIT_KIND = "log+inv"   

    if FIT_KIND == "power":
        # N ~ c * n^alpha (fit on log-log)
        mask = (L_arr > 0) & (means > 0)
        if mask.sum() < 2:
            raise ValueError("Need at least two positive points for a log-log fit.")
        logL = np.log(L_arr[mask]); logN = np.log(means[mask])
        alpha_hat, logc_hat = np.polyfit(logL, logN, 1)
        c_hat = np.exp(logc_hat)

        xgrid = np.geomspace(L_arr.min(), L_arr.max(), 400)
        yfit  = c_hat * xgrid**alpha_hat
        fit_label = rf"Fit: $c\,n^{{\alpha}}$ ($\alpha\!\approx\!{alpha_hat:.2f}$)"

        plt.xscale("log"); plt.yscale("log")
        plt.plot(xgrid, yfit, "-r", lw=2, label=fit_label)

        # Use actual L values as major ticks on log x-axis
        from matplotlib.ticker import FixedLocator, ScalarFormatter
        ax = plt.gca()
        ax.xaxis.set_major_locator(FixedLocator(L_arr))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(FixedLocator([]))

    elif FIT_KIND == "log+inv":
        # N ~ alog(n) + b + c/n
        X = np.column_stack([np.log(L_arr), np.ones_like(L_arr), 1.0 / L_arr])
        a_hat, b_hat, c_hat = np.linalg.lstsq(X, means, rcond=None)[0]
        xgrid = np.linspace(L_arr.min(), L_arr.max(), 400)
        yfit  = a_hat*np.log(xgrid) + b_hat + c_hat/xgrid
        fit_label = rf"Fit: $a\ln n + b + c/n$ ($a\!\approx\!{a_hat:.2f}$)"
        plt.plot(xgrid, yfit, "-r", lw=2, label=fit_label)

    elif FIT_KIND == "log":
        # N~alog(n) + b
        X = np.column_stack([np.log(L_arr), np.ones_like(L_arr)])
        a_hat, b_hat = np.linalg.lstsq(X, means, rcond=None)[0]
        xgrid = np.linspace(L_arr.min(), L_arr.max(), 400)
        yfit  = a_hat*np.log(xgrid) + b_hat
        fit_label = rf"Fit: $a\ln n + b$ ($a\!\approx\!{a_hat:.2f}$)"
        plt.plot(xgrid, yfit, "-r", lw=2, label=fit_label)

    elif FIT_KIND == "linear+inv":
        # N ~an + b + c/n  (used in alpha <= 2D section)
        X = np.column_stack([L_arr, np.ones_like(L_arr), 1.0 / L_arr])
        a_hat, b_hat, c_hat = np.linalg.lstsq(X, means, rcond=None)[0]
        xgrid = np.linspace(L_arr.min(), L_arr.max(), 400)
        yfit  = a_hat*xgrid + b_hat + c_hat/xgrid
        fit_label = rf"Fit: $a n + b + c/n$ ($a\!\approx\!{a_hat:.2f}$)"
        plt.plot(xgrid, yfit, "-r", lw=2, label=fit_label)



    plt.xlabel("Chain length $n$")
    plt.ylabel("$N$")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="lower right", framealpha=0.9)

    # If not log-log, make the x ticks the actual L values
    if FIT_KIND != "power":
        plt.xticks(L_arr, [str(int(x)) for x in L_arr])

    plt.tight_layout()
    plt.savefig(f"figure_ising_{FIT_KIND}_Alpha{ALPHA:.1f}_Beta{BETA_VAL}.png", dpi=300)

    logging.info("All done. Figure written to figure_ising_%s_Alpha%.1f_Beta%d.png (+ CSVs).", FIT_KIND, ALPHA, BETA_VAL)




