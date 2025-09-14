#!/usr/bin/env python3
import numpy as np, h5py, math, logging
from pathlib import Path
import matplotlib.pyplot as plt     
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from common.feature_map import FeatureMappedLassoThermal


# config
L_SIZES     = [8,16,32,64,128]
BETA_VAL    = 7
DB_ROOT     = Path("db/open_beta")
DISTANCE    = 4
TARGET_RMSE = 0.55
N_TEST      = 40
N_TRAIN_MAX = 260
REPEATS     = 5  # Set to 1 if no variability needed

RS_GRID     = [5, 10, 20, 40]
GAMMAS_GRID = [.4, .5, .6, .65, .7, .75]
ALPHAS_GRID   = [2**-k for k in (8,7,6,5)]

ENABLE_DIAGS = True  # change diagnostics in load_pool



logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

def load_pool(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Load and augment dataset for chain length L."""
    with h5py.File(DB_ROOT / f"{BETA_VAL}/L{L}.h5", "r") as h5:
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
def run_one_size(L: int) -> tuple[float, float]:
    """Run experiment for one system size L, returning mean/std of N_tr."""
    X, y = load_pool(L)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_TEST, random_state=0
    )

    y_std = float(np.std(y_test)) or 1.0
    logging.info("L=%d  y_test: mean=%.3f  std=%.3f  range=(%.3f, %.3f)",
        L, float(np.mean(y_test)), y_std, float(np.min(y_test)), float(np.max(y_test)))

    omegas = np.random.default_rng(0).standard_normal((max(RS_GRID), 2*DISTANCE+2))
    proto = FeatureMappedLassoThermal(False, L, DISTANCE, ALPHAS_GRID, omegas)

    grid_params = {"R": RS_GRID, "gamma": GAMMAS_GRID}

    counts = []
    for rep in range(REPEATS):
        rng_rep = np.random.default_rng(rep)
        idxs = rng_rep.permutation(len(X_train)) if REPEATS > 1 else np.arange(len(X_train))
        X_seq, y_seq = X_train[idxs], y_train[idxs]
        
        X_tr, y_tr = [], []
        for n_seen, (x, y_val) in enumerate(zip(X_seq, y_seq), start=1):
            X_tr.append(x); y_tr.append(y_val)
            if n_seen < 5:
                continue
            folds = 5
            grid = GridSearchCV(proto, grid_params, cv=folds, scoring='neg_root_mean_squared_error', n_jobs=-1)
            grid.fit(np.vstack(X_tr), np.asarray(y_tr))
            rmse = np.sqrt(((y_test - grid.predict(X_test))**2).mean())


            if rmse <= TARGET_RMSE or n_seen >= N_TRAIN_MAX:
                logging.info("L=%d  rep=%d  N_tr=%d  RMSE=%.3f",
                    L,          
                    rep + 1,   
                    n_seen,     
                    rmse,       
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
    means = np.array([results[L][0] for L in L_arr])
    stds  = np.array([results[L][1] for L in L_arr])

    # save data to csv
    pd.DataFrame({"L": L_arr, "mean_N": means, "std_N": stds}).to_csv(
        f"sample_complexity_heisenberg_openbc_summary_Beta{BETA_VAL}.csv", index=False
    )
    pd.DataFrame.from_dict(reps, orient="index").T.to_csv(
        f"sample_complexity_heisenberg_openbc_reps_Beta{BETA_VAL}.csv", index=False
    )


    # FIGURE
    plt.figure(figsize=(6.5, 4.25))

    # Green scatter of all repetitions
    for L in L_arr:
        plt.scatter([L] * len(reps[L]), reps[L],
                    marker="o", s=18, c="g", alpha=0.7, label="_nolegend_")

    # Blue line through (L, mean_N)
    plt.plot(L_arr, means, "-b", lw=2, marker="o", label="Average behavior")

    # log  fit 
    X = np.column_stack([np.log(L_arr), np.ones_like(L_arr), 1.0 / L_arr])
    (a, b, c), *_ = np.linalg.lstsq(X, means, rcond=None)
    fit_str = rf"Fit: $a\ln n + b + c/n$  ($a\!\approx\!{a:.2f}$)"



    Lgrid = np.linspace(L_arr.min(), L_arr.max(), 400)
    Nfit  = a * np.log(Lgrid) + b + (c / Lgrid)
    plt.plot(Lgrid, Nfit, "-r", lw=2, label=fit_str)

    # Axis formatting to match the style
    plt.xlabel("Chain length $n$")
    plt.ylabel("$N$")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="lower right", framealpha=0.9)

    plt.xticks(L_arr, [str(int(x)) for x in L_arr])
    ymin = max(0, np.floor(min(min(v) for v in reps.values())/10)*10)
    ymax = np.ceil(max(max(v) for v in reps.values())/10)*10
    plt.yticks(np.linspace(ymin, ymax, 5))

    plt.tight_layout()
    plt.savefig(f"figure_heisenberg_openbc_logfit_Beta{BETA_VAL}.png", dpi=300)




