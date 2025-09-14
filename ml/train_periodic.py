#!/usr/bin/env python3
import numpy as np, h5py, logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from common.feature_map import (
    FeatureMappedLassoThermalPeriodic,
    ZOfQubitHeisenberg_periodic,
    PhiOfZ,
)

# configuration
L_SIZES     = [8, 16, 32, 64, 128]
BETA_VAL    = 7.0
DB_ROOT     = Path(f"db/periodic_beta/{BETA_VAL:g}")
DISTANCE    = 4
N_TRAIN     = 40
N_TEST      = 40
REPEATS     = 5

RS_GRID       = [5, 10, 20, 40]
GAMMAS_GRID   = [.4, .5, .6, .65, .7, .75]
ALPHAS_GRID   = [2**-k for k in (8,7,6,5)]



logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

def load_pool(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Load periodic pool for chain length L."""
    f = DB_ROOT / f"L{L}.h5"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found - build the periodic pool first")
    with h5py.File(f, "r") as h5:
        geom = h5.attrs.get("geometry", "periodic")
        if isinstance(geom, bytes): geom = geom.decode()
        if geom != "periodic":
            raise ValueError(f"{f}: geometry={geom}, expected 'periodic'")
        beta_attr = float(h5.attrs.get("beta", BETA_VAL))
        if abs(beta_attr - BETA_VAL) > 1e-6:
            raise ValueError(f"{f}: beta attribute {beta_attr} =! {BETA_VAL}")
        J = h5["J"][:]  # (N, L)
        C = h5["C"][:]  # (N, L) or (N,1,L)
        if C.ndim == 3:
            if C.shape[1] != 1:
                raise ValueError("File contains multiple beta slices")
            C = np.squeeze(C, 1)       # goes to (N, L)
    X = np.hstack([J, np.full((len(J),1), BETA_VAL, dtype=J.dtype)])  # (N, L+1)
    return X, C

def predict_all_edges(model: FeatureMappedLassoThermalPeriodic, X: np.ndarray) -> np.ndarray:
    """Predict all L edges for each row in X using edge-0 weights."""
    L, R = model.n, model.R
    w = model.model.coef_   # (2R,)
    b = model.model.intercept_
    Y = np.empty((len(X), L), dtype=float)
    for m in range(len(X)):
        J, beta = X[m, :-1], X[m, -1]
        for a in range(L):
            z   = ZOfQubitHeisenberg_periodic(J, a, model.distance)
            phi = PhiOfZ(model.distance, R, model.gamma, np.append(z, beta), model.omegas)
            Y[m, a] = float(phi @ w + b)
    return Y

@ignore_warnings(category=ConvergenceWarning)
def run_one_size(L: int) -> tuple[float, float, list[float]]:
    """Train on 40, test on 40, repeat REPEATS times; return mean/std RMSE and per-rep RMSEs."""
    X, Y = load_pool(L)
    if len(X) < (N_TRAIN + N_TEST):
        raise RuntimeError(f"L={L}: need at least {N_TRAIN+N_TEST} samples")

    # Fixed omegas across reps
    omegas = np.random.default_rng(0).standard_normal((max(RS_GRID), (2*DISTANCE + 1) + 1))
    base = FeatureMappedLassoThermalPeriodic(
        periodic=True, n=L, distance=DISTANCE, alphas=ALPHAS_GRID,
        omegas=omegas, R=RS_GRID[0], gamma=GAMMAS_GRID[0]
    )
    grid_params = {"R": RS_GRID, "gamma": GAMMAS_GRID}

    rms_per_rep = []
    for rep in range(REPEATS):
        # Independent 40/40 split per repeat
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X, Y, train_size=N_TRAIN, test_size=N_TEST,
            random_state=rep, shuffle=True
        )
        grid = GridSearchCV(base, grid_params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        # train on edge 0 only
        grid.fit(X_tr, Y_tr[:, 0])  
        Y_hat = predict_all_edges(grid.best_estimator_, X_te)

        # average test RMSE across edges
        rmse_edges = np.sqrt(np.mean((Y_te - Y_hat)**2, axis=0))
        rms = float(rmse_edges.mean())
        rms_per_rep.append(rms)
        logging.info("L=%d rep=%d  avg test RMSE=%.4f", L, rep+1, rms)

    return float(np.mean(rms_per_rep)), float(np.std(rms_per_rep)), rms_per_rep

if __name__ == "__main__":
    results, reps = {}, {}
    for L in L_SIZES:
        mu, sigma, rmse_counts = run_one_size(L)
        logging.info("L=%d  mean_RMSE=%.4f  std=%.4f", L, mu, sigma)
        results[L] = (mu, sigma)
        reps[L] = rmse_counts

    # save CSV
    L_arr = np.array(sorted(results.keys()))
    means = np.array([results[L][0] for L in L_arr], dtype=float)
    stds  = np.array([results[L][1] for L in L_arr], dtype=float)

    pd.DataFrame({"L": L_arr, "mean_RMSE": means, "std_RMSE": stds}).to_csv(
        f"rmse_periodic_summary_Beta{BETA_VAL:g}.csv", index=False
    )
    pd.DataFrame.from_dict(reps, orient="index").T.to_csv(
        f"rmse_periodic_reps_Beta{BETA_VAL:g}.csv", index=False
    )


    # figure
    plt.figure(figsize=(6.5, 4.25))
    for L in L_arr:
        plt.scatter([L] * len(reps[L]), reps[L], marker="o", s=18, c="g", alpha=0.7, label="_nolegend_")
    plt.plot(L_arr, means, "-b", lw=2, marker="o", label="Average behavior")

    plt.xlabel(r"Chain length $n$")
    plt.ylabel("Average RMS error")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.xticks(L_arr, [str(int(x)) for x in L_arr])
    plt.errorbar(L_arr, means, yerr=stds, fmt="none", elinewidth=1.2, capsize=3, alpha=0.7, color="b", label="_nolegend_")

    ymin = max(0, np.floor(min(min(v) for v in reps.values()) * 10) / 10)
    ymax = np.ceil(max(max(v) for v in reps.values()) * 10) / 10
    plt.yticks(np.linspace(ymin, ymax, 5))

    plt.tight_layout()
    plt.savefig(f"figure_periodic_avg_rmse_Beta{BETA_VAL:g}.png", dpi=300)





