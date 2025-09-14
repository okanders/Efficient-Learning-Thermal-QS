#!/usr/bin/env python3
"""
Local dataset builder: reuse the same J couplings for all betas.
"""

from __future__ import annotations
from pathlib import Path
import argparse, time, warnings, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, h5py

warnings.filterwarnings("ignore", category=UserWarning, module="tenpy.tools.params")

from physics.periodicbc import simulate_periodic_ring, BETA_GRID as BETA_P
from physics.openbc import simulate_open_chain, BETA_GRID as BETA_O
from physics.isingopen import simulate_ising_open_chain, BETA_GRID as BETA_I

REGISTRY = {
    "periodic": (simulate_periodic_ring, BETA_P, lambda L: (L,)),
    "open": (simulate_open_chain,BETA_O, lambda L: (L-1,)),
    "ising_open": (simulate_ising_open_chain, BETA_I, lambda L: (L-1,)),
}

def parse_beta(tokens: list[str], full_grid: np.ndarray) -> np.ndarray:
    if len(tokens) == 1 and tokens[0].lower() == "all":
        return full_grid
    return np.fromstring(",".join(tokens), sep=",", dtype=float)

def _one_sample(sim, L: int, J: np.ndarray, beta_scalar: float):
    """Compute (E,C) for a single beta and J."""
    return sim(L, J, np.asarray([beta_scalar], dtype=float))

def build_db_local(geometry: str, sizes: list[int], N: int, 
                J_max: float, betas: np.ndarray,
                out_root: Path, n_workers: int = 16) -> None:
    sim, _, cshape_fn = REGISTRY[geometry]
    betas = np.asarray(betas, dtype=float)

    # Using a reproducible RNG once per entire run
    rng = np.random.default_rng(0)

    for L in sizes:
        # draw all J couplings for this L once
        # inside build_db_local
        bond_shape = cshape_fn(L) # (L,) or (L-1,)
        j_dim = L if geometry == "ising_open" else bond_shape[0]

        js = rng.uniform(0, J_max, size=(N, j_dim)).astype('f4')

        for beta in betas:
            beta_tag = f"{beta:g}"
            folder = out_root / f"{geometry}_beta" / beta_tag
            folder.mkdir(parents=True, exist_ok=True)
            fname = folder / f"L{L}.h5"

            if fname.exists():
                print(f"{fname} already exists - skip")
                continue

            print(f"beta={beta_tag:<6}  L={L:<4}  N={N}")
            t0 = time.perf_counter()

            with h5py.File(fname, "w") as h5:
                dJ = h5.create_dataset("J", (N, j_dim),dtype='f4')
                dE = h5.create_dataset("Energy", (N, 1),dtype='f4')
                dCor = h5.create_dataset("C", (N, 1) + bond_shape,dtype='f4')
                
                #double check sizing
                assert dJ.shape[1] == j_dim         

                assert dCor.shape[2] == bond_shape[0]


                h5.attrs.update(dict(
                    beta = beta,
                    geometry = geometry,
                    L = L,
                    N  = N,
                    J_max = J_max,
                    created = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
                ))

                # Use the same Js for all betas
                dJ[:] = js

                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = []
                    for idx, J in enumerate(js):
                        fut = pool.submit(_one_sample, sim, L, J, beta)
                        fut.idx = idx
                        futures.append(fut)

                    done = 0
                    for fut in as_completed(futures):
                        i = fut.idx
                        E, C = fut.result()
                        dE[i, 0] = E
                        dCor[i, 0] = C
                        done += 1
                        if done % 20 == 0 or done == N:
                            pct = 100.0 * done / N
                            print(f"\r {done:4d}/{N}  {pct:5.1f}%", end="")

            dt = (time.perf_counter() - t0) / 60.0
            print(f"[done] beta={beta_tag} L={L} {dt:6.2f} min")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", choices=("periodic","open", "ising_open"), required=True)
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--N", type=int, default=400)
    p.add_argument("--J-max", type=float, default=2.0, dest="J_max")
    p.add_argument("--beta", type=str, nargs="+", default=["all"], help='"all"  5 | "2,5,8" | 2 5 8')
    p.add_argument("--out-dir", default="db")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    _, full_grid, _ = REGISTRY[args.geometry]
    betas = parse_beta(args.beta, full_grid)

    mp.set_start_method("spawn", force=True)

    build_db_local(args.geometry,
                    args.sizes,
                    args.N,
                    args.J_max,
                    betas,
                    out_root=Path(args.out_dir),
                    n_workers=args.workers)

