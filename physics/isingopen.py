from __future__ import annotations

import numpy as np
import math
from typing import Tuple

from tenpy.networks.site             import SpinHalfSite
from tenpy.models.lattice            import Chain
from tenpy.networks.mpo              import MPO
from tenpy.models.model              import MPOModel
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification   import PurificationApplyMPO
from tenpy.tools.fit                 import fit_with_sum_of_exp
import warnings

# suppress the specific VariationalCompression warning
warnings.filterwarnings(
    "ignore",
    message="VariationalCompression with min_sweeps=max_sweeps: we recommend to set tol_theta_diff=None to avoid overhead",
    category=UserWarning,
)


__all__ = ["BETA_GRID", "simulate_open_chain"]

# default beta-grid - users may override via the command-line
BETA_GRID = np.arange(0.1, 10.1, 0.5)


DT_INITIAL = 0.2 
CHI_INIT  = 60   
CHI_CAP  = 400   # max. allowed bond dimension
SVD_MIN = 1e-10    # SVD truncation threshold
ALPHA  = 1.5     # power-law exponent of the long-range interactions
H_VAL = math.e  # transverse field strength

def sum_of_exp(lambdas, prefactors, x):
    """Evaluate sum_i prefactors[i] * lambdas[i]**x for different x"""
    return np.real_if_close(np.dot(np.power.outer(lambdas, x).T, prefactors))

def _fit_power_law(alpha: float, L: int, tolerance: float = 1e-8, max_exps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Fit 1/r^alpha on the range [1, L] with a sum of exponentials.
    """
    # define the target decay function
    def decay(x):
        return x**(-alpha)

    fit_range = L+L%2

    # Iterative fitting logic , adapted from RandomIsingChainFromGrid in GS
    n_exp = 0
    MaxError = 1
    MaxErrorLast = 10
    doublecheck = 1
    check = 0
    while MaxError > tolerance and n_exp < max_exps and doublecheck:
        n_exp += 1
        lam, pref = fit_with_sum_of_exp(decay, n_exp, fit_range)
        x = np.arange(1, fit_range + 1)
        y1 = decay(x)
        # helper function
        y2 = sum_of_exp(lam, pref, x)
        y = np.abs(y1 - y2)
        MaxError = y.max()

        if MaxError > MaxErrorLast and n_exp > 10:
            check = 1
            n_exp -= 2

        MaxErrorLast = MaxError
        if check:
            doublecheck = 0

    # if any lam >=1, reduce n_exp
    while np.any(lam >= 1):
        n_exp -= 1
        lam, pref = fit_with_sum_of_exp(decay, n_exp, fit_range)
    return lam, pref


class LongRangeIsingChainOpenBC(MPOModel):
    """
    Long-range transverse field Ising model with open boundary conditions.
    """

    def __init__(self, params: dict):
        L    = params['L']
        J    = np.asarray(params['JParams'])
        h    = np.asarray(params['hParams'])
        lam  = params['lam']
        pref = params['pref']
        # verify parameters
        assert np.all((lam > 0) & (lam < 1)), "lambda must be in (0,1)"
        assert np.all(pref > 0), "pref must be positive"

        site = SpinHalfSite(conserve=None, sort_charge=None)
        lat  = Chain(L, site, bc_MPS="finite", bc="open")
        Id, X, Z = site.Id, site.Sigmax, site.Sigmaz

        # one Ji branch + one constant branch per lambda
        W = 2 * len(lam) + 2
        grids = []
        for i in range(L):
            G = [[None] * W for _ in range(W)]
            # left and right identity legs
            G[0][0]     = Id
            G[W - 1][W - 1] = Id
            # onsite transverse field term
            G[0][W - 1]   = h[i] * X
            # long-range ZZ  through  exponential decomposition
            for k in range(len(lam)):
                # branch with Ji Jj
                G[0][1 + k]    = lam[k] * J[i] * Z
                G[1 + k][1 + k]  = lam[k] * Id
                G[1 + k][W - 1]  = pref[k] * J[i] * Z
                # branch with the constant 1
                off = 1 + len(lam) + k
                G[0][off]    = lam[k] * Z
                G[off][off]  = lam[k] * Id
                G[off][W - 1]  = pref[k] * Z
            # fill remaining None with zeros
            for r in range(W):
                for c in range(W):
                    if G[r][c] is None:
                        G[r][c] = 0 * Id
            grids.append(G)
        H_mpo = MPO.from_grids(lat.mps_sites(), grids, bc="finite", IdL=0, IdR=-1)
        super().__init__(lat, H_mpo)


def _simulate_beta(beta_target: float, psi: PurificationMPS, model: LongRangeIsingChainOpenBC,
                    chi_init: int = CHI_INIT, 
                    chi_cap: int  = CHI_CAP, 
                    svd_min: float = SVD_MIN ) -> None:
    """Evolve the purification up to beta/2 in imaginary time."""
    # setup
    chi_max = chi_init
    dt_fixed = DT_INITIAL # single fixed step size
    dt       = dt_fixed
    rel_tol  = 1e-6

    U = model.H_MPO.make_U_II(-dt)
    opts = dict(
        trunc_params=dict(chi_max=chi_max, svd_min=svd_min),
        compression_method='zip_up',
        max_sweeps=1,
    )
    eng = PurificationApplyMPO(psi, U, opts)
    # evolve to beta/2
    target    = 0.5 * beta_target  
    beta_half = 0.0
    E_prev    = model.H_MPO.expectation_value(psi)
    # counts consecutive small relative energy changes
    streak    = 0      

    # main loop
    while beta_half < target - 1e-12:
        # clamp final step to avoid overshoot
        rem    = target - beta_half
        new_dt = min(dt_fixed, rem)

        # rebuild U / engine only if the step changes
        if abs(new_dt - dt) > 1e-12:
            dt = new_dt
            U  = model.H_MPO.make_U_II(-dt)
            eng = PurificationApplyMPO(psi, U, opts)

        # single imaginary-time step
        eng.init_env(U)
        eng.run()
        beta_half += dt

        # energy / convergence tracking
        E_curr = model.H_MPO.expectation_value(psi)
        rel    = abs(E_curr - E_prev) / (abs(E_curr) + 1e-12)
        streak = streak + 1 if rel < rel_tol else 0

        # adaptive chi
        current_chi = max(psi.chi)
        need_more_bonds = current_chi >= 0.9 * chi_max
        below_cap  = (chi_cap is None) or (chi_max < chi_cap)
        if need_more_bonds and below_cap:
            proposed = int(1.5 * chi_max)
            if chi_cap is not None:
                proposed = min(proposed, chi_cap)
            if proposed > chi_max:
                chi_max = proposed
                opts['trunc_params']['chi_max'] = chi_max
                eng = PurificationApplyMPO(psi, U, opts)

        # early stopping  if were not cap limited
        if streak >= 3 and current_chi < 0.9 * chi_max:
            break

        E_prev = E_curr
        psi.canonical_form()



def simulate_ising_open_chain(L: int, Jbond: np.ndarray, beta_list: np.ndarray = BETA_GRID) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (Energy[beta], C[beta, L-1]) for one long-range Ising chain with open BC.
    """
    # J has length L and pad with zeros if not
    J = np.asarray(Jbond, dtype=float)
    if len(J) < L:
        # pad with zeros at the end to assign zero coupling to the last site
        J = np.concatenate([J, np.zeros(L - len(J))])
    elif len(J) > L:
        # truncate extra couplings
        J = J[:L]

    # compute exponential approximation of 1/r^alpha
    lam, pref = _fit_power_law(ALPHA, L)


    # build the model and initial purification at beta = 0
    params = dict(L=L, JParams=J, hParams=[H_VAL] * L, lam=lam, pref=pref)
    model  = LongRangeIsingChainOpenBC(params)
    psi    = PurificationMPS.from_infiniteT(model.lat.mps_sites(), bc=model.lat.bc_MPS)

    energies    = []
    correlators = []
    beta_prev   = 0.0

    # evolve through the beta grid in ascending order
    for beta in beta_list.astype(float):
        # evolve by the incremental beta
        _simulate_beta(beta - beta_prev, psi, model)
        beta_prev = beta

        # energy per site
        energy = model.H_MPO.expectation_value(psi) / math.sqrt(L)
        energies.append(energy)

        # nearest neighbour Z Z correlators
        corr = np.zeros(L - 1, dtype='f4')
        # copy the state and make sure it is canonical before measuring
        psi_ph = psi.copy()
        psi_ph.canonical_form()
        for i in range(L - 1):
            # expectation of corr on site i and i+1
            czz = psi_ph.expectation_value_term([('Sigmaz', i), ('Sigmaz', i + 1)])
            corr[i] = czz
        correlators.append(corr)

    return (np.asarray(energies, dtype='f4'),np.stack(correlators).astype('f4'))



