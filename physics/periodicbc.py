import numpy as np, math
import time
from tenpy.models.spins              import SpinModel
from tenpy.models.model              import NearestNeighborModel
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification   import PurificationTEBD

# beta grid
BETA_GRID = np.arange(0.1, 10.1, 0.5)

# Initial TEBD parameters 
GROUP, DT0, CHI_MAX, SVD_MIN = 2, 0.05, 100, 1e-10


def simulate_periodic_ring(L: int, Jbond: np.ndarray, beta_list: np.ndarray = BETA_GRID) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (Energy[beta],C[beta, L-1]) for one periodic boundary condition Heisenberg ring.
    """
    # construct periodic model with random J_i couplings
    model = SpinModel(dict(
        L=L, bc_x="periodic", bc_MPS="finite", order="folded",
        Jx=Jbond, Jy=Jbond, Jz=Jbond, conserve=None))

    dt = DT0
    chi_max = CHI_MAX
    if L == 128:
        dt = .3
        chi_max = 150

    # folded and physical index maps
    fold2phys = model.lat.order[:, 0]
    phys2fold = np.argsort(fold2phys)

    #  maximally-mixed purification at beta = 0
    psi = PurificationMPS.from_infiniteT(model.lat.mps_sites(),bc=model.lat.bc_MPS)

    # two-site grouping
    model.group_sites(GROUP)
    psi.group_sites(GROUP)

    nn  = NearestNeighborModel(model.lat, model.calc_H_bond_from_MPO())
    eng = PurificationTEBD(psi, nn, dict(
            dt=dt, max_N_sites_per_ring = L // 2, order=2,
            trunc_params=dict(chi_max=chi_max, svd_min=SVD_MIN)))

    # imaginary-time evolution over beta
    E, Cmat, beta_prev = [], [], 0.0
    for beta in beta_list.astype(float):

        eng.run_imaginary((beta - beta_prev) / 2.0)
        
        beta_prev = beta

        # Energy per site 
        E.append(model.H_MPO.expectation_value(psi) / L)

        # correlator for every bond 
        psi_ph = psi.copy()
        psi_ph.group_split(trunc_par=dict(chi_max=CHI_MAX))

        Cbeta = np.empty(L, dtype='f4')
        for i in range(L):
            j       = (i + 1) % L
            fi, fj  = phys2fold[i], phys2fold[j]
            cxx = psi_ph.expectation_value_term([('Sx', fi), ('Sx', fj)])
            cyy = psi_ph.expectation_value_term([('Sy', fi), ('Sy', fj)])
            czz = psi_ph.expectation_value_term([('Sz', fi), ('Sz', fj)])
            Cbeta[i] = (cxx + cyy + czz) / 3.0
            Cbeta[i] *= 4
        Cmat.append(Cbeta)
    # store properly
    return (np.asarray(E,dtype='f4'), np.stack(Cmat).astype('f4'))

