# physics/openbc.py
import numpy as np, math
from tenpy.networks.site             import SpinHalfSite
from tenpy.models.lattice            import Chain
from tenpy.networks.mpo              import MPO
from tenpy.models.model              import MPOModel, NearestNeighborModel
from tenpy.networks.purification_mps import PurificationMPS
from tenpy.algorithms.purification   import PurificationTEBD

BETA_GRID = np.arange(0.1, 10.1, 0.5)

class HeisenbergChainOpenBC(MPOModel):
    """Exact copy of original open boundary MPO constructor."""
    def __init__(self, L, J):
        site = SpinHalfSite(conserve=None)
        lat  = Chain(L, site, bc_MPS="finite", bc="open")
        Sx, Sy, Sz, I = site.Sigmax, site.Sigmay, site.Sigmaz, site.Id
        grids, W = [], 5
        for i in range(L):
            G = [[None]*W for _ in range(W)]
            G[0][0] = G[-1][-1] = I
            G[0][-1] = 0*I
            if i < L-1:
                for col, S in zip((1,2,3), (Sx,Sy,Sz)):
                    G[0][col] = J[i]*S
                    G[col][-1] = S
            grids.append(G)
        H = MPO.from_grids(lat.mps_sites(), grids, bc='finite', IdL=0, IdR=-1)
        super().__init__(lat, H)
        self.explicit_plus_hc = False


# physics/openbc.py
def simulate_open_chain(L: int, Jbond: np.ndarray, beta_list: np.ndarray = BETA_GRID, seed: int = 0):
    """
    Return (Energy[beta],C[beta, L-1]) for open bc chain.
    """
    mpo  = HeisenbergChainOpenBC(L, Jbond)
    nn   = NearestNeighborModel.from_MPOModel(mpo); nn.H_MPO = mpo.H_MPO
    psi  = PurificationMPS.from_infiniteT(nn.lat.mps_sites(), bc=nn.lat.bc_MPS)

    chi_max = 100 if L <= 64 else 200
    dt = 0.1

    eng = PurificationTEBD(psi, nn, dict(order=2, dt=dt,trunc_params=dict(chi_max=chi_max, svd_min=1e-10)))

    E, Cmat, beta_prev = [], [], 0.0
    # evolve over betas
    for beta in beta_list:                  
        eng.run_imaginary((beta-beta_prev)/2); beta_prev = beta
        # energy per site
        E.append(nn.H_MPO.expectation_value(psi) / math.sqrt(L))

        Cbeta = np.empty(L-1, dtype='f4') # one value per bond
        for i in range(L-1):
            # identical to your original snippet
            cxx = psi.expectation_value_term([('Sigmax', i), ('Sigmax', i+1)])
            cyy = psi.expectation_value_term([('Sigmay', i), ('Sigmay', i+1)])
            czz = psi.expectation_value_term([('Sigmaz', i), ('Sigmaz', i+1)])
            Cbeta[i] = (cxx + cyy + czz) / 3.0
        Cmat.append(Cbeta)

    return np.asarray(E,  dtype='f4'), np.stack(Cmat).astype('f4')
