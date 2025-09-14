# common/feature_map.py
import numpy as np, math
from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoCV
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from typing import Sequence, Tuple




#--------------------------------------------------------------------------------
# Heisenberg OPEN 
# --------------------------------------------------------------------------------


def ZOfQubitHeisenberg(periodic: bool, J: np.ndarray, a: int, distance: int) -> np.ndarray:
    """Length-(2delta+1) window centred at a (wrap if periodic)."""
    L = len(J)
    z = np.zeros(2*distance+1)
    for off in range(-distance, distance+1):
        idx = (a+off) % L if periodic else a+off
        if 0 <= idx < L:
            z[off+distance] = J[idx]
    return z

def PhiOfZ(distance, R, gamma, Z_ext, omegas):
    s = math.sqrt(len(Z_ext))
    return np.hstack([ [math.cos(gamma/s * omegas[r] @ Z_ext),math.sin(gamma/s * omegas[r] @ Z_ext)]for r in range(R) ])

class FeatureMappedLassoThermal(BaseEstimator):
    """
    One class for both geometries technically. I created separate Periodic however.
    Set periodic=True for rings,
    False for open chains.
    """
    def __init__(self, periodic, n, distance,alphas, omegas, R=20, gamma=0.5):
        self.periodic, self.n, self.distance = periodic, n, distance
        self.alphas, self.omegas = alphas, omegas
        self.R, self.gamma = R, gamma

    def get_params(self, deep=True):
        return dict(periodic=self.periodic,
                n=self.n,
                distance=self.distance,
                alphas=self.alphas,
                omegas=self.omegas,
                R=self.R,
                gamma=self.gamma)

    def set_params(self, **params):
        for k,v in params.items():
            setattr(self, k, v)
        return self

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # protection for LassoCV
        if len(y) < 5:             
            rep = (5 // len(y)) + 1
            X = np.vstack([X]*rep)[:5]
            y = np.tile(y, rep)[:5]

        L_eff = self.n if self.periodic else self.n-1
        phi = np.zeros((len(y), 2*self.R*L_eff))
        for i,(Jbeta) in enumerate(X):
            J, beta = Jbeta[:-1], Jbeta[-1]
            feats = [PhiOfZ(self.distance, self.R, self.gamma,
                    np.append(ZOfQubitHeisenberg(self.periodic, J, a, self.distance), beta),self.omegas)
                    for a in range(L_eff)]
            phi[i] = np.hstack(feats)
        self.model = LassoCV(alphas=self.alphas, cv=5, n_jobs=-1).fit(phi, y)
        return self

    def predict(self, X):
        check_is_fitted(self.model)
        L_eff = self.n if self.periodic else self.n-1
        phi = np.zeros((len(X), 2*self.R*L_eff))
        for i,(Jbeta) in enumerate(X):
            J, beta = Jbeta[:-1], Jbeta[-1]
            feats = [PhiOfZ(self.distance, self.R, self.gamma,
                    np.append(ZOfQubitHeisenberg(self.periodic, J, a, self.distance), beta), self.omegas) 
                    for a in range(L_eff)]
            phi[i] = np.hstack(feats)
        return self.model.predict(phi)


#--------------------------------------------------------------------------------
# Heisenberg Periodic 
# --------------------------------------------------------------------------------

def ZOfQubitHeisenberg_periodic(J, a, distance):
    """
    Build the (2*distance+1) length window around site a, wrapping mod L.
    """
    L = len(J)
    Z = np.zeros(2*distance + 1)
    for offset in range(-distance, distance+1):
        idx = (a + offset) % L
        Z[offset + distance] = J[idx]
    return Z


class FeatureMappedLassoThermalPeriodic(BaseEstimator):
    def __init__(self, periodic, n, distance, alphas, omegas, R=20, gamma=0.5):
        self.periodic = periodic
        self.n        = n
        self.distance = distance
        self.alphas   = alphas
        self.omegas   = omegas
        self.R        = R
        self.gamma    = gamma

    def get_params(self, deep=True):
        return dict(periodic=self.periodic,
                    n=self.n,
                    distance=self.distance,
                    alphas=self.alphas,
                    omegas=self.omegas,
                    R=self.R,
                    gamma=self.gamma)

    def set_params(self, **params):
        for k,v in params.items():
            setattr(self, k, v)
        return self

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if len(y) < 5:
            # pad
            rep = 5 // len(y)
            X = np.vstack([X] * (rep + 1))[:5]
            y = np.tile(y, rep + 1)[:5]

        Phi = np.zeros((len(y), 2 * self.R))
        for i, x in enumerate(X):
            J, beta = x[:-1], x[-1]
            Z_loc = ZOfQubitHeisenberg_periodic(J, a=0, distance=self.distance)
            Z_ext = np.append(Z_loc, beta)
            Phi[i, :] = PhiOfZ(self.distance, self.R, self.gamma, Z_ext, self.omegas)

        # Fit LassoCV on that window only design
        self.model = LassoCV(alphas=self.alphas, cv=5, random_state=0).fit(Phi, y)
        return self

    def predict(self, X):
        check_is_fitted(self.model)
        M = len(X)
        Phi = np.zeros((M, 2 * self.R)) # no * self.n here
        for i, x in enumerate(X):
            J, beta = x[:-1], x[-1]
            Z_loc    = ZOfQubitHeisenberg_periodic(J, a=0, distance=self.distance)
            Z_ext    = np.append(Z_loc, beta)
            Phi[i, :] = PhiOfZ(self.distance, self.R, self.gamma, Z_ext, self.omegas)
        return self.model.predict(Phi)

#--------------------------------------------------------------------------------
#  Window only predictor (made for  FeatureMappedLassoThermalPeriodic)
# --------------------------------------------------------------------------------
def predict_all_edges(model: FeatureMappedLassoThermalPeriodic, X):
    """
    Apply the single window Lasso (trained on edge 0) to every edge. Returns Y with shape (M, L).
    """
    L, R = model.n, model.R
    w, b = model.model.coef_, model.model.intercept_  # length 2R
    Y = np.empty((len(X), L), dtype=float)

    for m in range(len(X)):
        J, beta = X[m, :-1], X[m, -1]
        for a in range(L):
            z = ZOfQubitHeisenberg_periodic(J, a, model.distance)
            phi = PhiOfZ(model.distance, R, model.gamma, np.append(z, beta), model.omegas) # (2R,)
            Y[m, a] = phi @ w + b
    return Y






#--------------------------------------------------------------------------------
# ISING OPEN 
# --------------------------------------------------------------------------------
def ZOfQubitIsing(periodic: bool, J: np.ndarray, h: np.ndarray, a: int, distance: int):
    # Interleave J and h, following GS version of ZOfQubitIsing
    LJ = len(J)
    Z = np.zeros(2 *(2*distance + 1))
    for offset in range(-distance, distance + 1):
        idx = (a + offset) % LJ if periodic else a + offset
        if 0 <= idx < LJ:
            pos = 2 * (distance + offset)
            Z[pos] = J[idx]
            Z[pos + 1] = h[idx]
    return Z


def PhiOfZIsing(distance, R, gamma, Z, beta, omegas):
    # Maps to randomized Fourier features, appending beta for thermal
    Z_with_beta = np.append(Z,beta)
    l = len(Z_with_beta)
    phi = np.zeros(2 * R)
    for s in range(R):
        prod = np.dot(omegas[s], Z_with_beta)
        phi[2 * s] = math.cos(gamma / math.sqrt(l) * prod)
        phi[2 * s + 1] = math.sin(gamma / math.sqrt(l) * prod)
    return phi


class FeatureMappedLassoThermalIsing(BaseEstimator): 
    def __init__(self, periodic: bool, n: int, distance: int,alphas: Sequence[float], omegas: np.ndarray, H_eval: float,R: int=20, gamma: float=0.5):
        self.periodic, self.n, self.distance = periodic, n, distance
        self.alphas, self.omegas = alphas, omegas
        self.H_eval = H_eval
        self.R, self.gamma = R, gamma

        
    def get_params(self, deep=True): 
        return {'alphas': self.alphas, 'omegas': self.omegas, 'R': self.R, 'gamma': self.gamma, 'H_eval': self.H_eval,
                'n': self.n, 'distance': self.distance, 'periodic': self.periodic} 
    def set_params(self, **p):
        for k, v in p.items(): setattr(self, k, v)
        return self
    
    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)
        if len(y) < 5:
            rep = 5 // len(y)
            X = np.vstack([X] * (rep + 1))[:5]
            y = np.tile(y, rep + 1)[:5]
        M = len(X)
        Phi = np.zeros((M, 2 * self.R * self.n))  ##Shape matches repo (2*R*n for n sites)
        for i, x in enumerate(X):
            J, beta = x[:-1], x[-1]
            # fixed h array for Ising
            h_fixed = np.full(len(J), self.H_eval)  
            feats = []
            # range(self.n), matching repo's loop over n sites
            for a in range(self.n):  
                Z = ZOfQubitIsing(self.periodic, J, h_fixed, a, self.distance)
                feats.append(PhiOfZIsing(self.distance, self.R, self.gamma, Z, beta, self.omegas))
            Phi[i] = np.concatenate(feats)
        self.model = LassoCV(alphas=self.alphas, cv=5, random_state=0, n_jobs=-1)
        self.model.fit(Phi, y)
        return self
    
    def predict(self, X: np.ndarray):
        check_is_fitted(self.model)
        M = len(X)
        Phi = np.zeros((M, 2 * self.R * self.n))  # Same shape as in fit
        for i, x in enumerate(X):
            J, beta = x[:-1], x[-1]
            # Fixed h array for Ising
            h_fixed = np.full(len(J), self.H_eval) 
            feats = []
            #range(self.n)
            for a in range(self.n):  
                Z = ZOfQubitIsing(self.periodic, J, h_fixed, a, self.distance)
                feats.append(PhiOfZIsing(self.distance, self.R, self.gamma, Z, beta, self.omegas))
            Phi[i] = np.concatenate(feats)
        return self.model.predict(Phi)



