import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from fkigp.configs import DatasetType
from fkigp.wsrlib import get_volumes, z_to_refl, idb
from fkigp.gridutils import get_basis


def get_suff_stats(radar, grid, default_val=-30, zmax=3000):
    LON, LAT, ALT, DBZ = get_volumes(radar, field='reflectivity', coords='geographic')

    inds = ALT <= zmax
    LON, LAT, ALT, DBZ = LON[inds], LAT[inds], ALT[inds], DBZ[inds]

    # Transform response variable
    DBZ[np.isnan(DBZ)] = default_val
    ETA, _ = z_to_refl(idb(DBZ))

    ETA = ETA ** (1 / 7)  # Nusbaummer paper

    X = np.vstack([LAT, LON, ALT]).T  # Dimension order: lat, lon alt

    W = get_basis(X, grid)
    n = W.shape[0]

    WT = W.T.tocsr()
    WTW = WT * W
    WTy = WT * ETA

    yty = ETA.T @ ETA

    return WTW, WTy,  yty, n, len(W.nonzero()[0])


def rbf_cov1D(X, Y=None, gamma=1.0):
    X = X.reshape(-1, 1)
    if Y is not None:
        Y = Y.reshape(-1, 1)
    return rbf_kernel(X, Y=Y, gamma=gamma)


def rbf_covND(X, Y=None, gamma=1.0):
    assert len(X.shape) == 2
    if Y is not None:
        assert len(Y.shape) == 2 and X.shape[1] == Y.shape[1]
    return rbf_kernel(X, Y=Y, gamma=gamma)


def sample(x, cov, gamma=1.0):
    n = len(x)
    mu = np.zeros((n))
    Sigma = cov(x, gamma=gamma)
    f = np.random.multivariate_normal(mu, Sigma)
    return f


def get_num_dims(data_type):
    if data_type == DatasetType.SYNGP:
        return 1
    if data_type == DatasetType.SOLAR:
        return 1
    elif data_type == DatasetType.AIRLINE:
        return 8
    elif data_type == DatasetType.SINE:
        return 1
    elif data_type == DatasetType.SOUND:
        return 1
    else:
        raise NotImplementedError

