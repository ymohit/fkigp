import scipy
import pickle
import warnings
import numpy as np


from fkigp.cgvariants import mbcg, mbfcg
from fkigp.lzvariants import bsla, bfsla2


def logdet_estimate_using_lz_variants(A,
                                      WT=None,
                                      trials=100,
                                      rank=20,
                                      verbose=True,
                                      dump=False,
                                      dump_path=None):

    n, _ = A.shape
    if WT is not None:
        n = WT.shape[1]

    if rank is None:
        rank = np.ceil(n / 2).astype(int)

    rank = np.min([n, rank])

    # Random Gaussian probes
    Z = np.random.randn(n, trials)

    if hasattr(A, 'WTW'):
        B_norm = np.sqrt((Z * Z).sum(axis=0))
        B = Z.copy() / B_norm
        WTB = WT * B
        _, T, C = bfsla2(A, WTB, B_norm, k=rank, verbose=verbose)
    else:
        _, T = bsla(A, Z, k=rank, verbose=verbose)

    T_diag = T.diagonal().T
    T_subdiag = T.diagonal(1).T

    vals = np.zeros(trials)

    for i in range(trials):
        d = T_diag[:, i]
        e = T_subdiag[:, i]
        try:
            L, V = scipy.linalg.eigh_tridiagonal(d, e)
            assert (np.all(L > 0))
            vals[i] = np.dot(Z[:, i], Z[:, i]) * np.sum(V[0, :] ** 2 * np.log(L))

        except:
            warnings.warn("negative eigenvalue in logdet estimation")
            vals[i] = np.nan
    result = np.nanmean(vals)

    if dump:
        assert dump_path is not None
        pickle.dump((T_diag, T_subdiag, Z, result), open(dump_path + "/llk_dump.pkl", "wb"))
    return result, vals


def logdet_estimate_using_cg_variants(A, WT=None, trials=100, tol=1e-5, rank=20, verbose=True):
    n, _ = A.shape
    if WT is not None:
        n = WT.shape[1]

    if rank is None:
        rank = np.ceil(n / 2).astype(int)

    rank = np.min([n, rank])

    # Random Gaussian probes
    Z = np.random.randn(n, trials)

    if hasattr(A, 'WTW'):
        B_norm = np.sqrt((Z * Z).sum(axis=0))
        B = Z.copy() / B_norm
        WTB = WT * B
        _, T_diag, T_subdiag = mbfcg(A, WTB, B_norm, r=rank, tol=tol, maxiter=rank, verbose=verbose)
    else:
        _, T_diag, T_subdiag = mbcg(A, Z, r=rank, tol=tol, maxiter=rank, verbose=verbose)
    vals = np.zeros(trials)

    for i in range(trials):
        d = T_diag[:, i]
        e = T_subdiag[:, i]
        try:
            L, V = scipy.linalg.eigh_tridiagonal(d, e)
            assert (np.all(L > 0))
            vals[i] = np.dot(Z[:, i], Z[:, i]) * np.sum(V[0, :] ** 2 * np.log(L))

        except:
            warnings.warn("negative eigenvalue in logdet estimation")
            vals[i] = np.nan
    result = np.nanmean(vals)
    return result, vals
