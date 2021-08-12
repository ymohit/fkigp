import scipy
import numpy as np

import scipy.sparse.linalg
from fkigp.utils import NUMPY_DTYPE
from fkigp.solvers import solve
from fkigp.solvers import process_solve_kwargs


# Wrapper around scipy cg
def cg(A, b, x0=None, dump=None, **kwargs):
    """
    Runs scipy CG algorithm to solve sparse linear system Ax = b.

    :param A: Symmetric matrix describing the linear system
    :param b: Probe vector
    :param kwargs: Options supplied to reference implementation
    :return: solution x such that |Ax - b| < tol.
    """

    tag = 'Calling scipy.sparse.linalg.cg:'
    method = scipy.sparse.linalg.cg
    return solve(A, b, x0=x0, _solve_method=method, tag=tag, dump=dump, **kwargs)


def bfcg_1d(A_hat, R0, yty, dump, **kwargs):
    '''
    Non-batched Factorized Conjugate gradient solver. Wrapper to bfcg.
    '''
    R0 = R0.reshape(-1, 1)
    return bfcg(A_hat, R0=R0, yty=yty, dump=dump, **kwargs)


def bfcg(A_hat,  # LHS -- m x m linear operator
         R0,  # RHS -- m x t -- residual within in W span
         yty,  # RHS
         dump=None,
         **kwargs
         ):
    # Special
    if R0.ndim == 1:
        return bfcg_1d(A_hat, R0, yty=yty, dump=dump, **kwargs)

    tol, maxiter, __, verbose = process_solve_kwargs(**kwargs)

    atol = tol*np.sqrt(yty)

    # Initialization
    if type(A_hat) == np.ndarray:
        raise NotImplementedError
    else:
        if R0.shape[1] == 1:
            mvm_clsr = A_hat.matvec
        else:
            mvm_clsr = A_hat.matmat

    # Checking dimensions
    m, t = R0.shape
    m1, m2 = A_hat.shape
    assert m1 == m and m2 == m, "m1: " + str(m1) + ", m2: " + str(m2) + ", m: " + str(m)

    # Initializing vectors
    X_diff = np.zeros((m, t), dtype=NUMPY_DTYPE)

    R = R0.copy()
    V, U = mvm_clsr(R)
    U_T_R_prev = (U * R).sum(axis=0)

    beta = 0.0
    S = U
    Z = V

    # Terminated by maximum count
    iter_count = 1
    for iter_count in range(1, maxiter):

        # computing alpha
        S_T_Z = (S * Z).sum(axis=0)
        alpha = U_T_R_prev / S_T_Z

        # Updating iterates
        X_diff += alpha * S
        R -= alpha * Z  # r_kplus1

        # v_kplus1, u_kplus1
        V, U = mvm_clsr(R)
        U_T_R = (U * R).sum(axis=0)

        if all(np.sqrt(U_T_R) < atol):
            break

        beta = U_T_R / U_T_R_prev

        # Collecting dump variables
        # if dump is not None:
        #     dump += (X_diff.copy(), np.sqrt(U_T_R)),

        # defining iterates
        U_T_R_prev = U_T_R.copy()
        S = U + beta * S
        Z = V + beta * Z

    if verbose:
        if dump is not None:
            dump += iter_count + 1,
        print("NumIters:", iter_count + 1)

    elif kwargs.get('return_niters', False):
        return X_diff.squeeze(-1), iter_count

    if X_diff.shape[1] == 1:
        return X_diff.squeeze(-1)
    else:
        return X_diff


def bcg_1d(A, b, X0=None, **kwargs):
    '''
    Non-batched conjugate gradient solver. Wrapper to bcg.
    '''
    B = b.reshape(-1, 1)
    if X0 is not None:
        X0 = X0.reshape(-1, 1)
    return bcg(A, B, X0=X0, **kwargs)


def bcg(A,  # LHS -- n x n symmetric linear operator
        B,  # RHS -- n x t
        X0=None,  # initial solution -- n x t
        dump=None,
        **kwargs,
        ):

    # Special
    if B.ndim == 1:
        return bcg_1d(A, B, X0=X0, dump=dump, **kwargs)

    tol, maxiter, __, verbose = process_solve_kwargs(**kwargs)

    # Initialization
    if type(A) == np.ndarray:
        mvm_clsr = lambda x: A @ x
    else:
        if B.shape[1] == 1:
            mvm_clsr = A.matvec
        else:
            mvm_clsr = A.matmat

    # Checking dimensions
    n, t = B.shape
    n1, n2 = A.shape
    assert n1 == n and n2 == n

    # Standard conjugate gradient variables: all n x t
    if X0 is None:  # solutions
        X = np.zeros((n, t), dtype=NUMPY_DTYPE)
    else:  # solutions
        X = X0.copy()

    # Initializing vectors
    R = B - mvm_clsr(X)  # errors

    P = R.copy()  # preconditioned errors

    # For checking convergence
    atol = tol * np.linalg.norm(B, axis=0)

    RdotR_prev = np.sum(R * R, axis=0)

    iter_count = 0
    # Terminated by maximum count
    for iter_count in range(maxiter):

        # Standard conjugate gradient
        AP = mvm_clsr(P)
        alpha = RdotR_prev / (P * AP).sum(axis=0)

        X += alpha * P
        R -= alpha * AP

        RdotR = np.sum(R * R, axis=0)

        if all(np.sqrt(RdotR) < atol):
            break

        # if dump is not None:
        #     dump += (X.copy(), np.sqrt(RdotR.copy())),

        beta = RdotR / RdotR_prev
        P = R + beta * P

        RdotR_prev = RdotR.copy()

    if verbose:
        if dump is not None:
            dump += iter_count,
        print("NumIters:", iter_count)

    if X.shape[1] == 1:
        return X.squeeze(-1)
    else:
        return X


def mbcg_1d(A, b, X0=None, **kwargs):
    '''
    Non-batched modified conjugate gradient solver. Wrapper to mbcg.
    '''

    B = b.reshape(-1, 1)
    if X0 is not None:
        X0 = X0.reshape(-1, 1)

    X, T_diag, T_subdiag = mbcg(A, B, X0=X0, **kwargs)
    if T_diag is not None:
        T_diag = T_diag[:, 0]
        T_subdiag = T_subdiag[:, 0]

    return X, T_diag, T_subdiag


def mbcg(A,  # LHS -- n x n symmetric linear operator
         B,  # RHS -- n x t
         X0=None,  # initial solution -- n x t
         r=30,  # rank of returned T matrices
         returnT=True,  # whether to return T
         dump=None,
         **kwargs
         ):
    '''
    Modified block conjugate gradient from p.13 of Gardner, Pleiss, et al:

    Solve X = A^{-1}B

    Jacob R. Gardner, Geoff Pleiss, David Bindel, Killian Q. Weinberger,
    Andrew Gordon Wilson. GPyTorch: Blackbox Matrix-Matrix Gaussian Process
    Inference with GPU Acceleration NeurIPS 2018

    https://arxiv.org/abs/1809.11165

    There are several mistakes in the algorithm in the paper as of Jan 2020:
    -- In formula for beta_j, replace
            Z_j \circ Z_j     -->     Z_j \circ D_j
        Z_{j-1} \circ Z_{j-1} --> Z_{j-1} \circ D_{j-1}
    -- Update for D should *add* beta_j D_{j-1} instead of subtracting it
    -- There are indexing errors in the red formulas for the entries of T.
       Equation (S5) is correct and is a better reference
       1) A special case is needed when j = 0
       2) The RHS of the second update should use both beta[j-1] and alpha[j-1]
          so their indices match as in (S5). It currently uses beta[j-1] and alpha[j]
    '''

    # Special
    if B.ndim == 1:
        return mbcg_1d(A, B, X0=X0, r=r, dump=dump, **kwargs)

    tol, maxiter, __, verbose = process_solve_kwargs(**kwargs)

    # Initialization
    if type(A) == np.ndarray:
        mvm_clsr = lambda x: A @ x
    else:
        if B.shape[1] == 1:
            mvm_clsr = A.matvec
        else:
            mvm_clsr = A.matmat

    n, t = B.shape
    n1, n2 = A.shape
    assert n1 == n and n2 == n

    # Standard conjugate gradient variables: all n x t
    if X0 is None:  # solutions
        X = np.zeros((n, t), dtype=NUMPY_DTYPE)
    else:  # solutions
        X = X0.copy()

    # Initializing vectors
    R = B - mvm_clsr(X)  # errors
    P = R.copy()  # preconditioned errors

    # Lanczos coefficients
    alpha = np.zeros((maxiter, t))
    beta = np.zeros((maxiter, t))

    # For checking convergence
    atol = tol * np.linalg.norm(B, axis=0)
    RdotR_prev = np.sum(R * R, axis=0)

    j = 0
    # Run conjugate gradient
    for j in range(maxiter):

        # Standard conjugate gradient
        AP = mvm_clsr(P)

        alpha[j] = RdotR_prev / (P * AP).sum(axis=0)

        X += alpha[j] * P
        R -= alpha[j] * AP

        RdotR = np.sum(R * R, axis=0)

        if all(np.sqrt(RdotR) < atol):
            break

        beta[j] = RdotR / RdotR_prev

        P = R + beta[j] * P

        RdotR_prev = RdotR.copy()

    if verbose:
        if dump is not None:
            dump += j,
        print("NumIters:", j)

    # Compute tridiagonal matrices
    T_diag = None
    T_subdiag = None
    if returnT:
        # Rank for factored approximation should be at most:
        #    - number of iterations
        #    - the requested rank
        #    - the size of A
        r = np.min([j + 1, r, n])

        # Construct the diagonals and subdiagonals of the T matrices

        r = max(r, j)
        T_diag = 1 / alpha[0:r]
        T_diag[1:r] += beta[0:r - 1] / alpha[0:r - 1]
        T_subdiag = -np.sqrt(beta[0:r - 1]) / alpha[0:r - 1]

    return X, T_diag, T_subdiag


def mbfcg_1d(A, WTB, B_norm, X0=None, **kwargs):
    '''
    Non-batched modified conjugate gradient solver. Wrapper to mbcg.
    '''

    WTB = WTB.reshape(-1, 1)
    if X0 is not None:
        raise NotImplementedError
    X, T_diag, T_subdiag = mbcg(A, WTB, B_norm, X0=X0, **kwargs)
    if T_diag is not None:
        T_diag = T_diag[:, 0]
        T_subdiag = T_subdiag[:, 0]
    return X, T_diag, T_subdiag


def mbfcg(A_hat,  # LHS -- m x m symmetric linear operator
          WTB,  # modified RHS, i.e. W^T * B  -- m x t
          B_norm,  # -- t
          X0=None,  # initial solution -- n x t
          r=30,  # rank of returned T matrices
          returnT=True,  # whether to return T
          dump=None,
          **kwargs
          ):
    # Special
    if WTB.ndim == 1:
        return mbcg_1d(A_hat, WTB, B_norm, X0=X0, r=r, dump=dump)

    tol, maxiter, __, verbose = process_solve_kwargs(**kwargs)

    # Initialization
    if type(A_hat) == np.ndarray:
        raise NotImplementedError
    else:
        if WTB.shape[1] == 1:
            mvm_clsr = A_hat.matvec
        else:
            mvm_clsr = A_hat.matmat

    m, t = WTB.shape
    m1, m2 = A_hat.shape
    assert m1 == m and m2 == m, "m1: " + str(m1) + ", m2: " + str(m2) + ", m: " + str(m)

    # Standard conjugate gradient variables: all n x t
    if X0 is None:  # solutions
        X = np.zeros((m, t), dtype=NUMPY_DTYPE)
    else:  # solutions
        raise NotImplementedError
        # X = X0.copy()

    # Initializing vectors
    X = np.zeros((m, t))
    X_s = np.zeros(t)

    R = np.zeros((m, t))
    R_s = B_norm.copy()

    #     P = R.copy()  # preconditioned errors
    #     P_s = B_norm.copy()
    AP_prev = np.zeros((m, t))
    WTWP_prev = np.zeros((m, t))
    P_s = B_norm.copy()

    # Lanczos coefficientsx
    alpha = np.zeros((maxiter, t))
    beta = np.zeros((maxiter, t))

    # For checking convergence
    atol = tol * B_norm.copy()
    RdotR_prev = B_norm.copy() ** 2

    # Pre-computing for later usage
    KWTB = A_hat.kmm_matmat(WTB)
    sigma2 = A_hat.sigma ** 2

    WTBP = np.zeros(t)

    j = 0
    # Run conjugate gradient
    for j in range(maxiter):

        # Computing AP term
        # AP_1, WWP = mvm_clsr(P)
        AP = AP_prev + P_s * KWTB
        AP_s = sigma2 * P_s

        # Computing alpha term
        # PAP = (WTWP_prev * AP).sum(axis=0) + AP_s * P_s + P_s * (WTB * AP).sum(axis=0) + AP_s * (WTB * P).sum(axis=0)
        PAP = (WTWP_prev * AP).sum(axis=0) + AP_s * P_s + P_s * (WTB * AP).sum(axis=0) + AP_s * WTBP
        alpha[j] = RdotR_prev / PAP

        # Updating iterates
        X += alpha[j] * WTWP_prev
        X_s += alpha[j] * P_s
        R -= alpha[j] * AP
        R_s -= alpha[j] * AP_s

        # Extra WTW computation here ....?
        AR, WTWR = mvm_clsr(R)
        WTBR = (WTB * R).sum(axis=0)

        RdotR = (WTWR * R).sum(axis=0) + R_s * R_s + 2 * R_s * WTBR

        if all(np.sqrt(RdotR) < atol):
            break

        beta[j] = RdotR / RdotR_prev
        # print("j:", j, "alpha:", alpha[j], beta[j])

        AP_prev = AR + beta[j] * AP_prev
        WTWP_prev = WTWR + beta[j] * WTWP_prev
        WTBP = WTBR + beta[j] * WTBP
        P_s = R_s + beta[j] * P_s

        RdotR_prev = RdotR.copy()

    if verbose:
        if dump is not None:
            dump += j,
        print("NumIters:", j)

    # Compute tridiagonal matrices
    T_diag = None
    T_subdiag = None
    if returnT:
        # Rank for factored approximation should be at most:
        #    - number of iterations
        #    - the requested rank
        #    - the size of A
        r = np.min([j + 1, r, m])

        # Construct the diagonals and subdiagonals of the T matrices

        r = max(r, j)
        T_diag = 1 / alpha[0:r]
        T_diag[1:r] += beta[0:r - 1] / alpha[0:r - 1]
        T_subdiag = -np.sqrt(beta[0:r - 1]) / alpha[0:r - 1]

    return X, T_diag, T_subdiag
