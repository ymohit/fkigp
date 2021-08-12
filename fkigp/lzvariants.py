import numpy as np


def bsla_1d(A, B, k, **kwargs):
    '''
    Non-batched stochastic lanczos solver.
    '''
    B = B.reshape(-1, 1)
    return bsla(A, B, k, **kwargs)


def bsla(A,  # LHS -- n x n symmetric linear operator
         B,  # RHS -- n x t
         k,
         verbose=True):
    # Special
    if B.ndim == 1:
        return bsla_1d(A, B, k, verbose=verbose)

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

    # Initializing data holders
    Q = np.zeros((n, k, t))
    T = np.zeros((k, k, t))

    # Initalizing iterators
    q_i_minus_1 = np.zeros((n, t))
    q_i = B / np.sqrt((B * B).sum(axis=0))  ## Normalizing
    Q[:, 0, :] = q_i.copy()
    beta = np.zeros(t)

    # Running standard lanczos algorithm
    for i in range(k):

        # Computing alpha and T
        q_i_plus_1 = mvm_clsr(q_i) - beta * q_i_minus_1
        alpha = (q_i * q_i_plus_1).sum(axis=0)

        T[i, i, :] = alpha

        if i == k - 1:
            break
        # Computing beta and Q
        q_i_plus_1 = q_i_plus_1 - alpha * q_i
        q_i_plus_1 -= np.einsum('ijk,jk->ik', Q[:, :i, :], np.einsum('ijk,ik->jk', Q[:, :i, :], q_i_plus_1))
        beta = np.sqrt((q_i_plus_1 * q_i_plus_1).sum(axis=0))
        T[i, i + 1, :] = T[i + 1, i, :] = beta.copy()

        # Normalizing and sotring Q
        q_i_plus_1 = q_i_plus_1 / beta
        Q[:, i + 1, :] = q_i_plus_1.copy()

        q_i_minus_1 = q_i.copy()
        q_i = q_i_plus_1.copy()

    if B.shape[1] == 1:
        return Q.squeeze(), T.squeeze()
    else:
        return Q, T


def bfsla1_1d(A_hat, B, k, **kwargs):
    '''
    Non-batched stochastic lanczos solver.
    '''
    B = B.reshape(-1, 1)
    return bfsla1(A_hat, B, k, **kwargs)


def bfsla1(A_hat,  # LHS -- m x m symmetric linear operator
           B,  # RHS -- m x t
           k,
           verbose=True):
    # Special
    if B.ndim == 1:
        return bfsla1_1d(A_hat, B, k, verbose=verbose)

    # Initialization
    if type(A_hat) == np.ndarray:
        raise NotImplementedError
    else:
        if B.shape[1] == 1:
            mvm_clsr = A_hat.matvec
        else:
            mvm_clsr = A_hat.matmat

    # Checking dimensions
    m, t = B.shape
    m1, m2 = A_hat.shape
    assert m1 == m and m2 == m, "m1: " + str(m1) + ", m2: " + str(m2) + ", m: " + str(m)

    # Initializing data holders
    Q = np.zeros((m, k, t))
    WTWQ = np.zeros((m, k, t))
    T = np.zeros((k, k, t))

    WTW = A_hat.WTW

    # Initializing iterators
    q_i_minus_1 = np.zeros((m, t))

    AB, WTWB = mvm_clsr(B)
    q_i = B / np.sqrt((WTWB * B).sum(axis=0))  # Normalizing
    Aq_i = AB / np.sqrt((WTWB * B).sum(axis=0))
    WTWq_i = WTWB / np.sqrt((WTWB * B).sum(axis=0))
    beta = np.zeros(t)

    # Storing value for the first iterate
    Q[:, 0, :] = q_i.copy()
    WTWQ[:, 0, :] = WTWq_i.copy()

    # Running standard lanczos algorithm
    for i in range(k):

        # Computing alpha and T
        # Aq_i, WTWq_i = mvm_clsr(q_i)
        q_i_plus_1 = Aq_i - beta * q_i_minus_1
        alpha = (WTWq_i * q_i_plus_1).sum(axis=0)

        T[i, i, :] = alpha

        if i == k - 1:
            break

        # Computing beta and Q
        q_i_plus_1 = q_i_plus_1 - alpha * q_i
        q_i_plus_1 -= np.einsum('ijk,jk->ik', Q[:, :i, :], np.einsum('ijk,ik->jk', WTWQ[:, :i, :], q_i_plus_1))

        Aq_i_plus_1, WTW_q_i_plus_1 = mvm_clsr(q_i_plus_1)

        beta = np.sqrt((WTW_q_i_plus_1 * q_i_plus_1).sum(axis=0))

        T[i, i + 1, :] = T[i + 1, i, :] = beta.copy()

        # Normalizing and storing Q
        q_i_plus_1 = q_i_plus_1 / beta

        Q[:, i + 1, :] = q_i_plus_1.copy()
        WTWQ[:, i + 1, :] = WTW_q_i_plus_1 / beta

        q_i_minus_1 = q_i.copy()
        q_i = q_i_plus_1.copy()

        Aq_i = Aq_i_plus_1.copy() / beta
        WTWq_i = WTW_q_i_plus_1.copy() / beta

    if B.shape[1] == 1:
        return Q.squeeze(), T.squeeze()
    else:
        return Q, T


def bfsla2_1d(A, WTB, B_norm, **kwargs):
    WTB = WTB.reshape(-1, 1)
    return bfsla2(A, WTB, B_norm, **kwargs)


def bfsla2(A_hat,  # LHS -- m x m symmetric linear operator
           WTB,  # modified RHS, i.e. W^T * B  -- m x t
           B_norm,  # -- t
           k=30,  # rank of returned T matrices
           verbose=False
           ):
    # Special
    if WTB.ndim == 1:
        return bfsla2_1d(A_hat, WTB, B_norm, k=k, verbose=verbose)

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

    # Initializing data holders to store iterators
    # Representation here Q_bsla = W Q_bfsla + Q_s * B_normalized
    # Since, initial Q_bsla = B_normalized
    Q = np.zeros((m, k, t))
    Q_s = np.zeros((k, t))
    WTWQ = np.zeros((m, k, t))
    Q_s[0, :] = np.ones(t)

    T = np.zeros((k, k, t))
    L = np.zeros((k, t))  # (WTB * Q).sum(axis=0)  # Inner product between
    beta = np.zeros(t)

    # prev_q representation #  q_i_minus_1 = np.zeros((m, t))
    q_minus_1 = np.zeros((m, t))
    q_minus_1_s = np.zeros(t)

    # q representation # q_i
    KWTB = A_hat.kmm_matmat(WTB)
    sigma2 = A_hat.sigma ** 2

    q = q_minus_1.copy()
    q_s = np.ones(t)
    Aq = q_minus_1.copy()

    for i in range(k):

        q_plus_1 = Aq + q_s * KWTB - beta * q_minus_1  # KWTB_n
        q_plus_1_s = sigma2 - beta * q_minus_1_s  # sigma2 *

        # computing alpha
        alpha = (WTWQ[:, i, :] * q_plus_1).sum(axis=0) + q_s * q_plus_1_s + q_s * (WTB * q_plus_1).sum(
            axis=0) + q_plus_1_s * L[i, :]
        T[i, i, :] = alpha
        Q_s[i, :] = q_s

        if i == k - 1:
            break

        q_plus_1 -= alpha * q
        q_plus_1_s -= alpha * q_s

        # stabilization
        lmbda = np.einsum('ijk,ik->jk', WTWQ[:, :i, :], q_plus_1) \
            + (q_plus_1_s + (WTB * q_plus_1).sum(axis=0)) * Q_s[:i, :] + q_plus_1_s * L[:i, :]

        q_plus_1 -= np.einsum('ijk,jk->ik', Q[:, :i, :], lmbda)
        q_plus_1_s -= np.einsum('ij, ij->j', Q_s[:i, :], lmbda)

        # storing vectors after stabilization
        Aq, WTWQ[:, i + 1, :] = mvm_clsr(q_plus_1)
        L[i + 1, :] = (WTB * q_plus_1).sum(axis=0)

        beta = np.sqrt((WTWQ[:, i + 1, :] * q_plus_1).sum(axis=0) + q_plus_1_s ** 2 + 2 * q_plus_1_s * L[i + 1, :])

        Aq = Aq / beta.copy()
        WTWQ[:, i + 1, :] = WTWQ[:, i + 1, :] / beta.copy()
        L[i + 1, :] = L[i + 1, :] / beta.copy()
        q_plus_1 = q_plus_1 / beta.copy()
        q_plus_1_s = q_plus_1_s / beta.copy()

        T[i, i + 1, :] = T[i + 1, i, :] = beta.copy()
        Q[:, i + 1, :] = q_plus_1.copy()

        q_minus_1 = q.copy()
        q_minus_1_s = q_s.copy()
        q = q_plus_1.copy()
        q_s = q_plus_1_s.copy()

    if WTB.shape[1] == 1:
        return Q.squeeze(), T.squeeze(), Q_s.squeeze()
    else:
        return Q, T, Q_s
