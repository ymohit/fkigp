import scipy
import numpy as np

# Default parameters
DEFAULT_TOL = 1e-5
PRINT_FREQ = 200
MAX_ITER = 1000
VERBOSE = False
DUMP_SOL_FREQ = 5
DEBUG = False


def id_operator(n):
    return scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda v: v)


def chol_trid(d, e):
    '''
    Cholesky factorization of a symmetric, tridiagonal positive
    definite matrix A.

    From Chapter 7.3 of "Introduction to Scientific Computing,
    A Matrix-Vector Approach Using Matlab" by Van Loan

    Input:
      d: diagonal of A (length n)
      e: sub- and super-diagonal of A (length n-1)

    Returns:
      g: diagonal of G (length n)
      h: subdiagonal of G (length n-1)

    Satisfies A = GG^T
    '''

    n = len(d)
    assert len(e) == n - 1

    g = np.zeros(n)
    h = np.zeros(n)
    e = np.insert(e, 0, 0)  # extend by one so indexing matches reference implementation
    g[0] = np.sqrt(d[0])
    for i in range(1, n):
        h[i] = e[i] / g[i - 1]
        g[i] = np.sqrt(d[i] - h[i] ** 2)

    return g, h[1:]


def chol_trid_solve(g, h, b):
    '''
    Solves the linear system GG^Tx = b where b is an n-vector and
    G is a nonsingular lower bidiagonal matrix.

    From Chapter 7.3 of "Introduction to Scientific Computing,
    A Matrix-Vector Approach Using Matlab" by Van Loan

    Input:
      g: diagonal of G (length n)
      h: subdiagonal of G (length n-1)

    Returns:
      x: solution to GG^Tx = b (length n)
    '''

    n = len(g)
    y = np.zeros_like(b)

    assert len(b) == n
    assert len(h) == n - 1

    h = np.insert(h, 0, 0)  # extend by one so indexing matches reference implementation

    # Solve Gy = b
    y[0] = b[0] / g[0]
    for k in range(1, n):
        y[k] = (b[k] - h[k] * y[k - 1]) / g[k]

    # Solve G'x = y
    x = np.zeros_like(b)
    x[n - 1] = y[n - 1] / g[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = (y[k] - h[k + 1] * x[k + 1]) / g[k]

    return x


# Source: https://github.com/NathanWycoff/PivotedCholesky/blob/master/pivoted_chol.py
def pivoted_chol(get_diag, get_row, M, err_tol=1e-6):
    """
    A simple python function which computes the Pivoted Cholesky decomposition/approximation of positive semi-definite
        operator. Only diagonal elements and select rows of that operator's matrix represenation are required.
    get_diag - A function which takes no arguments and returns the diagonal of the matrix when called.
    get_row - A function which takes 1 integer argument and returns the desired row (zero indexed).
    M - The maximum rank of the approximate decomposition; an integer.
    err_tol - The maximum error tolerance, that is difference between the approximate decomposition and true matrix,
        allowed. Note that this is in the Trace norm, not the spectral or frobenius norm.
    Returns: R, an upper triangular matrix of column dimension equal to the target matrix. It's row dimension will be
        at most M, but may be less if the termination condition was acceptably low error rather than max iters reached.
    """

    d = np.copy(get_diag())
    N = len(d)
    pi = list(range(N))
    R = np.zeros([M, N])
    err = np.sum(np.abs(d))

    m = 0
    while (m < M) and (err > err_tol):

        i = m + np.argmax([d[pi[j]] for j in range(m, N)])

        tmp = pi[m]
        pi[m] = pi[i]
        pi[i] = tmp

        R[m, pi[m]] = np.sqrt(d[pi[m]])
        Apim = get_row(pi[m])
        for i in range(m+1, N):
            if m > 0:
                ip = np.inner(R[:m, pi[m]], R[:m, pi[i]])
            else:
                ip = 0
            R[m, pi[i]] = (Apim[pi[i]] - ip) / R[m, pi[m]]
            d[pi[i]] -= pow(R[m, pi[i]], 2)

        err = np.sum([d[pi[i]] for i in range(m+1, N)])
        m += 1

    R = R[:m, :]

    return R


# Wrapper around scipy cg
def cg(A, b, **kwargs):
    tag = 'Calling scipy.sparse.linalg.cg:'
    method = scipy.sparse.linalg.cg
    return solve(A, b, _solve_method=method, tag=tag, **kwargs)


# Wrapper around scipy bicgstab
def bicgstab(A, b, **kwargs):
    tag = 'Calling scipy.sparse.linalg.bicgstab:'
    method = scipy.sparse.linalg.bicgstab
    return solve(A, b, _solve_method=method, tag=tag, **kwargs)


def process_solve_kwargs(**kwargs):
    """
    Processes arguments before supplying to solve
    :param kwargs:
    :return:
    """

    tol = kwargs.get('tol', DEFAULT_TOL)
    maxiter = kwargs.get('maxiter', MAX_ITER)
    Ainv = kwargs.get('Ainv', None)
    verbose = kwargs.get('verbose', False)

    if VERBOSE:
        print("tol:", tol)
        print("maxiter:", maxiter)
        print("Ainv:", Ainv)

    return tol, int(maxiter), Ainv, verbose


# Copied private method scipy.sparse.linalg.isolve.iterative._get_atol
def get_atol(tol, atol, bnrm2, get_residual, routine_name):
    """
    Parse arguments for absolute tolerance in termination condition.

    Parameters
    ----------
    tol, atol : object
        The arguments passed into the solver routine by user.
    bnrm2 : float
        2-norm of the rhs vector.
    get_residual : callable
        Callable ``get_residual()`` that returns the initial value of
        the residual.
    routine_name : str
        Name of the routine.
    """

    if atol is None:
        atol = 'legacy'

    tol = float(tol)

    if atol == 'legacy':
        resid = get_residual()
        if resid <= tol:
            return 'exit'
        if bnrm2 == 0:
            return tol
        else:
            return tol * float(bnrm2)
    else:
        return max(float(atol), tol * float(bnrm2))


def solve(A, b, x0=None, _solve_method=scipy.sparse.linalg.cg,
          tag='Calling scipy.sparse.linalg.cg:',
          dump=None,
          **kwargs):

    tol, maxiter, Ainv, verbose = process_solve_kwargs(**kwargs)

    if maxiter <= 0:
        maxiter = None

    i = 0

    def callback(x):
        nonlocal i
        nonlocal dump
        i += 1

        if dump is not None:
            dump += x.copy(),

        if i % PRINT_FREQ == 0 and verbose:
            print("Iter: ", i)

    x, info = _solve_method(A, b, x0=x0, callback=callback, tol=tol, maxiter=maxiter, M=Ainv)
    print('NumIters:', i)

    return x
