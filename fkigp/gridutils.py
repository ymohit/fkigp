import scipy
import numpy as np


# Return the grid coordinates in each dimension
def grid_coords(grid):
    assert all([g[2] > 1] for g in grid) # check n_points > 1 in each dimension
    return [np.linspace(*g) for g in grid]


# Return total number of grid points
def grid_size(grid):
    return np.prod([g[2] for g in grid])


# Return the actual grid points in an m x d array
def grid_points(grid):
    d = len(grid)
    coords = np.meshgrid(*grid_coords(grid)) # grid points with coordinates in separate array
    points = np.stack(coords).reshape(d,-1).T # stack arrays and reshape to m x d
    return points


def get_basis(x, grid, **kwargs):
    '''
    Get the KISS-GP "W" matrix, which is a basis expansion

    In multiple dimensions, the basis a product of 1d basis expansion, so this
    functions calls get_basis_1d separately for each dimension separately and
    the combines the results.
    '''

    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim > 2:
        raise ValueError("x should have at most two-dimensions")

    n, d = x.shape

    if len(grid) != d:
        raise ValueError("Second dim of x (shape (%d, %d)) must match len(grid)=%d" % (n, d, len(grid)))

    m = grid_size(grid)

    # Get basis expansions in each dimension
    W_dims = [get_basis_1d(x[:, j], grid[j], **kwargs) for j in range(d)]

    # Compute kron product of rows
    W = W_dims[0]
    for j in range(1, d):
        W = sparse_outer_by_row(W, W_dims[j])

    return W


def sparse_outer_by_row(A, B):
    n, m1 = A.shape
    n2, m2 = B.shape

    assert (n == n2)

    A_row_size = np.diff(A.indptr)
    B_row_size = np.diff(B.indptr)

    # Size of each row of C is product of A and B sizes
    C_row_size = A_row_size * B_row_size

    # Construct indptr for C (indices of first entry of each row)
    C_indptr = np.zeros(n + 1, dtype='int')
    C_indptr[1:] = np.cumsum(C_row_size)

    # These arrays have one entry for each entry of C
    #
    #   C_row_num    what row entry is in
    #   C_row_start  start index of the row
    #   C_row_pos    position within nonzeros of this row
    #
    C_row_num = np.repeat(np.arange(n), C_row_size)
    C_row_start = np.repeat(C_indptr[:-1], C_row_size)
    C_nnz = np.sum(C_row_size)
    C_row_pos = np.arange(C_nnz) - C_row_start

    # Now compute corresponding row positions for A and B
    second_dim_size = B_row_size[C_row_num]
    A_row_pos = C_row_pos // second_dim_size
    B_row_pos = C_row_pos % second_dim_size

    # Convert row positions to absolute positions in the data/indices arrays for A and B
    A_pos = A.indptr[C_row_num] + A_row_pos
    B_pos = B.indptr[C_row_num] + B_row_pos

    # Construct the indices and data for C
    C_indices = A.indices[A_pos] * m2 + B.indices[B_pos]
    C_data = A.data[A_pos] * B.data[B_pos]

    # Finally, make C
    C = scipy.sparse.csr_matrix((C_data, C_indices, C_indptr), shape=(n, m1 * m2))

    return C


# Helper routine to compute outer products between paired rows from (A,B)
# when A and B are sparse. The outer products are then flattened into single
# rows of the new matrix
def sparse_outer_by_row_slow(A, B):
    # Based on https://stackoverflow.com/questions/57099722/row-wise-outer-product-on-sparse-matrices

    assert scipy.sparse.isspmatrix_csr(A)
    assert scipy.sparse.isspmatrix_csr(B)

    n, m1 = A.shape
    n2, m2 = B.shape

    assert (n == n2)

    A_splits = A.indptr[1:-1]
    B_splits = B.indptr[1:-1]

    A_data = np.split(A.data, A_splits)
    B_data = np.split(B.data, B_splits)

    AB_data = [np.outer(a, b).ravel() for a, b in zip(A_data, B_data)]

    A_indices = np.split(A.indices, A_splits)
    B_indices = np.split(B.indices, B_splits)
    AB_subs = [np.ix_(a, b) for a, b in zip(A_indices, B_indices)]

    AB_col_indices = [np.ravel_multi_index(subs, (m1, m2)).ravel() for subs in AB_subs]

    # Create flat arrays for final indices and data
    AB_row_indices = np.repeat(np.arange(n), [len(row) for row in AB_data])
    AB_col_indices = np.concatenate(AB_col_indices)
    AB_data = np.concatenate(AB_data)

    # Create return matrix
    C = scipy.sparse.coo_matrix(((AB_data), (AB_row_indices, AB_col_indices)), (n, m1 * m2)).tocsr()

    return C


# Prototype linear basis function: value 1 at zero, and ramps linearly
# down to value 0 at +-1
def linear_kernel(dist):
    vals = np.zeros_like(dist)
    inds = dist < 1
    vals[inds] = 1-dist[inds]
    return vals


# Prototype cubic basis function centered at zero:
# positive between [-1, +1], negative in [-2, -1] and [1, 2], and zero outside of [-2, 2]
def cubic_kernel(dist, a=-0.5):
    # Formula from:
    #   https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    #
    # Original source:
    #   R. Keys (1981). "Cubic convolution interpolation for digital image processing".
    #   IEEE Transactions on Acoustics, Speech, and Signal Processing. 29 (6): 1153–1160.
    #   doi:10.1109/TASSP.1981.1163711.

    vals = np.zeros_like(dist)

    inds = dist <= 1
    vals[inds] = (a + 2) * dist[inds] ** 3 - (a + 3) * dist[inds] ** 2 + 1

    inds = (1 < dist) & (dist < 2)
    vals[inds] = a * dist[inds] ** 3 - 5 * a * dist[inds] ** 2 + 8 * a * dist[inds] - 4 * a
    return vals


# Get grid-based basis function expansion for points in x in 1d
def get_basis_1d(x, grid, kind='cubic', **kwargs):
    n = len(x)

    umin, umax, m = grid  # grid min, max, # points
    assert (m > 1)
    u = np.linspace(umin, umax, m)  # grid points
    du = u[1] - u[0]  # grid spacing

    # Returns index of closest grid point to left of z
    def grid_pos(z):
        return np.floor(((z - umin) / du)).astype(int)

    # Specifies which grid points will have nonzero basis function
    # values for a point relative to it's grid position
    if kind == 'cubic':
        offsets = np.array([-1, 0, 1, 2])
    elif kind == 'linear':
        offsets = np.array([0, 1])
    else:
        raise ValueError('unrecognized kind')

    # Generate indices for all (input point, offset) pairs
    I, K = np.mgrid[0:n, 0:len(offsets)]

    # Index of neighboring grid point for each (input point, offset) pair
    J = grid_pos(x[I]) + offsets[K]

    # Drop (input point, grid point) pairs where grid index is out of bounds
    valid_inds = (J >= 0) & (J < m)
    I = I[valid_inds]
    J = J[valid_inds]

    # Compute distance of each (inputs point, grid point) pair and scale by du
    dist = np.abs((x[I] - u[J]) / du)

    # Now evaluate the kernel
    if kind == 'cubic':
        vals = cubic_kernel(dist, **kwargs)
    elif kind == 'linear':
        vals = linear_kernel(dist, **kwargs)
    else:
        raise NotImplementedError

    # Return sparse matrix in sparse row format
    W = scipy.sparse.coo_matrix((vals, (I, J)), shape=(n, m)).tocsr()

    return W
