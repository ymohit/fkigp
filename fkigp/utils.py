import os
import time
import scipy
import argparse
import logging
import numpy as np


from pprint import pprint
from fkigp.configs import Structdict
from fkigp.configs import GridSizeFunc
from fkigp.configs import GsGPType
from fkigp.configs import Frameworks
from fkigp.configs import DatasetType
from fkigp.configs import MethodName
from fkigp.configs import ExperimentType

NUMPY_DTYPE = np.float64

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def carray(*args, **kwargs):
    kwargs.setdefault("dtype", NUMPY_DTYPE)
    return np.array(*args, **kwargs)


def tic():
    wall = time.perf_counter_ns()
    sys = time.process_time_ns()
    return wall, sys


def toc(t0):
    wall0, sys0 = t0
    wall1, sys1 = tic()
    return wall1 - wall0, sys1 - sys0


def toc_report(t, tag='', return_val=False):
    wall, sys = t
    print('%8s: wall=%.2f ms, sys=%.2f ms' % (tag, wall / 1e6, sys / 1e6))
    if return_val:
        return wall / 1e6
    return


def ndimension(x):

    if len(x.shape) == 1:
        return 1
    elif len(x.shape) == 2:
        return x.shape[1]
    else:
        raise NotImplementedError


# Get error bars from a covariance matrix
def cov2err(K):
    return 1.96*np.sqrt(np.diag(K))


def id_operator(n):
    return scipy.sparse.linalg.LinearOperator((n,n), matvec = lambda v: v)


# Source:- https://gist.github.com/ahwillia/f65bc70cb30206d4eadec857b98c4065
def unfold(tens, mode, dims):
    """
    Unfolds tensor into matrix.
    Parameters
    ----------
    tens : ndarray, tensor with shape == dims
    mode : int, which axis to move to the front
    dims : list, holds tensor shape
    Returns
    -------
    matrix : ndarray, shape (dims[mode], prod(dims[/mode]))
    """
    if mode == 0:
        return tens.reshape(dims[0], -1)
    else:
        return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)


# Source:- https://gist.github.com/ahwillia/f65bc70cb30206d4eadec857b98c4065
def refold(vec, mode, dims):
    """
    Refolds vector into tensor.
    Parameters
    ----------
    vec : ndarray, tensor with len == prod(dims)
    mode : int, which axis was unfolded along.
    dims : list, holds tensor shape
    Returns
    -------
    tens : ndarray, tensor with shape == dims
    """
    if mode == 0:
        return vec.reshape(dims)
    else:
        # Reshape and then move dims[mode] back to its
        # appropriate spot (undoing the `unfold` operation).
        tens = vec.reshape(
            [dims[mode]] +
            [d for m, d in enumerate(dims) if m != mode]
        )
        return np.moveaxis(tens, 0, mode)


# Convert 2d grid specification into "extent" argument for plt.imshow()
def grid2extent(grid):
    assert (len(grid) == 2)

    ymin, ymax, ny = grid[0]
    xmin, xmax, nx = grid[1]
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    return [xmin - 0.5 * dx, xmax + 0.5 * dx, ymin - 0.5 * dy, ymax + 0.5 * dy]


def get_options():

    parser = argparse.ArgumentParser(description="Running experiments ...")

    # experiment level
    parser.add_argument("--experiment_type", default=0, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--store_ref", action='store_true')
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--framework", default=1, type=int)
    parser.add_argument("--download_radar", action='store_true')
    parser.add_argument("--entire_us", action='store_true')

    parser.add_argument("--method", default=1, type=int)
    parser.add_argument("--gsgp_type", default=0, type=int)  # 0-> Asym and 1-> Sym
    parser.add_argument("--grid_size_f", default=-1, type=int)
    parser.add_argument("--grid_size", default=-1, type=int)    # -1 positive is applicable in deciding the grid size
    parser.add_argument("--grid_idx", default=-1, type=int)

    # data options
    parser.add_argument("--data_type", default=0, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--sigma", default=-1, type=float)

    # inference arguments
    parser.add_argument("--tol", default=1e-2, type=float)
    parser.add_argument("--maxiter", default=1e3, type=int)
    parser.add_argument("--variant", default=2, type=int)
    parser.add_argument("--ntrials", default=30, type=int)
    parser.add_argument("--ref_logdet", default=0.0, type=float)

    # parsing options
    options = parser.parse_args()

    if options.debug:
        import pdb; pdb.set_trace()

    # processing options
    options = Structdict(vars(options))

    options.framework = Frameworks(options.framework)
    options.data_type = DatasetType(options.data_type)
    options.gsgp_type = GsGPType(options.gsgp_type)
    options.grid_size_f = GridSizeFunc(options.grid_size_f)
    options.method = MethodName(options.method)
    options.experiment_type = ExperimentType(options.experiment_type)

    pprint(options)
    return options
