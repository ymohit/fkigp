import math
import numpy as np
import fastmat as fm
from sklearn.metrics.pairwise import rbf_kernel

from fkigp.gps.gpbase import BaseKernel
from fkigp.gps.constraints import Positive
from fkigp.gps.constraints import DEFAULT_SOFTPLUS_VALUE
from fkigp.gridutils import get_basis, grid_coords


class ScaleKernel(BaseKernel):

    def __init__(self, base_kernel, outputscale_constraint=None, **kwargs):
        super().__init__(has_lengthscale=False, **kwargs)

        if outputscale_constraint is None:
            outputscale_constraint = Positive(initial_value=DEFAULT_SOFTPLUS_VALUE, transform='softplus')

        self.base_kernel = base_kernel
        self.register_parameter(name="outputscale",
                                value=DEFAULT_SOFTPLUS_VALUE,
                                constraint=outputscale_constraint)

    def forward(self, x1, x2=None, diag=False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag, **params)
        if diag:
            raise NotImplementedError
        return self.outputscale*orig_output

    @property
    def outputscale(self):
        return self.get_parameter(name='outputscale')

    @outputscale.setter
    def outputscale(self, value):
        self.set_parameter(name='outputscale', value=value)

    @property
    def raw_outputscale(self):
        return self.get_raw_parameter(name='outputscale')

    @raw_outputscale.setter
    def raw_outputscale(self, value):
        self.set_raw_parameter(name='outputscale', value=value)


class RBFKernel(BaseKernel):

    def __init__(self, **kwargs):

        ard_num_dims = kwargs.get('ard_num_dims', None)
        super().__init__(has_lengthscale=True, ard_num_dims=ard_num_dims)

    def forward(self, x1, x2=None, diag=False, **params):

        # Getting lengthscale for the correct dimension
        active_dim = params.get('active_dim', None)
        if active_dim is None:
            assert np.isscalar(self.lengthscale) or len(self.lengthscale) == 1
            if np.isscalar(self.lengthscale):
                lengthscale = self.lengthscale
            else:
                lengthscale = self.lengthscale[0]

        elif active_dim == 0:
            assert np.isscalar(self.lengthscale) or len(self.lengthscale) >= 1
            if np.isscalar(self.lengthscale):
                lengthscale = self.lengthscale
            else:
                lengthscale = self.lengthscale[0]
        else:
            assert not np.isscalar(self.lengthscale) and len(self.lengthscale) > active_dim
            lengthscale = self.lengthscale[active_dim]

        # Returning the kernel values
        x1_ = np.divide(x1, lengthscale*np.sqrt(2))
        if x2 is None:
            x2_ = x1_
        else:
            x2_ = np.divide(x2, lengthscale*np.sqrt(2))

        return rbf_kernel(x1_, Y=x2_, gamma=1)


class GridKernel(BaseKernel):

    def __init__(self,  base_kernel, grid, dtype, active_dims=None):
        super().__init__(active_dims=active_dims)

        self.base_kernel = base_kernel
        self.num_dims = len(grid)

        self._grid = grid
        self.dtype = dtype
        self._sub_matrices = []
        self._sizes = [_grid[2] for _grid in grid]

        self._has_build = False

    @property
    def has_build(self):
        return self._has_build

    @has_build.setter
    def has_build(self, value):
        self._has_build = value

    @property
    def shape(self):
        return np.prod(self._sizes), np.prod(self._sizes)

    @property
    def grid(self):
        return self._grid

    def forward(self, x1, x2=None, diag=False, **params):

        is_kron = params.get('is_kron', False)

        if not self.has_build:
            self.build()

        if is_kron:
            return fm.Kron(*self._sub_matrices[::-1])
        else:
            return self._sub_matrices

    def build(self):

        if self.has_build:
            return

        grid = grid_coords(self.grid)

        for dim in range(self.num_dims):
            ith_row = self.base_kernel.forward(grid[dim].reshape(-1, 1), grid[dim][0].reshape(-1, 1), active_dim=dim)
            self._sub_matrices += fm.Toeplitz(ith_row, ith_row[1:]),

        self.has_build = True


class GridInterpolationKernel(GridKernel):

    def __init__(self, base_kernel, grid, dtype,  active_dims=None, num_dims=None):

        self.num_dims = num_dims
        super().__init__(
            base_kernel=base_kernel, grid=grid, dtype=dtype, active_dims=active_dims)

    def _inducing_forward(self, **params):
        return super().forward(self.grid, self.grid, **params)

    def forward(self, x1, x2=None, diag=False, **params):

        is_kmm = params.get('is_kmm', False)
        if is_kmm:
            kmm = self._inducing_forward(**params)
        else:
            kmm = None

        if x1 is not None:
            w_x1 = get_basis(x1, self.grid)         # Interpolation points for x1
        else:
            w_x1 = None

        if x2 is not None:
            w_x2 = get_basis(x2, self.grid)     # Interpolation points for x2
        else:
            w_x2 = None

        return w_x1, kmm, w_x2


class SpectralMixtureKernel(BaseKernel):
    is_stationary = True  # kernel is stationary even though it does not have a lengthscale

    def __init__(
            self,
            num_mixtures=None,
            ard_num_dims=1,
            active_dims=None,
            use_default_value_one=False,
            **kwargs,
    ):
        if num_mixtures is None:
            raise RuntimeError("num_mixtures is a required argument")

        # We will set to has_lengthscale=False, as we want to control scales here explicitly.
        super().__init__(active_dims=active_dims)

        self.num_mixtures = num_mixtures
        num_dims = ard_num_dims

        # Initial_value in constraint dominates over value in register parameters
        if use_default_value_one:
            default_value = DEFAULT_SOFTPLUS_VALUE
        else:
            default_value = 0

        # The choice of these values enforces that we utilize some information from data
        weights_initial_value = np.ones(num_mixtures)*default_value
        means_initial_value = np.ones((num_mixtures, num_dims))*default_value
        scales_initial_value = np.ones((num_mixtures, num_dims))*default_value

        self.register_parameter(
            name="mixture_weights",
            constraint=Positive(initial_value=weights_initial_value, transform='softplus')
        )

        self.register_parameter(
            name="mixture_means",
            constraint=Positive(initial_value=means_initial_value, transform='softplus')
        )

        self.register_parameter(
            name="mixture_scales",
            constraint=Positive(initial_value=scales_initial_value, transform='softplus')
        )

    @property
    def mixture_scales(self):
        return self.get_parameter(name='mixture_scales')

    @mixture_scales.setter
    def mixture_scales(self, value):
        self.set_parameter('mixture_scales', value)

    @property
    def mixture_means(self):
        return self.get_parameter(name='mixture_means')

    @mixture_means.setter
    def mixture_means(self, value):
        self.set_parameter('mixture_means', value)

    @property
    def mixture_weights(self):
        return self.get_parameter(name='mixture_weights')

    @mixture_weights.setter
    def mixture_weights(self, value):
        self.set_parameter('mixture_weights', value)

    def initialize_from_data_empspect(self, train_x, train_y):
        raise NotImplementedError

    def initialize_from_data(self, train_x, train_y, **kwargs):
        raise NotImplementedError

    def forward(self, x1, x2=None, diag=False, **params):

        if x2 is None:
            x2 = x1

        active_dim = params.get('active_dim', None)

        if active_dim is None:

            m = self.num_mixtures

            n1, d1 = x1.shape
            n2, d2 = x2.shape
            d = d1
            assert d1 == d2

            mixture_scales = self.mixture_scales
            mixture_scales = mixture_scales.reshape(m, 1, d)
            mixture_means = self.mixture_means
            mixture_means = mixture_means.reshape(m, 1, d)
        else:

            m = self.num_mixtures

            n1, d1 = x1.shape
            n2, d2 = x2.shape
            d = d1
            assert d1 == d2 and d == 1

            mixture_scales = self.mixture_scales[:, active_dim]
            mixture_scales = mixture_scales.reshape(m, 1, d)
            mixture_means = self.mixture_means[:, active_dim]
            mixture_means = mixture_means.reshape(m, 1, d)

        # computing the kernel
        x1_exp = x1 * mixture_scales  # (m, n1, d)
        x2_exp = x2 * mixture_scales  # (m, n2, d)
        x1_cos = x1 * mixture_means  # (m, n1, d)
        x2_cos = x2 * mixture_means  # (m, n2, d)

        x1_exp = x1_exp.reshape(m, n1, 1, d)  # (m, n1, 1, d)
        x2_exp = x2_exp.reshape(m, 1, n2, d)  # (m, 1, n2, d)

        x1_cos = x1_cos.reshape(m, n1, 1, d)  # (m, n1, 1, d)
        x2_cos = x2_cos.reshape(m, 1, n2, d)  # (m, 1, n2, d)

        exp_term = ((x1_exp - x2_exp) ** 2) * (-2 * math.pi ** 2)  # (m, n1, n2, d)
        cos_term = (x1_cos - x2_cos) * (2 * math.pi)  # (m, n1, n2, d)
        res = np.exp(exp_term) * np.cos(cos_term)  # (m, n1, n2, d)

        res = np.prod(res, axis=-1)  # (m, n1, n2)

        res = np.einsum('i,ijk->jk', self.mixture_weights, res)  # (n1, n2)

        return res
