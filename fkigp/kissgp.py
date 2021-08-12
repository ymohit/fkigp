import numpy as np
import fkigp.configs as configs

from fkigp.configs import MethodName
from fkigp.datautils import DatasetType
from fkigp.datautils import get_num_dims

from fkigp.gps.gpbase import GpModel, NoiseCovar

from fkigp.experiment import Experiment
from fkigp.configs import DEFAULT_GRID_RATE as GRID_RATE


import fkigp.gps.kernels as kernels
from fkigp.gps.means import ConstantMean
from fkigp.gridutils import get_basis


class DefaultKissGpRegressionModel(GpModel):

    def __init__(self, W, train_y, grid, zero_mean=True, num_dims=1):

        super().__init__()

        self.method_name = MethodName.KISSGP
        self.W = W
        self.train_y = train_y
        self.num_dims = num_dims

        if zero_mean:
            pass
        else:
            raise NotImplementedError

        self.covar_module = kernels.GridInterpolationKernel(
            base_kernel=kernels.ScaleKernel(kernels.RBFKernel(ard_num_dims=num_dims)),
            grid=grid,
            dtype=self.train_y.dtype,
            num_dims=num_dims)

        self.noise_covar = NoiseCovar(num_dims=self.W.shape[0])
        self.prediction_strategy = None

    def predict(self, X, **kwargs):
        self.training = False
        """
        :param X:
        :return:
        """
        return super().__call__(X, **kwargs)


class PrecipitationKissGpRegressionModel(GpModel):

    def __init__(self, W, train_x, train_y, grid, num_dims=3):

        super().__init__()

        self.method_name = MethodName.KISSGP
        self.W = W
        self.train_x = train_x
        self.train_y = train_y
        self.num_dims = num_dims

        self.mean_module = ConstantMean(value=9.641058973572967)

        self.covar_module = kernels.GridInterpolationKernel(
            base_kernel=kernels.ScaleKernel(kernels.RBFKernel(ard_num_dims=self.num_dims)),
            grid=grid,
            dtype=self.train_x.dtype,
            num_dims=num_dims)

        self.noise_covar = NoiseCovar(num_dims=self.train_x.shape[0])
        self.prediction_strategy = None

    def predict(self, X, **kwargs):

        self.training = False
        """
        :param X:
        :return:
        """

        mean, lower, upper = super().__call__(X, **kwargs)
        return self.mean_module.value + mean, lower, upper


class KissGpExp(Experiment):

    def __init__(self, config=None, data_loader=None):

        assert data_loader is not None, "Data_loader is None!"

        super().__init__(config=config)
        self.data_loader = data_loader
        self.configure()

    def configure(self):

        super().configure()

        if 'num_dims' not in self.config:
            self.config['num_dims'] = get_num_dims(self.config.data_type)

        if 'grid_rate' not in self.config:
            self.config['grid_rate'] = GRID_RATE

        if 'zero_mean' not in self.config:
            self.config['zero_mean'] = True

        if self.config['data_type'] == DatasetType.SOUND:

            if 'grid_size' not in self.config:
                self.config['grid_size'] = 8000
            self.config['num_dims'] = 1

            if 'num_iterations' not in self.config:
                self.config['num_iterations'] = 200
            self.config['zero_mean'] = True

        elif self.config['data_type'] == DatasetType.RADAR:

            if 'grid_size' not in self.config:
                self.config['grid_size'] = [100, 100, 6]

            if 'grid_bounds' not in self.config:
                self.config['grid_bounds'] = ((36, 48), (-83, -64), (10, 3010),)

            self.config['num_dims'] = 3

        elif self.config['data_type'] == DatasetType.PRECIPITATION:
            if 'grid_idx' in self.config and self.config['grid_idx'] > 0:
                self.config['grid_size'] = configs.get_precip_grid(self.config['grid_idx'])

            elif 'grid_size' not in self.config:
                self.config['grid_size'] = [35, 35, 100]
            self.config['num_dims'] = 3

            self.config['num_dims'] = 3

    def load_data(self, **params):

        self.train_x, self.train_y, self.test_x, self.test_y = self.data_loader.get_data(**params)

        # Computing grid_bounds
        grid_bounds = self.config.get('grid_bounds', None)

        if grid_bounds is None:

            if self.config.data_type == DatasetType.SOUND:
                grid_bounds = ((1, 60000),)

            elif self.config.data_type == DatasetType.SOLAR:
                grid_bounds = ((self.train_x.min(), self.train_x.max()),)

            elif self.config.data_type == DatasetType.PRECIPITATION:
                min_ = np.min(self.train_x, axis=0)
                max_ = np.max(self.train_x, axis=0)
                grid_bounds = tuple((min_[i] - 0.01*min_[i], max_[i] + 0.01*max_[i])
                                    for i in range(self.config.num_dims))
            else:
                if grid_bounds is None:
                    grid_bounds = tuple((-1.0, 1.0) for _ in range(self.config.num_dims))

        self.grid_bounds = grid_bounds
        assert self.grid_bounds is not None

        # Computing grid parameters
        grid = self.compute_grid(grid_bounds=grid_bounds)
        self.grid = grid
        W = get_basis(self.train_x, grid)

        # Building model
        if self.config.data_type == DatasetType.SOUND:

            self.model = DefaultKissGpRegressionModel(
                W,
                self.train_y,
                grid=grid,
                zero_mean=self.config.zero_mean)

        elif self.config.data_type == DatasetType.PRECIPITATION:
            grid = self.compute_grid(grid_bounds=grid_bounds)
            self.model = PrecipitationKissGpRegressionModel(
                W,
                self.train_x,
                self.train_y,
                grid=grid)

        else:
            grid = self.compute_grid(grid_bounds)
            self.model = DefaultKissGpRegressionModel(
                W,
                self.train_y,
                zero_mean=self.config.zero_mean,
                grid=grid,
                num_dims=self.config.num_dims)

    def build(self, verbose=True):

        if not verbose:
            return

        print("\n#### Model description:")
        print("Grid sizes:", self.config.grid_sizes)
        print("Num dims:", self.model.covar_module.num_dims)
        print("Grid bounds:", self.grid_bounds)
        print("####\n")
