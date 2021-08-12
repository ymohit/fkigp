import fastmat as fm
import fkigp.gps.kernels as kernels
import fkigp.configs as configs

from fkigp.gridutils import get_basis
from fkigp.experiment import Experiment
from fkigp.gps.gpbase import GpModel, NoiseCovar
from fkigp.configs import GsGPType, MethodName
from fkigp.datautils import get_num_dims, DatasetType


class DefaultGsGpRegressionModel(GpModel):

    def __init__(self, WT_times_W, WT_times_Y, YT_times_Y, grid, data_type, gsgp_type,
                 ard_num_dims=None, zero_mean=False, num_dims=1):

        super().__init__()

        self.method_name = MethodName.GSGP

        self.WT_times_W = WT_times_W
        self.WT_times_Y = WT_times_Y
        self.YT_times_Y = YT_times_Y

        self.data_type = data_type
        self.gsgp_type = gsgp_type
        self.num_dims = num_dims

        if zero_mean:
            pass
        else:
            raise NotImplementedError

        if ard_num_dims is None:
            base_kernel = kernels.ScaleKernel(kernels.RBFKernel())
        else:
            base_kernel = kernels.ScaleKernel(kernels.RBFKernel(ard_num_dims=ard_num_dims))

        self.covar_module = kernels.GridInterpolationKernel(
            base_kernel=base_kernel,
            grid=grid,
            dtype=self.WT_times_Y.dtype,
            num_dims=num_dims)

        self.noise_covar = NoiseCovar(num_dims=WT_times_W.shape[0])
        self.prediction_strategy = None

    def predict(self, X, **kwargs):
        self.training = False
        """
        :param X:
        :return:
        """
        return super().__call__(X, **kwargs)


class PrecipGsGpRegressionModel(GpModel):

    def __init__(self, WT_times_W, WT_times_Y, YT_times_Y, grid, gsgp_type,
                 num_dims=3):

        super().__init__()

        self.method_name = MethodName.GSGP

        self.WT_times_W = WT_times_W
        self.WT_times_Y = WT_times_Y
        self.YT_times_Y = YT_times_Y

        self.data_type = DatasetType.PRECIPITATION
        self.gsgp_type = gsgp_type
        self.num_dims = num_dims

        base_kernel = kernels.ScaleKernel(kernels.RBFKernel(ard_num_dims=3))

        self.covar_module = kernels.GridInterpolationKernel(
            base_kernel=base_kernel,
            grid=grid,
            dtype=self.WT_times_Y.dtype,
            num_dims=num_dims)

        self.noise_covar = NoiseCovar(num_dims=WT_times_W.shape[0])
        self.prediction_strategy = None

    def predict(self, X, **kwargs):
        self.training = False
        """
        :param X:
        :return:
        """
        mean, lower, upper = super().__call__(X, **kwargs)
        return 9.641058973572967 + mean, lower, upper


class GsGpExp(Experiment):

    def save(self):
        pass

    def load(self):
        pass

    def fit(self, verbose=True):
        pass

    def __init__(self, config=None, data_loader=None):

        super().__init__(config=config)

        self.noise_model = None
        self.grid = None
        self.grid_bounds = None

        self.data_loader = data_loader
        self.configure()

    def configure(self):
        super().configure()
        if 'num_dims' not in self.config:
            self.config['num_dims'] = get_num_dims(self.config.data_type)
        if 'gsgp_type' not in self.config:
            self.config['gsgp_type'] = GsGPType.FACT

        if self.config['data_type'] == DatasetType.SOUND:
            if 'grid_size' not in self.config:
                self.config['grid_size'] = 8000
            self.config['num_dims'] = 1
            self.config['num_iterations'] = 100

        if self.config['data_type'] == DatasetType.PRECIPITATION:

            if 'grid_idx' in self.config and self.config['grid_idx'] > 0:
                self.config['grid_size'] = configs.get_precip_grid(self.config['grid_idx'])

            elif 'grid_size' not in self.config:
                self.config['grid_size'] = [35, 35, 100]
            self.config['num_dims'] = 3

        self.config['zero_mean'] = True

        return

    def load_data(self, **params):
        """
        This function loads necessary projections of the training data and test data as it as.
        :return:
        """

        if self.config.data_type == DatasetType.RADAR:
            WT_times_W, WT_times_Y, YT_times_Y, n = self.data_loader.get_data()

        else:
            # Loading raw data
            train_x, train_y, self.test_x, self.test_y = self.data_loader.get_data(**params)

            # Saving data for small datasets
            if self.config.data_type in [DatasetType.SINE, DatasetType.SOLAR, DatasetType.SYNGP, DatasetType.SYNGPND]:
                self.train_y = train_y
                self.train_x = train_x

            if self.config['data_type'] == DatasetType.PRECIPITATION:
                train_y = train_y - 9.641058973572967

            # Computing grid_bounds
            grid_bounds = self.config.get('grid_bounds', None)

            if grid_bounds is None:
                if self.config.data_type == DatasetType.SOUND:
                    grid_bounds = ((1, 60000),)
                elif self.config.data_type == DatasetType.SOLAR:
                    grid_bounds = ((self.train_x.min(), self.train_x.max()),)
                elif self.config.data_type == DatasetType.PRECIPITATION:
                    grid_bounds = ((-123.30945000000001, -68.45191188895595),
                                   (24.309450000000005, 49.48939351237175),
                                   (25312.32, 26191.32))
                else:
                    if grid_bounds is None:
                        grid_bounds = tuple((-1.0, 1.0) for _ in range(self.config.num_dims))
            assert grid_bounds is not None

            self.grid_bounds = grid_bounds

            # Computing grids
            grid = self.compute_grid(grid_bounds, len(train_x))
            self.grid = grid
            W = get_basis(train_x, grid)

            WT_times_W = fm.Sparse((W.T * W).tocsr())
            WT_times_Y = W.T * train_y
            YT_times_Y = train_y.T @ train_y

            if self.config.data_type in [DatasetType.SOUND, DatasetType.PRECIPITATION]:
                self.WT = W.T.tocsr()

        # Building the regression model
        if self.config.data_type == DatasetType.PRECIPITATION:
            self.model = PrecipGsGpRegressionModel(WT_times_W=WT_times_W,
                                                   WT_times_Y=WT_times_Y,
                                                   YT_times_Y=YT_times_Y,
                                                   grid=self.grid,
                                                   gsgp_type=self.config.gsgp_type)

        else:
            self.model = DefaultGsGpRegressionModel(WT_times_W=WT_times_W,
                                                    WT_times_Y=WT_times_Y,
                                                    YT_times_Y=YT_times_Y,
                                                    grid=self.grid,
                                                    gsgp_type=self.config.gsgp_type,
                                                    data_type=self.config.data_type,
                                                    zero_mean=self.config.zero_mean,
                                                    num_dims=self.config.num_dims)

    def build(self, verbose=True):

        if not verbose:
            return

        print("\n#### Model description:")
        print("Grid sizes:", self.config.grid_sizes)
        print("Num dims:", self.model.covar_module.num_dims)
        print("Grid bounds:", self.grid_bounds)
        print("####\n")
