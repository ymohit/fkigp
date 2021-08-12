import numpy as np

from fkigp.configs import Structdict
from fkigp.configs import DatasetType
from fkigp.configs import LOG_DIR
from fkigp.configs import NUM_ITERATIONS
from fkigp.configs import DEFAULT_GRID_RATE

from matplotlib import pyplot as plt


class Experiment(object):

    def __init__(self, config=None):

        # Configuring experiment
        self.config = config if config is not None else Structdict()
        self.configure()

        # Placeholders
        self.model = None
        self.likelihood = None
        self.data_loader = None
        self.train_y = None
        self.train_x = None
        self.test_y = None
        self.test_x = None
        self.grid = None
        self.test_grid = None

        self.WT = None
        self.grid_bounds = None

    def configure(self):
        # Add default config parameters
        if 'grid_rate' not in self.config:
            self.config['grid_rate'] = DEFAULT_GRID_RATE
        if 'num_iterations' not in self.config:
            self.config['num_iterations'] = NUM_ITERATIONS
        if 'log_dir' not in self.config:
            self.config['log_dir'] = LOG_DIR

    def sample_data(self, **params):
        self.data_loader.sample_data(**params)

    def load_data(self, **params):
        self.train_x, self.train_y, self.test_x, self.test_y = self.data_loader.get_data(**params)

    def build(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def fit(self, verbose=True):
        raise NotImplementedError

    def compute_grid(self, grid_bounds, len_train=None):

        grid_size = self.config.get('grid_size', None)
        if grid_size is not None and isinstance(grid_size, int) and grid_size <= 0:
            grid_size = None

        num_dims = self.config.num_dims
        grid_rate = self.config.grid_rate

        # At-least one among grid-rate or grid-size should be provided
        if grid_size is None and grid_rate is None:
            print("At-least one of [grid_rate, grid_size is required to create grid kernel.]")
            raise NotImplementedError

        # Building up grid bounds
        if grid_bounds is None:
            if num_dims is None:
                raise RuntimeError("num_dims must be supplied if grid_bounds is None")
            else:
                # Create some temporary grid bounds - they'll be changed soon
                grid_bounds = tuple((-1.0, 1.0) for _ in range(num_dims))
        else:
            assert len(grid_bounds) == num_dims, "Grid bounds are not matching number of dimensions!"

        # Registering grid size
        if grid_size is None:
            assert len_train is not None or self.train_x is not None
            len_train = len_train if len_train is not None else len(self.train_x)
            grid_size = len_train * grid_rate

        # Processing grid sizes further
        if isinstance(grid_size, int) or isinstance(grid_size, float):
            grid_size = int(grid_size)
            grid_sizes = [int(np.ceil(grid_size ** (1/num_dims))) for _ in range(num_dims)]
        else:  # case when grid sizes are provided for all dimensions separately
            grid_sizes = list(grid_size)

        if len(grid_sizes) != num_dims:
            raise RuntimeError("The number of grid sizes provided through grid_size do not match num_dims.")

        self.config.grid_sizes = grid_sizes
        return [(gb[0], gb[1], gs) for gb, gs in zip(grid_bounds, grid_sizes)]

    def print_model_params(self, raw=True):
        for param_name, param in self.model.get_parameters(raw=raw).items():
            print(f'Parameter name: {param_name:42} value = {param}')

    def init_params(self, raw_noise=None, raw_outputscale=None, raw_lengthscale=None, hypers=None):

        # Initializing using dictionary
        if hypers is not None:
            assert type(hypers) == dict
            self.model.initialize(**hypers)
            return

        # When hyper-parameters are provided
        if raw_noise is not None:
            assert raw_outputscale is not None
            assert raw_lengthscale is not None

            hypers = {
                'noise_covar.raw_noise': raw_noise,
                'covar_module.base_kernel.raw_outputscale': raw_outputscale,
                'covar_module.base_kernel.base_kernel.raw_lengthscale': raw_lengthscale,
            }
            self.model.initialize(**hypers)
            return

        # Setting default for data-specific initialization
        if self.config.data_type == DatasetType.SOUND:
            hypers = {
                'noise_covar.raw_noise': -4.89,
                'covar_module.base_kernel.raw_outputscale': -3.14,
                'covar_module.base_kernel.base_kernel.raw_lengthscale': 2.52,
            }
            self.model.initialize(**hypers)
        else:
            pass  # Use default parameters

    def perform_mean_prediction(self, **kwargs):
        assert self.test_x is not None
        test_x = self.test_x
        mean, __, __ = self.model.predict(test_x, **kwargs)
        return mean

    def predict(self, test_x=None, **kwargs):

        assert test_x is not None
        mean, lower, upper = self.model.predict(test_x, **kwargs)
        return mean, None, None

    def compute_mae(self, **kwargs):
        test_y = self.test_y.reshape(-1, 1)
        predict_y = self.predict(self.test_x, **kwargs)[0].reshape(-1, 1)
        mae = np.sum(np.abs(test_y - predict_y)) / len(test_y)
        return mae

    def compute_smae(self, **kwargs):
        mae = self.compute_mae(**kwargs)
        trivmae = np.sum(np.abs(self.test_y - np.zeros_like(self.test_y))) / len(self.test_y)
        return mae / trivmae

    def compute_rmse(self, **kwargs):
        test_y = self.test_y
        predict_y = self.predict(self.test_x, **kwargs)[0]
        rmse = np.sqrt(np.sum(np.abs(test_y.reshape(-1, 1) - predict_y.reshape(-1, 1)) ** 2) / len(test_y))
        return rmse

    def report(self, fill=True, title="KissGP inference"):

        if self.config.data_type == DatasetType.SINE:

            mean, lower, upper = self.predict(self.test_x)

            train_x = self.train_x
            train_y = self.train_y
            test_x = self.test_x

            f, ax = plt.subplots(1, 1, figsize=(10, 6))

            # Plot the training data as black stars
            ax.plot(train_x, train_y, 'k*')

            # Plot predictive means as blue line
            ax.plot(test_x, mean, 'b')

            # Plot confidence bounds as lightly shaded region
            if fill:
                ax.fill_between(test_x, lower, upper, alpha=0.5)

            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])

            print("RMSE: ", self.compute_rmse())
            print("Final MAE: ", self.compute_mae())
            if title:
                plt.title(title)
            plt.show()

        elif self.config.data_type == DatasetType.SOLAR:
            raise NotImplementedError
        elif self.config.data_type == DatasetType.SOUND:
            print('Test MAE: ', self.compute_mae())
            print('Test SMAE: ', self.compute_smae())

        elif self.config.data_type == DatasetType.PRECIPITATION:
            print('Test SMAE: ', self.compute_smae())

        elif self.config.data_type == DatasetType.SYNGP:
            pass
        else:
            raise NotImplementedError
