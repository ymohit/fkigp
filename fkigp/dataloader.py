import os
import math
import pyart
import numpy as np
import scipy.io as sio

from fkigp.utils import tic, toc
from fkigp.configs import Structdict, DatasetType, MethodName
from fkigp.utils import NUMPY_DTYPE as DEFAULT_NUMPY_DTYPE

from fkigp.datautils import get_suff_stats, rbf_covND, sample, idb

from fkigp.gridutils import grid_points


# Data level configuration - default values
ONE_DIM_NUM_POINTS = 100
ONE_DIM_NOISE_LEVEL = 0.25
ONE_DIM_NUM_CYCLES = 1

DATA_PATHS = {
    DatasetType.SOLAR: os.environ['PRJ'] + '/data/solar_data.txt',
    DatasetType.AIRLINE: os.environ['PRJ'] + '/data/airline_dumps/',
    DatasetType.SOUND: os.environ['PRJ'] + '/data/audio_data.mat',
    DatasetType.SYNGP: os.environ['PRJ'] + '/data/syngp/',
    DatasetType.PRECIPITATION: os.environ['PRJ'] + '/data/precip_data.mat',
    DatasetType.RADAR: os.environ['PRJ'] + '/data/radar_files'
}

DEFAULT_DATA_TYPE = DatasetType.SOLAR
DEFAULT_AIRLINE_SAMPLES = 10000


class DataLoader(object):

    def __init__(self, config=None):

        self.config = config if config is not None else Structdict()

        # Add default config parameters
        if 'one_dim_num_points' not in self.config:
            self.config['one_dim_num_points'] = ONE_DIM_NUM_POINTS
        if 'one_dim_noise_level' not in self.config:
            self.config['one_dim_noise_level'] = ONE_DIM_NOISE_LEVEL
        if 'data_type' not in self.config:
            self.config['data_type'] = DEFAULT_DATA_TYPE
        if 'one_dim_num_cycles' not in self.config:
            self.config['one_dim_num_cycles'] = ONE_DIM_NUM_CYCLES
        if 'num_dims' not in self.config:
            self.config['num_dims'] = 1

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.test_grid = None

        self.sample = False

    def get_data(self, data_type=None, cov=None, **params):

        config = self.config
        if data_type is None:
            data_type = config.data_type

        if data_type == DatasetType.RADAR:
            return self.sample_data(data_type=data_type, cov=cov, **params)

        if not self.sample:
            self.sample_data(data_type=data_type, cov=cov, **params)

        return self.train_x, self.train_y, self.test_x, self.test_y

    def sample_data(self, data_type=None, cov=None, **params):

        self.sample = True
        config = self.config

        if data_type is None:
            data_type = config.data_type

        if data_type == DatasetType.SINE:

            sigma = params.get('sigma',  config.one_dim_noise_level)
            xmin, xmax = -1, 1

            X = np.random.uniform(xmin, xmax, (config.one_dim_num_points))
            X = np.sort(X)
            Y = np.sin(X * (2 * config.one_dim_num_cycles * math.pi)
                       + np.random.rand(config.one_dim_num_points) * sigma - sigma/2)
            X_test = np.linspace(np.min(X), np.max(X), 30)
            Y_test = np.sin(X_test * (2 * config.one_dim_num_cycles * math.pi))

        elif data_type == DatasetType.SYNGPND:

            # Processing params specific to n-dimensional data
            ndim = params.get('ndim', config.num_dims)
            gamma = params.get('gamma', 1.0)
            sigma = params.get('sigma', config.one_dim_noise_level)
            ntest = params.get('ntest', 500)
            xmin, xmax = -5, 5

            n = params.get('N', config.one_dim_num_points)

            if cov is None:
                cov = rbf_covND

            # randomly sampled train points
            X = np.random.uniform(xmin, xmax, (n, ndim))

            # test points
            test_dim = int(np.ceil(ntest ** (1 / ndim)))
            test_grid = [(xmin, xmax, test_dim), (xmin, xmax, test_dim)]
            X_test = grid_points(test_grid)

            # draw random function on combined set of points
            XX = np.concatenate([X, X_test])
            ff = sample(XX, cov, gamma=gamma)

            # extract values for different sets
            f_train = ff[:n]
            Y_test = ff[n:]

            Y = f_train + np.random.normal(0, sigma, (n))

            self.test_grid = test_grid

        elif data_type == DatasetType.SOLAR:

            data = np.genfromtxt(DATA_PATHS[data_type], delimiter=',')
            X = data[:, 0:1]
            Y = data[:, 2:3]
            Y = (Y - Y.mean()) / Y.std()

            # remove some chunks of data
            X_test, Y_test = [], []

            intervals = ((1620, 1650), (1700, 1720), (1780, 1800), (1850, 1870), (1930, 1950))
            for low, up in intervals:
                ind = np.logical_and(X.flatten() > low, X.flatten() < up)
                X_test.append(X[ind])
                Y_test.append(Y[ind])
                X = np.delete(X, np.where(ind)[0], axis=0)
                Y = np.delete(Y, np.where(ind)[0], axis=0)

            X_test, Y_test = np.vstack(X_test), np.vstack(Y_test)

            X = X.squeeze()
            Y = Y.squeeze()
            X_test = X_test.squeeze()
            Y_test = Y_test.squeeze()

        elif data_type == DatasetType.SOUND:

            data = sio.loadmat(DATA_PATHS[data_type])
            X = data['xtrain'].squeeze().astype(DEFAULT_NUMPY_DTYPE)
            Y = data['ytrain'].squeeze().astype(DEFAULT_NUMPY_DTYPE)
            X_test = data['xtest'].squeeze().astype(DEFAULT_NUMPY_DTYPE)
            Y_test = data['ytest'].squeeze().astype(DEFAULT_NUMPY_DTYPE)

        elif data_type == DatasetType.PRECIPITATION:

            fpath = DATA_PATHS[DatasetType.PRECIPITATION]
            data = sio.loadmat(fpath)
            X, Y, X_test, Y_test = data['X'], data['y'], data['Xtest'], data['ytest']
            Y_test = np.array(Y_test).astype(np.float).squeeze()
            Y = np.array(Y).astype(np.float).squeeze()
            X_test = X_test.astype(np.float)
            X = X.astype(np.float)

        elif data_type == DatasetType.RADAR:
            raise NotImplementedError('See experiments/radar_processing as radar is handled as special case.')
        else:
            raise NotImplementedError

        self.train_x, self.train_y, self.test_x, self.test_y = X, Y, X_test, Y_test
        return


