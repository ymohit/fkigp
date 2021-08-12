import os
import yaml
import numpy as np

from enum import Enum
from abc import abstractmethod, ABCMeta


# Experiment level configuration
# Either set PRJ environment variable here
# OR comment below line and set in bash environment
os.environ['PRJ'] = os.environ['HOME'] + '/skigp/'
PRJ_PATH = os.environ['PRJ']
LOG_DIR = os.environ['PRJ'] + "logs/default"

PRINT_FREQ = 5
DEFAULT_GRID_RATE = 0.5
NUM_ITERATIONS = 500
DEFAULT_LANCZOS_RANK = 30


def get_hypers(data_type='sine', options=None):
    file_path = os.environ['PRJ'] + 'configs/hypers.yaml'
    with open(file_path, 'r') as outfile:
        data = yaml.safe_load(outfile)

    if isinstance(data_type, DatasetType):
        return data[data_type.name.lower()]
    else:
        return data[data_type]


class Structdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class BasicConfig(Structdict):

    def __init__(self):
        super(BasicConfig, self).__init__()
        for var_name in self.list_config_vars():
            self.__setattr__(var_name, getattr(self, var_name))

    def list_config_vars(self):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return members


class DataReader(object):

    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.config = config
        self.features = None

    @abstractmethod
    def read_data(self, *args, **kwargs):
        raise NotImplementedError


class MethodName(Enum):
    NAIVEGP = 0
    KISSGP = 1
    GSGP = 2


class GsGPType(Enum):
    ASYM = 0    # Asymmetric expressions using BiCG and similar solvers
    SYM = 1     # Symmetric expressions using CG and similar solvers
    FACT = 2    # KissGP expressions using Factorized solvers


class Frameworks(Enum):
    GPYTORCH = 0
    NUMPY = 1


class DatasetType(Enum):
    SINE = 0
    SOLAR = 1
    SOUND = 2
    AIRLINE = 3
    SYNGP = 4
    SYNGPND = 5
    PRECIPITATION = 6
    RADAR = 7


class GridSizeFunc(Enum):
    NOT_SUPPLIED = -1
    N_BY_2 = 0
    N_BY_4 = 1
    N_BY_16 = 2
    SQRT_N = 3
    CONST_N = 4
    N = 5


class ExperimentType(Enum):
    MEAN_INF = 0
    LOGDET = 1
    SKI_COV = 2
    APPROX_COV = 3


def get_grid_size(num_points, grid_size_f, data_type=None):

    if grid_size_f == GridSizeFunc.N:
        return num_points
    elif grid_size_f == GridSizeFunc.N_BY_2:
        return num_points//2
    elif grid_size_f == GridSizeFunc.N_BY_4:
        return num_points//4
    elif grid_size_f == GridSizeFunc.N_BY_16:
        return num_points // 16
    elif grid_size_f == GridSizeFunc.SQRT_N:
        return int(np.sqrt(num_points))
    elif grid_size_f == GridSizeFunc.CONST_N:
        assert data_type is not None
        if data_type == DatasetType.SOUND:
            return 8000
        elif data_type == DatasetType.SINE:
            return 500
        elif data_type == DatasetType.SYNGP:
            return 1000


def get_precip_grid(idx=1):
    config_path = os.environ['PRJ'] + 'configs/precipitation_grids.yaml'
    return yaml.safe_load(open(config_path))[idx]


def get_radar_grid(idx=1):
    config_path = os.environ['PRJ'] + 'configs/radar_grids.yaml'
    return yaml.load(open(config_path))[idx]

