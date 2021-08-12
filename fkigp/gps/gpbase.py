import numpy as np
import fastmat as fm
from abc import abstractmethod

from fkigp.utils import ndimension
from fkigp.configs import MethodName

from fkigp.gps.constraints import inv_softplus, Positive
from fkigp.gps.constraints import NoConstraint, DEFAULT_SOFTPLUS_VALUE

from fkigp.gps.prediction import KissGpPredictionStrategy
from fkigp.gps.prediction import GsGpPredictionStrategy


def get_all_parameters(module_handle, params, prefix='', raw=True):

    modules = [name for name in dir(module_handle) if not name.startswith('_')
               and name != 'get_parameters'
               and 'lengthscale' not in name
               and 'raw_noise' not in name
               and 'noise' != name]

    modules = [name for name in modules if hasattr(getattr(module_handle, name), 'parameters')]

    for module_name in modules:
        # Collecting all params within a particular BaseModule
        for param, constraint in getattr(module_handle, module_name).parameters.items():

            if prefix == '':
                param_name = module_name + "." + param
            else:
                param_name = prefix + "." + module_name + "." + param

            if not raw:
                params[param_name] = constraint.value
            else:
                params[param_name] = constraint.raw_value

        # Recursively calling for all possible sub BaseModules
        if prefix == '':
            module_prefix = module_name
        else:
            module_prefix = prefix + "." + module_name
        get_all_parameters(getattr(module_handle, module_name), params, prefix=module_prefix, raw=raw)


class BaseModule(object):

    def __init__(self):
        self._parameters = dict()

    def register_parameter(self, name, value=None, constraint=None):

        if value is None:
            value = DEFAULT_SOFTPLUS_VALUE

        if constraint is None:
            constraint = NoConstraint(initial_value=value)

        self._parameters[name] = constraint

    def get_parameter(self, name):
        return self._parameters[name].value

    def get_raw_parameter(self, name):
        return self._parameters[name].raw_value

    def set_parameter(self, name, value):

        # We always store raw value and also update only raw
        self._parameters[name]._value = self._parameters[name].inverse_transform(value)

    def set_raw_parameter(self, name, value):

        # We always store raw value and also update only raw
        self._parameters[name]._value = value

    @property
    def parameters(self):
        return self._parameters


class BaseKernel(BaseModule):
    """
    Base Args:
        :attr:`has_lengthscale` (bool):
            Set this if the kernel has a lengthscale. Default: `False`.
        :attr:`ard_num_dims` (int, optional):
            Set this if you want a separate lengthscale for each input
            dimension. It should be `d` if :attr:`x1` is a `n x d` matrix.  Default: `None`
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.
    """
    def __init__(
            self,
            has_lengthscale=False,
            lengthscale_constraint=None,
            ard_num_dims=None,
            active_dims=None,
            eps=1e-6,
    ):

        # super(BaseKernel, self).__init__()
        super().__init__()

        self.active_dims = active_dims
        self.ard_num_dims = ard_num_dims
        self.eps = eps
        self.__has_lengthscale = has_lengthscale

        if has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims

            if lengthscale_constraint is None:
                lengthscale_constraint = Positive(initial_value=DEFAULT_SOFTPLUS_VALUE*np.ones(lengthscale_num_dims),
                                                  transform='softplus')

            self.register_parameter(
                name="lengthscale",
                value=np.zeros(lengthscale_num_dims),
                constraint=lengthscale_constraint,
            )

    @abstractmethod
    def forward(self, x1, x2=None, diag=False, **params):
        raise NotImplementedError

    @property
    def has_lengthscale(self):
        return self.__has_lengthscale

    @property
    def lengthscale(self,):
        if self.has_lengthscale:
            return self.get_parameter(name='lengthscale')
        else:
            raise ValueError("lengthscale parameter does not exists!")

    @lengthscale.setter
    def lengthscale(self, value):
        self.set_parameter('lengthscale', value)

    @property
    def raw_lengthscale(self,):
        if self.has_lengthscale:
            return self.get_raw_parameter(name='lengthscale')
        else:
            raise ValueError("lengthscale parameter does not exists!")

    @raw_lengthscale.setter
    def raw_lengthscale(self, value):
        self.set_raw_parameter('lengthscale', value)

    def __call__(self, x1, x2=None, diag=False, **params):
        x1_, x2_ = x1, x2

        # Select the active dimensions
        if self.active_dims is not None:
            x1_ = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2_ = x2_.index_select(-1, self.active_dims)

        # Give x1_ and x2_ a last dimension, if necessary
        if ndimension(x1_) == 1:
            x1_ = x1_.reshape(-1, 1)

        if x2_ is not None:
            if ndimension(x2_) == 1:
                x2_ = x2_.reshape(-1, 1)
            if not x1_.shape[-1] == x2_.shape[-1]:
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")

        # Check that ard_num_dims matches the supplied number of dimensions
        if self.ard_num_dims is not None and self.ard_num_dims != x1_.shape[0]:
            raise RuntimeError(
                "Expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, x1_.shape[0])
            )

        if diag:
            raise NotImplementedError
        else:
            return self.forward(x1_, x2_, **params)


class GpModel(object):

    def __init__(self,):

        self.training = False
        self.method_name = None
        self.train_x = None
        self.train_y = None
        self.grid = None
        self.gsgp_type = None
        self.num_dims = None

        self.W = None
        self.WT_times_W = None
        self.WT_times_Y = None
        self.YT_times_Y = None

        # Place holders for modules
        self.mean_module = None
        self.covar_module = None
        self.noise_covar = None
        self.prediction_strategy = None

    def initialize(self, **params):

        for param_name,  param_value in params.items():
            param_name = param_name.split('.')
            module_handle = self
            for module_name in param_name[:-1]:
                module_handle = getattr(module_handle, module_name)

            if "raw" in param_name[-1]:
                param_name = param_name[-1][4:]
                module_handle.set_raw_parameter(param_name, param_value)
            else:
                module_handle.set_parameter(param_name[-1], param_value)

    def get_parameters(self, raw=True):
        params = {}
        get_all_parameters(self, params=params, raw=raw)
        return params

    def __call__(self, *args, **kwargs):

        grid_prediction = kwargs.get('grid', False)

        if not grid_prediction:
            inputs = [i.reshape(-1, 1) if ndimension(i) == 1 else i for i in args]
            assert inputs is not None

        # Training mode: optimizing
        if self.training:
            # return self.compute_loss()
            raise NotImplementedError

        # Posterior mode
        else:

            # Step1: Build prediction strategy if not initialized already
            if self.prediction_strategy is None:

                if self.method_name == MethodName.GSGP:

                    # Step 1: Build prediction strategy
                    __, Kmm, __ = self.covar_module.forward(x1=None, is_kmm=True)

                    self.prediction_strategy = GsGpPredictionStrategy(
                        Kmm=Kmm,
                        WT_times_W=self.WT_times_W,
                        WT_times_Y=self.WT_times_Y,
                        YT_times_Y=self.YT_times_Y,
                        sigma=self.noise_covar.noise,
                        gsgp_type=self.gsgp_type,
                        mean_value=self.mean_module.value if self.mean_module is not None else 0
                    )

                elif self.method_name == MethodName.KISSGP:

                    # Step 1: Build prediction strategy
                    __, Kmm, __ = self.covar_module.forward(x1=None, is_kmm=True)

                    self.prediction_strategy = KissGpPredictionStrategy(
                        W=self.W,
                        Kmm=Kmm,
                        sigma=self.noise_covar.noise,
                        train_y=self.train_y,
                        mean_value=self.mean_module.value if self.mean_module is not None else 0
                    )
                else:
                    raise NotImplementedError

            # Step 2: Make exact prediction on the test
            if grid_prediction:
                mean, lower, upper = self.prediction_strategy.exact_prediction(None, **kwargs)
                return mean, lower, upper

            W_test, __, __ = self.covar_module.forward(x1=inputs[0])
            mean, lower, upper = self.prediction_strategy.exact_prediction(W_test, **kwargs)
            return mean, lower, upper


class NoiseCovar(BaseModule):

    def __init__(self, num_dims, noise_value=None, noise_constraint=None):

        """
        This function return noise diagonal matrix of size [shape x shape].
        :param num_dims:
        :param noise_constraint:
        """
        # super(NoiseCovar, self).__init__()
        super().__init__()

        self.num_dims = num_dims
        if noise_constraint is None:
            if noise_value is not None:
                noise_value = inv_softplus(noise_value)
            else:
                noise_value = DEFAULT_SOFTPLUS_VALUE
            noise_constraint = Positive(initial_value=noise_value, transform='softplus')

        self.register_parameter(
            name="noise",
            constraint=noise_constraint,
        )

    @property
    def noise(self):
        return self.get_parameter(name='noise')

    @noise.setter
    def noise(self, value):
        self.set_parameter('noise', value)

    @property
    def raw_noise(self,):
        return self.get_raw_parameter(name='raw_noise')

    @raw_noise.setter
    def raw_noise(self, value):
        self.set_raw_parameter('raw_noise', value)

    def __call__(self, *args, **kwargs):
        shape = kwargs.get('shape', None)
        return self.forward(shape=shape)

    def forward(self, shape=None):

        if shape is not None:
            if isinstance(shape, int) or isinstance(shape, float):
                shape = int(shape)
            else:
                raise NotImplementedError
        else:
            shape = int(self.num_dims)
        return (self.noise ** 2) * fm.Eye(shape)

    def matvec(self, vector):

        if len(vector.shape) == 1:
            assert len(vector) == self.num_dims
            vector = vector.reshape(-1, 1)
        elif len(vector.shape) == 2:
            assert vector.shape[1] == 1
        else:
            raise ValueError('Expected vector ... got incompatible shape, i.e., ' + str(vector.shape))

        n = self.num_dims
        return self.noise ** 2 * (fm.Eye(n) * vector).real

