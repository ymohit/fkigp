import math
import numpy as np


# Disclaimer: this way of maintaining constraints for hyper-parameters is inspired from gpytorch.
# Gpytorch: https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/constraints/constraints.py


DEFAULT_SOFTPLUS_VALUE = 0.5413248546129181     # This leads to 1 in the parametric space


def inv_softplus(x):
    return x + np.log(-np.exp(-x) + 1)


def inv_sigmoid(x):
    return np.log(x) - np.log(1 - x)


def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def softplus(x, limit=30):

    try:
        # Scalar case
        if x > limit:
            return x
        else:
            return np.log(1.0 + np.exp(x))

    except ValueError:
        if (x > limit).any():
            r_val = np.log(1.0 + np.exp(x))
            r_val[x > limit] = x[x > limit]
            return r_val
        else:
            return np.log(1.0 + np.exp(x))


def exp(x):
    return np.exp(x)


def log(x):
    return np.log(x)


class Interval(object):

    def __init__(self, lower_bound, upper_bound, transform='softplus', initial_value=None):

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self._value = initial_value

        if transform == 'sigmoid':
            self._transform = sigmoid
            self._inv_transform = inv_sigmoid
        elif transform == 'softplus':
            self._transform = softplus
            self._inv_transform = inv_softplus
        elif transform == 'exp':
            self._transform = exp
            self._inv_transform = log
        else:
            raise NotImplementedError

    @property
    def enforced(self):
        return self._transform is not None

    def check(self, vector):
        return bool(np.all(vector <= self.upper_bound) and np.all(vector >= self.lower_bound))

    def check_raw(self, vector):
        return bool(
            np.all((self.transform(vector) <= self.upper_bound))
            and np.all(self.transform(vector) >= self.lower_bound)
        )

    def intersect(self, other):
        if self.transform != other.transform:
            raise RuntimeError("Cant intersect Interval constraints with conflicting transforms!")
        lower_bound = np.max(self.lower_bound, other.lower_bound)
        upper_bound = np.min(self.upper_bound, other.upper_bound)
        return Interval(lower_bound, upper_bound)

    def transform(self, vector):
        if not self.enforced:
            return vector
        transformed_tensor = (self._transform(vector) * (self.upper_bound - self.lower_bound)) + self.lower_bound
        return transformed_tensor

    def inverse_transform(self, transformed_vector):
        """
        Applies the inverse transformation.
        """
        if not self.enforced:
            return transformed_vector

        tensor = self._inv_transform((transformed_vector - self.lower_bound) / (self.upper_bound - self.lower_bound))
        return tensor

    @property
    def value(self):
        """
        The initial parameter value (if specified, None otherwise)
        """
        return self.transform(self._value)

    @property
    def raw_value(self):
        """
        The initial parameter value (if specified, None otherwise)
        """
        return self._value


class GreaterThan(Interval):
    def __init__(self, lower_bound, transform='softplus', initial_value=None):
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=math.inf,
            transform=transform,
            initial_value=initial_value,
        )

    def transform(self, vector):
        transformed_vector = self._transform(vector) + self.lower_bound if self.enforced else vector
        return transformed_vector

    def inverse_transform(self, transformed_vector):
        vector = self._inv_transform(transformed_vector - self.lower_bound) if self.enforced else transformed_vector
        return vector


class Positive(GreaterThan):
    def __init__(self, transform='softplus', initial_value=None):
        super().__init__(lower_bound=0.0, transform=transform, initial_value=initial_value)

    def transform(self, vector):
        transformed_vector = self._transform(vector) if self.enforced else vector
        return transformed_vector

    def inverse_transform(self, transformed_tensor):
        tensor = self._inv_transform(transformed_tensor) if self.enforced else transformed_tensor
        return tensor


class LessThan(Interval):

    def __init__(self, upper_bound, transform='softplus'):

        super().__init__(lower_bound=-math.inf, upper_bound=upper_bound, transform=transform)

    def transform(self, vector):
        transformed_vector = -self._transform(-vector) + self.upper_bound if self.enforced else vector
        return transformed_vector

    def inverse_transform(self, transformed_vector):
        vector = -self._inv_transform(-(transformed_vector - self.upper_bound)) if self.enforced else transformed_vector
        return vector


class NoConstraint(object):

    def __init__(self, initial_value):
        self._value = initial_value

    def transform(self, vector):
        return vector

    def inverse_transform(self, vector):
        return vector

    @property
    def value(self):
        """
        The initial parameter value (if specified, None otherwise)
        """
        return self.transform(self._value)

    @property
    def raw_value(self):
        """
        The initial parameter value (if specified, None otherwise)
        """
        return self._value
