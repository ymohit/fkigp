import numpy as np


class ZeroMean(object):
    def __init__(self):
        pass


class ConstantMean(object):

    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, *args, **kwargs):
        shape = kwargs.get('shape', None)
        if None:
            raise NotImplementedError
        return self.value * np.ones(shape=shape)