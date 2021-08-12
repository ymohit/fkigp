from fkigp.configs import GsGPType
from fkigp.cgvariants import bcg
from fkigp.cgvariants import bfcg

from fkigp.gps.gpoperators import GsGpLinearOperator
from fkigp.gps.gpoperators import KissGpLinearOperator


class GpPredictionStrategy(object):

    def __init__(self):
        self.operator = None

    def exact_prediction(self, W, **kwargs):
        mean_cache = self.get_mean_cache(**kwargs)
        if W is None:
            return mean_cache, None, None

        mean_test = W * mean_cache
        return mean_test, None, None

    def get_mean_cache(self, **kwargs):
        raise NotImplementedError


class KissGpPredictionStrategy(GpPredictionStrategy):

    def __init__(self, W, train_y, Kmm, sigma, mean_value=0):

        super().__init__()
        self.operator = KissGpLinearOperator(W, Kmm, sigma, dtype=W.dtype)
        self.train_y = train_y
        self.mean_value = mean_value

    def get_mean_cache(self, **kwargs):

        """
        :param kwargs:
        :return:
        """

        # Processing for dump
        dump = kwargs.get('dump', None)

        if dump is None:
            mean_cache = self.operator.kmm_matvec(self.operator.WT * bcg(self.operator,
                                                                         self.train_y - self.mean_value,
                                                                         **kwargs))
        else:
            mean_cache = self.operator.kmm_matvec(self.operator.WT * bcg(self.operator,
                                                                         self.train_y - self.mean_value,
                                                                         X0=(1/(self.operator.sigma *2))*self.train_y,
                                                                         **kwargs))
        return mean_cache


class GsGpPredictionStrategy(GpPredictionStrategy):

    def __init__(self, WT_times_W, WT_times_Y, YT_times_Y, Kmm, sigma, gsgp_type, mean_value):

        super().__init__()

        self.operator = GsGpLinearOperator(WT_times_W, Kmm, sigma, dtype=WT_times_Y.dtype)
        self.WT_times_Y = WT_times_Y
        self.YT_times_Y = YT_times_Y
        self.gsgp_type = gsgp_type
        self.mean_value = mean_value

    def get_mean_cache(self, **kwargs):

        # Running for the chosen variant
        if self.gsgp_type == GsGPType.SYM:
            raise NotImplementedError

        elif self.gsgp_type == GsGPType.ASYM:
            raise NotImplementedError

        elif self.gsgp_type == GsGPType.FACT:

            sigma2 = self.operator.sigma ** 2
            b_hat = self.operator.kmm_matvec(self.WT_times_Y).real
            r0_hat = - (1 / sigma2) * b_hat
            x_diff = bfcg(self.operator, r0_hat.squeeze(), yty=self.YT_times_Y, **kwargs)
            mean_cache = (1 / sigma2) * (self.operator.kmm_matvec(self.WT_times_Y)).real \
                + self.operator.kmm_matvec(x_diff)

        else:
            raise NotImplementedError

        return mean_cache
