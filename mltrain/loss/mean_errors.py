from mltrain.loss._base import LossValue, LossFunction


class RMSEValue(LossValue):

    def __repr__(self):
        return f'RMSE({self._value_str})'


class RMSE(LossFunction):
    """ RMSE = √(Σ_i (y_i^predicted - y_i^true)^2)"""

    def __call__(self, configurations, mlp) -> RMSEValue:

        raise NotImplemented


class MAD(LossValue):

    def __repr__(self):
        return f'MAD({self._value_str})'
