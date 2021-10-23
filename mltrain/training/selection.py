from abc import ABC, abstractmethod


class SelectionMethod(ABC):
    """Active learning selection method. Should execute in serial"""

    @abstractmethod
    def __call__(self,
                 configuration: 'mltrain.Configuration',
                 mlp:           'mltrain.potentials.MLPotential',
                 **kwargs
                 ) -> bool:
        """Should this configuration be selected?"""


class AbsDiffE(SelectionMethod):

    def __init__(self,
                 e_thresh: float = 0.1):
        """
        Selection method based on the absolute difference between the
        true and predicted total energies.

        Arguments:
            e_thresh: E_T
        """

        self.e_thresh = e_thresh

    def __call__(self, configuration, mlp, method_name=None, **kwargs) -> bool:
        """

                |E_predicted - E_true| > E_T

        -----------------------------------------------------------------------
        Arguments:
            configuration: Configuration that may or may not be selected

            mlp: Machine learnt potential

            method_name: Name of the reference method to use
        """
        method_name = kwargs.get('method_name', None)

        if method_name is None:
            raise ValueError('Evaluating the absolute difference requires a '
                             'method name but None was present')

        if configuration.energy.predicted is None:
            configuration.single_point(mlp)

        configuration.single_point(method_name)
        return abs(configuration.energy.delta) > self.e_thresh
