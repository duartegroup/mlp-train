import mlptrain
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional
from mlptrain.descriptors import SoapDescriptor
from mlptrain.log import logger
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


class SelectionMethod(ABC):
    """Active learning selection method

    NOTE: Should execute in serial
    """

    def __init__(self):
        """A selection method should determine whether its configuration
        should be selected during active learning"""

        self._configuration: Optional['mlptrain.Configuration'] = None

    @abstractmethod
    def __call__(
        self,
        configuration: 'mlptrain.Configuration',
        mlp: 'mlptrain.potentials.MLPotential',
        **kwargs,
    ) -> None:
        """Evaluate the selector"""

    @property
    @abstractmethod
    def select(self) -> bool:
        """Should this configuration be selected?"""

    @property
    @abstractmethod
    def too_large(self) -> bool:
        """Is the error/discrepancy too large to be selected?"""

    @property
    @abstractmethod
    def n_backtrack(self) -> int:
        """
        Number of backtracking steps that this selection method should evaluate
        if the value is 'too_large'
        """

    @property
    def check(self) -> bool:
        """
        Should we keep checking configurations in the MLP-MD trajectory
        until the first configuration that will be selected by the selector is found?
        """
        return False

    def copy(self) -> 'SelectionMethod':
        return deepcopy(self)


class AbsDiffE(SelectionMethod):
    def __init__(self, e_thresh: float = 0.1):
        super().__init__()
        self.e_thresh = e_thresh

    def __call__(self, configuration, mlp, **kwargs) -> None:
        method_name = kwargs.get('method_name', None)
        self._configuration = configuration

        if method_name is None:
            raise ValueError(
                'Evaluating the absolute difference requires a '
                'method name but None was present'
            )

        if configuration.energy.predicted is None:
            self._configuration.single_point(mlp)

        self._configuration.single_point(
            method_name, n_cores=kwargs.get('n_cores', 1)
        )
        return None

    @property
    def select(self) -> bool:
        abs_dE = abs(self._configuration.energy.delta)
        logger.info(f'|E_MLP - E_true| = {abs_dE:.4} eV')
        return 10 * self.e_thresh > abs_dE > self.e_thresh

    @property
    def too_large(self) -> bool:
        return abs(self._configuration.energy.delta) > 10 * self.e_thresh

    @property
    def n_backtrack(self) -> int:
        return 10


class AtomicEnvSimilarity(SelectionMethod):
    def __init__(self, threshold: float = 0.999):
        super().__init__()

        if threshold < 0.1 or threshold >= 1.0:
            raise ValueError('Cannot have a threshold outside [0.1, 1]')

        self.threshold = float(threshold)
        self._k_vec = np.array([])

    def __call__(
        self,
        configuration: 'mlptrain.Configuration',
        mlp: 'mlptrain.potentials.MLPotential',
        **kwargs,
    ) -> None:
        if len(mlp.training_data) == 0:
            return None

        self._k_vec = SoapDescriptor.kernel_vector(
            configuration, configurations=mlp.training_data, zeta=8
        )
        return None

    @property
    def select(self) -> bool:
        if self._n_training_envs == 0:
            return True

        return self.threshold**2 < np.max(self._k_vec) < self.threshold

    @property
    def too_large(self) -> bool:
        return np.max(self._k_vec) < self.threshold**2

    @property
    def n_backtrack(self) -> int:
        return 100

    @property
    def _n_training_envs(self) -> int:
        return len(self._k_vec)


def outlier_identifier(
    configuration: 'mlptrain.Configuration',
    configurations: 'mlptrain.ConfigurationSet',
    dim_reduction: bool = False,
    distance_metric: str = 'euclidean',
    n_neighbors: int = 15,
) -> int:
    m1 = SoapDescriptor.compute_representation(configurations)
    m1 /= np.linalg.norm(m1, axis=1).reshape(len(configurations), 1)

    v1 = SoapDescriptor.compute_representation(configuration)
    v1 /= np.linalg.norm(v1, axis=1).reshape(1, -1)

    if dim_reduction:
        pca = PCA(n_components=3)
        m1 = pca.fit_transform(m1)
        v1 = pca.transform(v1)

    clf = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        metric=distance_metric,
        novelty=True,
        contamination=0.2,
    )
    clf.fit(m1)
    new = clf.predict(v1)

    return new


class AtomicEnvDistance(SelectionMethod):
    def __init__(
        self,
        pca: bool = False,
        distance_metric: str = 'euclidean',
        n_neighbors: int = 15,
    ):
        super().__init__()
        self.pca = pca
        self.metric = distance_metric
        self.n_neighbors = n_neighbors

    def __call__(self, configuration, mlp, **kwargs) -> None:
        self.mlp = mlp
        self._configuration = configuration

    @property
    def select(self) -> bool:
        metric = outlier_identifier(
            self._configuration,
            self.mlp.training_data,
            self.pca,
            self.metric,
            self.n_neighbors,
        )
        return metric == -1

    @property
    def too_large(self) -> bool:
        return False

    @property
    def n_backtrack(self) -> int:
        return 10

    @property
    def check(self) -> bool:
        return self.mlp.n_train > 30
