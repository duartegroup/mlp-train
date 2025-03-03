import mlptrain
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional
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

        -----------------------------------------------------------------------
        Returns:
            (int):
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
        """
        Selection method based on the absolute difference between the
        true and predicted total energies.

        -----------------------------------------------------------------------
        Arguments:
            e_thresh: E_T
        """
        super().__init__()

        self.e_thresh = e_thresh

    def __call__(self, configuration, mlp, **kwargs) -> None:
        """
        Evaluate the true and predicted energies, used to determine if this
        configuration should be selected.

        -----------------------------------------------------------------------
        Arguments:
            configuration: Configuration that may or may not be selected

            mlp: Machine learnt potential

            method_name: Name of the reference method to use
        """
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
        """
        10 E_T > |E_predicted - E_true| > E_T
        """
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
    def __init__(self, descriptor, threshold: float = 0.999):
        """
        Selection criteria based on the maximum distance between any of the
        training set and a new configuration. Evaluated based on the similarity
        SOAP kernel vector (K*) between a new configuration and prior training
        data

        -----------------------------------------------------------------------
        Arguments:
            descriptor: A descriptor instance (e.g., SoapDescriptor) with user-defined parameters.
            threshold: Value below which a configuration will be selected
        """
        super().__init__()

        if threshold < 0.1 or threshold >= 1.0:
            raise ValueError('Cannot have a threshold outside [0.1, 1]')

        self.descriptor = descriptor
        """
        Call the descriptor instance with user-defined parameters,
        eg. SoapDescriptor = SoapDescriptor(average="outer", r_cut=6.0, n_max=8, l_max=8)
        selector = AtomicEnvSimilarity(descriptor=SoapDescriptor, threshold=0.95)
        """
        self.threshold = float(threshold)
        self._k_vec = np.array([])

    def __call__(
        self,
        configuration: 'mlptrain.Configuration',
        mlp: 'mlptrain.potentials.MLPotential',
        **kwargs,
    ) -> None:
        """
        Evaluate the selection criteria

        -----------------------------------------------------------------------
        Arguments:
            configuration: Configuration to evaluate on

            mlp: Machine learning potential with some associated training data
        """
        if len(mlp.training_data) == 0:
            return None

        self._k_vec = self.descriptor.kernel_vector(
            configuration, configurations=mlp.training_data, zeta=8
        )

        return None

    @property
    def select(self) -> bool:
        """
        Determine if this configuration should be selected, based on the
        minimum similarity between it and all of the training data

        -----------------------------------------------------------------------
        Returns:
            (bool): If this configuration should be selected
        """
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
    descriptor,
    dim_reduction: bool = False,
    distance_metric: str = 'euclidean',
    n_neighbors: int = 15,
) -> int:
    """
    This function identifies whether a new data (configuration)
    is the outlier in comparison with the existing data (configurations) by Local Outlier
    Factor (LOF). For more details about the LOF method, please see the lit.
    Breunig, M. M., Kriegel, H.-P., Ng, R. T. & Sander, J. LOF: Identifying
    density-based local outliers. SIGMOD Rec. 29, 93–104 (2000).

    -----------------------------------------------------------------------
    Arguments:
    descriptor: Descriptor instance with `compute_representation` method.
    dim_reduction: if Ture, dimensionality reduction will
                   be performed before LOF calculation (so far only PCA available).
    distance_metric: distance metric used in LOF,
                     which could be one of 'euclidean',
                     'cosine' and 'manhattan’.
    n_neighbors: number of neighbors considered when computing the LOF.

    -----------------------------------------------------------------------
    Returns:

    -1 for anomalies/outliers and +1 for inliers.
    """
    if not hasattr(descriptor, 'compute_representation'):
        raise ValueError(
            "The provided descriptor does not have a 'compute_representation' method."
        )

    m1 = descriptor.compute_representation(configurations)
    m1 /= np.linalg.norm(m1, axis=1).reshape(len(configurations), 1)

    v1 = descriptor.compute_representation(configuration)
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
    'contamination: define the porpotional of outliner in the data, the higher, the less abnormal'
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
        """
        Selection criteria based on analysis whether the configuration is
        outlier by outlier_identifier function
        -----------------------------------------------------------------------
        Arguments:
            pca: whether to do dimensionality reduction by PCA.
                 As the selected distance_metric may potentially suffer from
                 the curse of dimensionality, the dimensionality reduction step
                 (using PCA) could be applied before calculating the LOF.
                 This would ensure good performance in high-dimensional data space.
            For the other arguments, please see details in the outlier_identifier function
        """
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
