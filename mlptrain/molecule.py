import numpy as np
import autode as ade
from scipy.spatial.distance import cdist


class Molecule(ade.Molecule):
    @property
    def centroid(self) -> np.ndarray:
        """
        Centroid of this molecule

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): shape = (3,)
        """
        return np.average(self.coordinates, axis=0)

    def is_in_box(self, box: "mlptrain.box.Box") -> bool:
        """Is this molecule totally inside a box with an origin at
        (0,0,0) and top right corner (a, b, c) = box.size

        -----------------------------------------------------------------------
        Arguments:
            box:

        Returns:
            (bool):
        """
        coords = self.coordinates

        if np.min(coords) < 0.0:
            return False

        # Maximum x, y, z component of all atoms should be < a, b, c
        if max(np.max(coords, axis=0) - box.size) > 0:
            return False

        return True

    def min_distance_to(self, coords: np.ndarray) -> float:
        """Calculate the minimum distance from this molecule to a set
        of coordinates

        -----------------------------------------------------------------------
        Arguments:
            coords: shape = (n, 3)

        Returns:
            (float): Minimum distance (Å)
        """
        # Infinite distance to the other set if there are no coordinates
        if len(coords) == 0:
            return np.inf

        return np.min(cdist(coords, self.coordinates))

    def random_normal_jiggle(self, sigma: float = 0.01) -> None:
        """
        Add a random displacement to each atoms position.

        -----------------------------------------------------------------------
        Arguments:
            sigma: Standard deviation of the standard deviation
        """
        dx = np.random.normal(
            scale=sigma, loc=0.0, size=(self.n_atoms, 3)
        )  # Å

        self.coordinates += dx

        return None
