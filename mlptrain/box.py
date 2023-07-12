import numpy as np
from typing import Sequence


class Box:

    def __init__(self, size: Sequence[float]):
        """
        Periodic cuboidal box

        -----------------------------------------------------------------------
        Arguments:
            size: Lattice vectors a, b, c defining the box size
        """
        assert len(size) == 3
        self.size = np.array([float(k) for k in size])

    @property
    def random_point(self) -> np.ndarray:
        """Get a random point inside the box"""
        return np.array([np.random.uniform(0.0, k) for k in self.size])

    @property
    def volume(self) -> float:
        """Volume of this box"""
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def has_zero_volume(self) -> bool:
        """Is this box essentially of zero size"""
        return self.volume < 1E-10

    @property
    def midpoint(self) -> np.ndarray:
        """Midpoint inside this box"""
        return self.size / 2.0

    def __eq__(self, other):
        """Equality of two boxes"""

        return (isinstance(other, Box)
                and np.linalg.norm(other.size - self.size) < 1E-10)
