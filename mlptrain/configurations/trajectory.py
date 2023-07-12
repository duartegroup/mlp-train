from typing import Union

from mlptrain.log import logger
from mlptrain.configurations.configuration import Configuration
from mlptrain.configurations.configuration_set import ConfigurationSet


class Trajectory(ConfigurationSet):
    """Trajectory"""

    def __init__(self, *args: Union[Configuration, str]):
        super().__init__(*args, allow_duplicates=True)

    @property
    def t0(self) -> float:
        """Initial time of this trajectory

        -----------------------------------------------------------------------
        Returns:
            (float): t_0 in fs
        """
        return 0.0 if len(self) == 0 else self[0].time

    @t0.setter
    def t0(self, value: float):
        """Set the initial time for a trajectory"""

        for frame in self:
            if frame.time is None:
                logger.warning(
                    "Attempted to set the initial time but a "
                    f"time was note defined. Setting to {value}"
                )
                frame.time = value

            else:
                frame.time += value

        return

    @property
    def final_frame(self) -> "mlptrain.Configuration":
        """
        Return the final frame from this trajectory

        -----------------------------------------------------------------------
        Returns:
            (mlptrain.Configuration): Frame
        """

        if len(self) == 0:
            raise ValueError("Had no final frame - no configurations present")

        return self[-1]
