import mlptrain
from mlptrain.configurations.configuration import Configuration
from mlptrain.configurations.configuration_set import ConfigurationSet
from mlptrain.log import logger


class Trajectory(ConfigurationSet):
    """Trajectory"""

    def __init__(self, *args: Configuration | str):
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
    def t0(self, value: float) -> None:
        """Set the initial time for a trajectory"""

        for frame in self:
            if frame.time is None:
                logger.warning(
                    'Attempted to set the initial time but a '
                    f'time was not defined. Setting to {value}'
                )
                frame.time = value

            else:
                frame.time += value

    @property
    def final_frame(self) -> 'mlptrain.Configuration':
        """
        Return the final frame from this trajectory

        -----------------------------------------------------------------------
        Returns:
            (mlptrain.Configuration): Frame
        """

        if len(self) == 0:
            raise ValueError('Trajectory is empty, there is no final frame')

        return self[-1]
