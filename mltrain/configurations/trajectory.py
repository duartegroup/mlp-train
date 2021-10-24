from mltrain.configurations.configuration_set import ConfigurationSet


class Trajectory(ConfigurationSet):
    """Trajectory"""

    @property
    def t0(self) -> float:
        """Initial time of this trajectory

        Returns:
            (float): t_0 in fs
        """
        return 0. if len(self) == 0 else self[0].time

    @t0.setter
    def t0(self, value: float):
        """Set the initial time for a trajectory"""
        for frame in self:
            frame.time += value

    @property
    def final_frame(self) -> 'mltrain.Configuration':
        """
        Return the final frame from this trajectory

        Returns:
            (mltrain.Configuration): Frame
        """

        if len(self) == 0:
            raise ValueError('Had no final frame - no configurations present')

        return self[-1]
