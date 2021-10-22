import numpy as np
from mltrain.md import run_mlp_md
from mltrain.log import logger
from mltrain.loss._base import LossFunction, LossValue


class Tau(LossValue):

    def __repr__(self):
        return f'τ_acc = {self._value_str}'


class TauCalculator(LossFunction):

    def __call__(self,
                 configurations: 'mltrain.ConfigurationSet',
                 mlp:            'mltrain.potentials.MLPotential') -> Tau:

        """
        Calculate τ_acc from a set of initial configurations

        Arguments:
            configurations: A set of initial configurations from which dynamics
                           will be propagated from

            mlp: Machine learnt potential

        Returns:
            (Tau): τ_acc
        """
        if len(configurations) < 2:
            raise ValueError(f'Cannot calculate τ_acc over only '
                             f'{len(configurations)} configurations. Must be > 1')

        taus = [self._calculate_single(config=c,
                                       mlp=mlp,
                                       method_name=mlp.reference_method_str)
                for c in configurations]

        # Calculate τ_acc as the average ± the standard error in the mean
        return Tau(np.average(taus),
                   error=np.std(taus) / np.sqrt(len(taus) - 1))

    def _calculate_single(self, config, mlp, method_name):
        """Calculate a single τ_acc from one configuration"""

        cuml_error, curr_time = 0, 0

        block_time = self.time_interval * gt.GTConfig.n_cores
        step_interval = self.time_interval // self.dt

        while curr_time < self.max_time:

            traj = run_mlp_md(config,
                              gap=gap,
                              temp=self.temp,
                              dt=self.dt,
                              interval=step_interval,
                              fs=block_time,
                              n_cores=min(gt.GTConfig.n_cores, 4))
            true = traj.copy()

            # Only evaluate the energy
            try:
                true.single_point(method_name=method_name)
            except (ValueError, TypeError):
                logger.warning('Failed to calculate single point energies with'
                               f' {method_name}. τ_acc will be underestimated '
                               f'by <{block_time}')
                return curr_time

            logger.info('      ___ |E_true - E_GAP|/eV ___')
            logger.info(f' t/fs      err      cumul(err)')

            for j in range(len(traj)):

                if true[j].energy is None:
                    logger.warning(f'Frame {j} had no energy')
                    e_error = np.inf
                else:
                    e_error = np.abs(true[j].energy - traj[j].energy)

                # Add any error above the allowed threshold
                cuml_error += max(e_error - self.e_l, 0)
                curr_time += self.dt * step_interval
                logger.info(f'{curr_time:5.0f}     '
                            f'{e_error:6.4f}     '
                            f'{cuml_error:6.4f}')

                if cuml_error > self.e_t:
                    return curr_time

            config = traj[-1]

        logger.info(f'Reached max(τ_acc) = {self.max_time} fs')
        return self.max_time

    def calculate(self, gap, method_name):
        """Calculate the time to accumulate self.e_t eV of error above
        self.e_l eV"""

        return None

    def __init__(self,
                 e_lower=0.1,
                 e_thresh=None,
                 max_time=1000,
                 time_interval=20,
                 temp=300,
                 dt=0.5):
        """
        τ_acc prospective error metric in fs

        ----------------------------------------------------------------------
        Arguments:

            e_lower: (float) E_l energy threshold in eV below which
                     the error is zero-ed, i.e. the acceptable level of
                     error possible in the system

            e_thresh: (float | None) E_t total cumulative error in eV. τ_acc
                      is defined at the time in the simulation where this
                      threshold is exceeded. If None then:
                      e_thresh = 10 * e_lower

            max_time: (float) Maximum time in femto-seconds for τ_acc

            time_interval: (float) Interval between which |E_true - E_GAP| is
                         calculated. Must be at least one timestep

            temp: (float) Temperature of the simulation to perform

            dt: (float) Timestep of the simulation in femto-seconds
        """

        if time_interval < dt:
            raise ValueError('The calculated interval must be more than a '
                             'single timestep')

        self.dt = float(dt)
        self.temp = float(temp)
        self.max_time = float(max_time)
        self.time_interval = float(time_interval)

        self.e_l = float(e_lower)
        self.e_t = 10 * self.e_l if e_thresh is None else float(e_thresh)

        logger.info('Successfully initialised τ_acc, will do a maximum of '
                    f'{int(self.max_time // self.time_interval)} reference '
                    f'calculations')
