import os
import glob
import mlptrain as mlt
from .test_potential import TestPotential
from .molecules import _h2
from .utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


def _h2_configuration():
    system = mlt.System(_h2(), box=[50, 50, 50])
    config = system.random_configuration()

    return config


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics():

    cv0 = mlt.PlumedAverageCV('cv0', [(0, 1)])
    metadynamics = mlt.Metadynamics([cv0])
    assert metadynamics.bias is not None

    metadynamics.run_metadynamics(start_config=_h2_configuration(),
                                  mlp=TestPotential('1D'),
                                  temp=300,
                                  interval=10,
                                  dt=1,
                                  pace=100,
                                  width=0.05,
                                  height=0.1,
                                  biasfactor=3,
                                  n_runs=2,
                                  ps=1)

    assert os.path.exists('combined_trajectory.xyz')

    assert glob.glob(f'plumed_files/colvar_cv0_pid*.dat')
    assert glob.glob(f'plumed_files/HILLS_pid*.dat')
    assert glob.glob(f'plumed_logs/plumed_pid*.log')
