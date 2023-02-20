import os
import glob
import numpy as np
import mlptrain as mlt
from .test_potential import TestPotential
from .molecules import _h2
from .utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


def _h2_configuration():
    system = mlt.System(_h2(), box=[50, 50, 50])
    config = system.random_configuration()

    return config


def _run_metadynamics(metadynamics):
    metadynamics.run_metadynamics(start_config=_h2_configuration(),
                                  mlp=TestPotential('1D'),
                                  temp=300,
                                  interval=10,
                                  dt=1,
                                  pace=100,
                                  width=0.05,
                                  height=0.1,
                                  biasfactor=3,
                                  n_runs=4,
                                  ps=1)


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metadynamics = mlt.Metadynamics(cv1)

    assert metadynamics.bias is not None

    _run_metadynamics(metadynamics)

    assert os.path.exists('combined_trajectory.xyz')

    assert glob.glob('plumed_files/colvar_cv1_pid*.dat')
    assert glob.glob('plumed_files/HILLS_pid*.dat')
    assert glob.glob('plumed_logs/plumed_pid*.log')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_compute_fes():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metadynamics = mlt.Metadynamics(cv1)

    _run_metadynamics(metadynamics)
    metadynamics.compute_fes(n_bins=100)

    assert glob.glob('plumed_files/fes_pid*.dat')
    assert glob.glob('plumed_logs/fes_pid*.log')

    assert os.path.exists('fes_raw.npy')
    assert os.path.exists('fes.npy')

    fes_raw = np.load('fes_raw.npy')
    fes = np.load('fes.npy')

    # 1 cv, 4 fes -> 5; 101 bins
    assert np.shape(fes_raw) == (5, 101)

    # 1 cv, 1 mean fes, 1 std dev -> 3; 101 bins
    assert np.shape(fes) == (3, 101)


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics_with_component():

    cv1 = mlt.PlumedCustomCV('plumed_cv_dist.dat', 'x')
    metadynamics = mlt.Metadynamics(cv1)

    _run_metadynamics(metadynamics)

    assert glob.glob('plumed_files/colvar_cv1_x_pid*.dat')
