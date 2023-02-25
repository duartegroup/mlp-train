import os
import glob
import numpy as np
import mlptrain as mlt
from .test_potential import TestPotential
from .molecules import _h2
from .utils import work_in_zipped_dir
mlt.Config.n_cores = 2
here = os.path.abspath(os.path.dirname(__file__))


def _h2_configuration():
    system = mlt.System(_h2(), box=[50, 50, 50])
    config = system.random_configuration()

    return config


def _run_metadynamics(metadynamics):
    metadynamics.run_metadynamics(configuration=_h2_configuration(),
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
    metad = mlt.Metadynamics(cv1)

    assert metad.bias is not None

    _run_metadynamics(metad)

    assert os.path.exists('combined_trajectory.xyz')

    assert glob.glob('plumed_files/colvar_cv1_*.dat')
    assert glob.glob('plumed_files/HILLS_*.dat')
    assert glob.glob('plumed_logs/plumed_*.log')

    metad.compute_fes(n_bins=100)

    assert glob.glob('plumed_files/fes_*.dat')
    assert glob.glob('plumed_logs/fes_*.log')

    assert os.path.exists('fes_raw.npy')
    assert os.path.exists('fes.npy')

    fes_raw = np.load('fes_raw.npy')
    fes = np.load('fes.npy')

    # 1 cv, 4 fes -> 5; 100 bins + 1 -> 101
    assert np.shape(fes_raw) == (5, 101)

    # 1 cv, 1 mean fes, 1 std dev -> 3; 100 bins + 1 -> 101
    assert np.shape(fes) == (3, 101)

    # metad.plot_fes(fes)

    # assert os.path.exists('metad_free_energy.pdf')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_run_metadynamics_with_component():

    cv1 = mlt.PlumedCustomCV('plumed_cv_dist.dat', 'x')
    metad = mlt.Metadynamics(cv1)

    _run_metadynamics(metad)

    assert glob.glob('plumed_files/colvar_cv1_x_*.dat')


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_estimate_width():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    width = metad.estimate_width(configurations=_h2_configuration(),
                                 mlp=TestPotential('1D'),
                                 plot=True,
                                 fs=500)

    assert len(width) == 1

    files_directory = 'plumed_files/width_estimation'
    logs_directory = 'plumed_logs/width_estimation'

    assert os.path.isdir(files_directory)
    assert os.path.exists(os.path.join(files_directory, 'colvar_cv1_1.dat'))
    assert os.path.exists(os.path.join(files_directory, 'cv1_1.pdf'))

    assert os.path.isdir(logs_directory)
    assert os.path.exists(os.path.join(logs_directory, 'plumed_1.log'))


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_try_multiple_biasfactors():

    cv1 = mlt.PlumedAverageCV('cv1', (0, 1))
    metad = mlt.Metadynamics(cv1)

    metad.try_multiple_biasfactors(configuration=_h2_configuration(),
                                   mlp=TestPotential('1D'),
                                   temp=300,
                                   interval=10,
                                   dt=1,
                                   pace=100,
                                   width=0.05,
                                   height=0.1,
                                   biasfactors=range(5, 16, 5),
                                   plotted_cvs=cv1,
                                   ps=1)

    files_directory = 'plumed_files/multiple_biasfactors'
    logs_directory = 'plumed_logs/multiple_biasfactors'

    assert os.path.isdir(files_directory)
    assert os.path.exists(os.path.join(files_directory, 'colvar_cv1_1.dat'))
    assert os.path.exists(os.path.join(files_directory, 'colvar_cv1_3.dat'))
    assert os.path.exists(os.path.join(files_directory, 'cv1_1.pdf'))
    assert os.path.exists(os.path.join(files_directory, 'cv1_3.pdf'))

    assert os.path.isdir(logs_directory)
    assert os.path.exists(os.path.join(logs_directory, 'plumed_1.log'))
    assert os.path.exists(os.path.join(logs_directory, 'plumed_3.log'))
