import os
import numpy as np
import mlptrain as mlt
from ase.io.trajectory import Trajectory as ASETrajectory
from .test_potential import TestPotential
from .molecules import _h2, _h2o
from .utils import work_in_zipped_dir

mlt.Config.n_cores = 2
here = os.path.abspath(os.path.dirname(__file__))


def _h2_configuration():
    system = mlt.System(_h2(), box=[50, 50, 50])
    config = system.random_configuration()

    return config


def _h2o_configuration():
    system = mlt.System(_h2o(), box=[50, 50, 50])
    config = system.random_configuration()

    return config


def _run_metadynamics(
    metadynamics,
    n_runs,
    configuration=None,
    al_iter=None,
    save_sep=False,
    all_to_xyz=False,
    restart=False,
    **kwargs,
):

    if configuration is None:
        configuration = _h2_configuration()

    metadynamics.run_metadynamics(
        configuration=configuration,
        mlp=TestPotential("1D"),
        temp=300,
        dt=1,
        interval=10,
        pace=100,
        width=0.05,
        height=0.1,
        biasfactor=3,
        al_iter=al_iter,
        n_runs=n_runs,
        save_sep=save_sep,
        all_to_xyz=all_to_xyz,
        restart=restart,
        **kwargs,
    )


@work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_run_metadynamics():

    cv1 = mlt.PlumedAverageCV("cv1", (0, 1))
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    assert metad.bias is not None

    _run_metadynamics(metad, n_runs, all_to_xyz=True, save_fs=200, fs=500)

    assert os.path.exists("trajectories")
    assert os.path.exists("trajectories/combined_trajectory.xyz")

    metad_dir = "plumed_files/metadynamics"
    for idx in range(1, n_runs + 1):
        assert os.path.exists(f"trajectories/trajectory_{idx}.traj")

        for sim_time in [200, 400]:
            assert os.path.exists(
                f"trajectories/trajectory_{idx}_{sim_time}fs.traj"
            )
            assert os.path.exists(
                f"trajectories/metad_{idx}_{sim_time}fs.xyz"
            )

        assert os.path.exists(os.path.join(metad_dir, f"colvar_cv1_{idx}.dat"))
        assert os.path.exists(os.path.join(metad_dir, f"HILLS_{idx}.dat"))

        assert os.path.exists(f"gaussian_heights/gaussian_heights_{idx}.pdf")

    metad.compute_fes(n_bins=100)

    for idx in range(1, n_runs + 1):
        assert os.path.exists(f"plumed_files/metadynamics/fes_{idx}.dat")

    assert os.path.exists("fes_raw.npy")
    fes_raw = np.load("fes_raw.npy")

    # 1 cv, 4 fes -> 5; 100 bins
    assert np.shape(fes_raw) == (5, 100)

    metad.plot_fes("fes_raw.npy")
    assert os.path.exists("metad_free_energy.pdf")

    metad.plot_fes_convergence(stride=2, n_surfaces=2)

    # 500 / 100: simulation time divided by the pace <=> number of gaussians
    # Surfaces are computed every 2 gaussians
    n_computed_surfaces = (500 / 100) // 2
    for idx in range(int(n_computed_surfaces)):
        assert os.path.exists(f"plumed_files/fes_convergence/fes_1_{idx}.dat")

    assert os.path.exists("fes_convergence/fes_convergence_diff.pdf")
    assert os.path.exists("fes_convergence/fes_convergence.pdf")


@work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_run_metadynamics_restart():

    cv1 = mlt.PlumedAverageCV("cv1", (0, 1))
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    _run_metadynamics(metad, n_runs, fs=500)

    _run_metadynamics(metad, n_runs, restart=True, fs=500)

    n_steps = len(
        np.loadtxt("plumed_files/metadynamics/colvar_cv1_1.dat", usecols=0)
    )
    n_gaussians = len(
        np.loadtxt("plumed_files/metadynamics/HILLS_1.dat", usecols=0)
    )

    # Adding two 500 fs simulations with interval 10 -> 51 frames each, but
    # removing one duplicate frame
    assert n_steps == 51 + 51 - 1
    assert n_gaussians == 5 + 5

    assert os.path.exists("trajectories/trajectory_1.traj")

    trajectory = ASETrajectory("trajectories/trajectory_1.traj")

    # Adding two 1 ps simulations with interval 10 -> 101 frames each, but
    # removing one duplicate frame (same as before, except testing this for
    # the generated .traj file instead of .dat file)
    assert len(trajectory) == 51 + 51 - 1


@work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_run_metadynamics_with_inherited_bias():

    cv1 = mlt.PlumedAverageCV("cv1", (0, 1))
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    _run_metadynamics(metad, n_runs, al_iter=3, fs=500)

    _run_metadynamics(metad, n_runs, al_iter=3, restart=True, fs=500)

    metad_dir = "plumed_files/metadynamics"
    for idx in range(1, n_runs + 1):
        assert os.path.exists(f"trajectories/trajectory_{idx}.traj")

        assert os.path.exists(os.path.join(metad_dir, f"colvar_cv1_{idx}.dat"))
        assert os.path.exists(os.path.join(metad_dir, f"HILLS_{idx}.dat"))

    metad.compute_fes(via_reweighting=True)
    assert os.path.exists("fes_raw.npy")


@work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_run_metadynamics_with_component():

    cv1 = mlt.PlumedCustomCV("plumed_cv_dist.dat", "x")
    metad = mlt.Metadynamics(cv1)
    n_runs = 4

    _run_metadynamics(metad, n_runs, fs=100)

    metad_dir = "plumed_files/metadynamics"
    for idx in range(1, n_runs + 1):
        assert os.path.exists(
            os.path.join(metad_dir, f"colvar_cv1_x_{idx}.dat")
        )


@work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_run_metadynamics_with_additional_cvs():

    cv1 = mlt.PlumedAverageCV("cv1", (0, 1))
    cv2 = mlt.PlumedAverageCV("cv2", (2, 1))
    cv2.attach_upper_wall(location=3.0, kappa=150.0)

    bias = mlt.PlumedBias(cvs=(cv1, cv2))

    metad = mlt.Metadynamics(cvs=cv1, bias=bias)

    assert metad.bias == bias
    assert metad.n_cvs == 1

    n_runs = 1
    _run_metadynamics(
        metad,
        configuration=_h2o_configuration(),
        n_runs=n_runs,
        write_plumed_setup=True,
        fs=100,
    )

    with open("plumed_files/metadynamics/plumed_setup.dat", "r") as f:
        plumed_setup = [line.strip() for line in f]

    # Not including the units
    assert plumed_setup[1:] == [
        "cv1_dist1: DISTANCE ATOMS=1,2",
        "cv1: CUSTOM ARG=cv1_dist1 VAR=cv1_dist1 "
        f"FUNC={1/1}*(cv1_dist1) PERIODIC=NO",
        "cv2_dist1: DISTANCE ATOMS=3,2",
        "cv2: CUSTOM ARG=cv2_dist1 VAR=cv2_dist1 "
        f"FUNC={1/1}*(cv2_dist1) PERIODIC=NO",
        "UPPER_WALLS ARG=cv2 AT=3.0 KAPPA=150.0 EXP=2",
        "metad: METAD ARG=cv1 PACE=100 HEIGHT=0.1 "
        "SIGMA=0.05 TEMP=300 BIASFACTOR=3 "
        "FILE=HILLS_1.dat",
        "PRINT ARG=cv1,cv1_dist1 FILE=colvar_cv1_1.dat STRIDE=10",
        "PRINT ARG=cv2,cv2_dist1 FILE=colvar_cv2_1.dat STRIDE=10",
    ]


@work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_estimate_width():

    cv1 = mlt.PlumedAverageCV("cv1", (0, 1))
    metad = mlt.Metadynamics(cv1)

    width = metad.estimate_width(
        configurations=_h2_configuration(),
        mlp=TestPotential("1D"),
        plot=True,
        fs=100,
    )

    assert len(width) == 1

    files_directory = "plumed_files/width_estimation"
    plots_directory = "width_estimation"

    assert os.path.isdir(files_directory)
    assert os.path.exists(os.path.join(files_directory, "colvar_cv1_1.dat"))

    assert os.path.isdir(plots_directory)
    assert os.path.exists(os.path.join(plots_directory, "cv1_config1.pdf"))


@work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_try_multiple_biasfactors():

    cv1 = mlt.PlumedAverageCV("cv1", (0, 1))
    metad = mlt.Metadynamics(cv1)
    biasfactors = range(5, 11, 5)

    metad.try_multiple_biasfactors(
        configuration=_h2_configuration(),
        mlp=TestPotential("1D"),
        temp=300,
        interval=10,
        dt=1,
        pace=100,
        width=0.05,
        height=0.1,
        biasfactors=biasfactors,
        plotted_cvs=cv1,
        fs=100,
    )

    files_dir = "plumed_files/multiple_biasfactors"
    assert os.path.isdir(files_dir)

    plots_dir = "multiple_biasfactors"
    assert os.path.isdir(plots_dir)

    for idx, biasf in enumerate(biasfactors, start=1):
        assert os.path.exists(os.path.join(files_dir, f"colvar_cv1_{idx}.dat"))
        assert os.path.exists(os.path.join(plots_dir, f"cv1_biasf{biasf}.pdf"))


@work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_block_analysis():

    cv1 = mlt.PlumedAverageCV("cv1", (0, 1))
    metad = mlt.Metadynamics(cv1)
    dt = 1
    interval = 10
    n_runs = 1
    ps = 2
    start_time = 0.5

    metad.run_metadynamics(
        configuration=_h2_configuration(),
        mlp=TestPotential("1D"),
        temp=300,
        dt=dt,
        interval=interval,
        pace=100,
        width=0.05,
        height=0.1,
        biasfactor=3,
        n_runs=n_runs,
        ps=ps,
    )

    metad.block_analysis(start_time=start_time)

    assert os.path.exists("block_analysis.pdf")
    assert os.path.exists("block_analysis.npz")

    start_time_fs = start_time * 1e3
    n_steps = int(start_time_fs / dt)
    n_used_frames = n_steps // interval

    min_n_blocks = 10
    min_blocksize = 10
    blocksize_interval = 10
    max_blocksize = n_used_frames // min_n_blocks

    data = np.load("block_analysis.npz")

    # axis 0: CV1; axis 1: 300 bins
    assert np.shape(data["CVs"]) == (1, 300)
    for blocksize in range(
        min_blocksize, max_blocksize + 1, blocksize_interval
    ):

        # axis 0: error; axis 1: 300 bins
        assert np.shape(data[str(blocksize)]) == (3, 300)
