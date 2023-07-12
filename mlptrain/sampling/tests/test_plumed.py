import os
import pytest
import mlptrain as mlt
from .utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


def test_plumed_cv_from_atom_groups():

    cv1 = mlt.PlumedDifferenceCV('cv1', ((0, 1), (2, 3)))

    assert cv1.name == 'cv1'
    assert cv1.units == 'Å'
    assert cv1.dof_names == ['cv1_dist1', 'cv1_dist2']
    assert cv1.setup == ['cv1_dist1: DISTANCE ATOMS=1,2',
                         'cv1_dist2: DISTANCE ATOMS=3,4',
                         'cv1: CUSTOM '
                         'ARG=cv1_dist1,cv1_dist2 '
                         'VAR=cv1_dist1,cv1_dist2 '
                         'FUNC=cv1_dist1-cv1_dist2 '
                         'PERIODIC=NO']

    cv2 = mlt.PlumedAverageCV('cv2', (0, 1, 2))

    assert cv2.name == 'cv2'
    assert cv2.units == 'rad'
    assert cv2.dof_names == ['cv2_ang1']
    assert cv2.setup == ['cv2_ang1: ANGLE ATOMS=1,2,3',
                         'cv2: CUSTOM '
                         'ARG=cv2_ang1 '
                         'VAR=cv2_ang1 '
                         'FUNC=1.0*(cv2_ang1) '
                         'PERIODIC=NO']

    with pytest.raises(TypeError):
        mlt.PlumedAverageCV('')

    with pytest.raises(TypeError):
        mlt.PlumedAverageCV('', 0)

    with pytest.raises(TypeError):
        mlt.PlumedAverageCV('', ())

    with pytest.raises(ValueError):
        mlt.PlumedAverageCV('', (1,))

    with pytest.raises(NotImplementedError):
        mlt.PlumedAverageCV('', [(0, 1, 2, 3, 4, 5), (1, 2, 3)])

    with pytest.raises(ValueError):
        mlt.PlumedDifferenceCV('', ((0, 1), (2, 3), (4, 5)))


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_plumed_cv_from_file():

    cv1 = mlt.PlumedCustomCV('plumed_cv_custom.dat',
                             component='spath',
                             units='Å')

    assert cv1.name == 'p1.spath'
    assert cv1.units == 'Å'
    assert cv1.setup == ['p1: PATH '
                         'REFERENCE=path.pdb '
                         'TYPE=OPTIMAL '
                         'LAMBDA=500.0']

    with open('path.pdb', 'r') as f:
        data1 = f.read()

    assert cv1.files == [('path.pdb', data1)]

    os.remove('path.pdb')
    cv1.write_files()

    with open('path.pdb', 'r') as f:
        data2 = f.read()

    assert data1 == data2


def test_plumed_cv_walls():

    cv1 = mlt.PlumedDifferenceCV('cv1', ((0, 1), (2, 3)))

    cv1.attach_lower_wall(location=1, kappa=150.0, exp=3)
    cv1.attach_upper_wall(location=3, kappa=150.0, exp=3)

    assert cv1.setup == ['cv1_dist1: DISTANCE ATOMS=1,2',
                         'cv1_dist2: DISTANCE ATOMS=3,4',
                         'cv1: CUSTOM '
                         'ARG=cv1_dist1,cv1_dist2 '
                         'VAR=cv1_dist1,cv1_dist2 '
                         'FUNC=cv1_dist1-cv1_dist2 '
                         'PERIODIC=NO',
                         'LOWER_WALLS ARG=cv1 AT=1 KAPPA=150.0 EXP=3',
                         'UPPER_WALLS ARG=cv1 AT=3 KAPPA=150.0 EXP=3']

    with pytest.raises(TypeError):
        cv1.attach_lower_wall(location=0.5, kappa=150.0, exp=3)


def test_plumed_bias_from_cvs():

    cv1 = mlt.PlumedAverageCV('cv1', [(0, 1, 2, 3)])
    cv2 = mlt.PlumedAverageCV('cv2', [(4, 5, 6, 7)])

    bias = mlt.PlumedBias((cv1, cv2))

    with pytest.raises(ValueError):
        bias._set_metad_params(pace=10, width=(0.2, 0.3), height=0.5, biasfactor=0.5)

    with pytest.raises(ValueError):
        bias._set_metad_params(pace=10, width=0.2, height=0.5, biasfactor=2)

    bias.initialise_for_metad_al(pace=10,
                                 width=(0.2, 0.3),
                                 height=0.5,
                                 biasfactor=2,
                                 grid_min=(0.5, 1.5),
                                 grid_max=(0.6, 1.6))

    assert bias.cvs == (cv1, cv2)
    assert bias.pace == 10
    assert bias.width == (0.2, 0.3)
    assert bias.height == 0.5
    assert bias.biasfactor == 2
    assert bias.metad_grid_min == (0.5, 1.5)
    assert bias.metad_grid_max == (0.6, 1.6)
    assert bias.metad_grid_bin is None

    assert bias.metad_grid_setup == 'GRID_MIN=0.5,1.5 GRID_MAX=0.6,1.6 '

    bias.strip()

    for attribute, value in bias.__dict__.items():
        if attribute == 'cvs':
            assert value is not None

        else:
            assert value is None


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_plumed_bias_from_file():

    bias = mlt.PlumedBias(filename='plumed_bias.dat')

    assert bias.setup == ['dof1: DISTANCE ATOMS=1,2',
                          'dof2: DISTANCE ATOMS=2,3',
                          'cv1: CUSTOM ARG=dof1,dof2 VAR=dof1,dof2 '
                          'FUNC=dof2-dof1 PERIODIC=NO',
                          'lwall: LOWER_WALLS ARG=cv1 AT=1 KAPPA=150.0 EXP=3',
                          'p1: PATH REFERENCE=path.pdb TYPE=OPTIMAL '
                          'LAMBDA=500.0',
                          'UPPER_WALLS ARG=cv1 AT=3 KAPPA=150.0 EXP=3',
                          'METAD ARG=cv1,p1.spath PACE=100 HEIGHT=0.1 '
                          'SIGMA=0.5 BIASFACTOR=4 FILE=HILLS.dat',
                          'PRINT ARG=cv1,p1.spath FILE=colvar.dat STRIDE=10']

    with open('path.pdb', 'r') as f:
        data1 = f.read()

    assert bias.cv_files == [('path.pdb', data1)]

    os.remove('path.pdb')
    bias.write_cv_files()

    with open('path.pdb', 'r') as f:
        data2 = f.read()

    assert data1 == data2

    bias.strip()

    assert bias.setup == ['dof1: DISTANCE ATOMS=1,2',
                          'dof2: DISTANCE ATOMS=2,3',
                          'cv1: CUSTOM ARG=dof1,dof2 VAR=dof1,dof2 '
                          'FUNC=dof2-dof1 PERIODIC=NO',
                          'lwall: LOWER_WALLS ARG=cv1 AT=1 KAPPA=150.0 EXP=3',
                          'p1: PATH REFERENCE=path.pdb TYPE=OPTIMAL '
                          'LAMBDA=500.0',
                          'UPPER_WALLS ARG=cv1 AT=3 KAPPA=150.0 EXP=3']


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_plumed_plot():

    colvar1 = 'test_plumed_plot/colvar1.dat'
    colvar2 = 'test_plumed_plot/colvar2.dat'

    mlt.plot_cv_versus_time(filename=colvar1,
                            time_units='fs',
                            cv_units='Å',
                            cv_limits=(0.5, 1.5),
                            label='0')

    assert os.path.exists('cv1_0.pdf')

    mlt.plot_cv1_and_cv2(filenames=(colvar1, colvar2),
                         cvs_units=('Å', 'Å'),
                         cvs_limits=((0.5, 1.5), (0.5, 1.5)),
                         label='0')

    assert os.path.exists('cv1_cv2_0.pdf')
