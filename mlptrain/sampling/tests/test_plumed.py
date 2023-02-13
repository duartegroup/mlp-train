import os
import pytest
import mlptrain as mlt
from .utils import work_in_zipped_dir
here = os.path.abspath(os.path.dirname(__file__))


@work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_plumed_cv_initialisation():

    cv0 = mlt.PlumedAverageCV('cv0', ((0, 1), (2, 3)))

    assert cv0.name == 'cv0'
    assert cv0.dof_names == ['cv0_dist0', 'cv0_dist1']
    assert cv0.setup == ['cv0_dist0: DISTANCE ATOMS=1,2',
                         'cv0_dist1: DISTANCE ATOMS=3,4',
                         'cv0: CUSTOM '
                         'ARG=cv0_dist0,cv0_dist1 '
                         'VAR=cv0_dist0,cv0_dist1 '
                         'FUNC=0.5*(cv0_dist0+cv0_dist1) '
                         'PERIODIC=NO']

    with pytest.raises(ValueError):
        mlt.PlumedAverageCV('cv2')
        mlt.PlumedDifferenceCV('cv3', ((0, 1), (2, 3), (4, 5)))

    with pytest.raises(NotImplementedError):
        mlt.PlumedAverageCV('cv1', [(0, 1, 2, 3, 4, 5)])

    cv4 = mlt.PlumedCustomCV('from_file.dat')

    assert cv4.name == 'cv4'
    assert cv4.dof_names == ['cv4_dof0', 'cv4_dof1']
    assert cv4.setup == ['dof0: DISTANCE ATOMS=1,2',
                         'dof1: DISTANCE ATOMS=3,4',
                         'cv4: CUSTOM '
                         'ARG=dof0,dof1 '
                         'VAR=dof0,dof1 '
                         'FUNC=dof0*dof1 '
                         'PERIODIC=NO']


def test_plumed_bias_initialisation():

    cv0 = mlt.PlumedAverageCV('cv0', [(0, 1, 2, 3)])
    cv1 = mlt.PlumedAverageCV('cv1', [(4, 5, 6, 7)])

    bias = mlt.PlumedBias((cv0, cv1))

    bias.set_metad_params(pace=10, width=0.2, height=0.5, biasfactor=2)

    assert bias.cvs == (cv0, cv1)
    assert bias.pace == 10
    assert bias.width == 0.2
    assert bias.height == 0.5
    assert bias.biasfactor == 2

    with pytest.raises(ValueError):
        bias.set_metad_params(pace=10, width=0.2, height=0.5, biasfactor=0.5)
