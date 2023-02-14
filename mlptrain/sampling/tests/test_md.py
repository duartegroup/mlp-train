import os
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
def test_md_using_full_plumed_input():

    bias = mlt.PlumedBias(file_name='plumed_bias.dat')

    mlt.md.run_mlp_md(configuration=_h2_configuration(),
                      mlp=TestPotential('1D'),
                      temp=300,
                      dt=1,
                      interval=10,
                      bias=bias,
                      ps=1)

    assert os.path.exists('colvar.dat')
    assert os.path.exists('HILLS.dat')
