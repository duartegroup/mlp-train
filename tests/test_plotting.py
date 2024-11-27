import numpy as np
from autode.atoms import Atom
from mlptrain.configurations import ConfigurationSet, Configuration
from mlptrain.configurations.plotting import parity_plot
from mlptrain.utils import work_in_tmp_dir


@work_in_tmp_dir()
def test_parity_plot():
    config_set = ConfigurationSet(allow_duplicates=True)

    for i in range(10):
        config = Configuration(atoms=[Atom('H'), Atom('C'), Atom('C')])
        config.energy.true = 0.0 + (1.0 * i)
        config.energy.predicted = 1.0 + (1.0 * i)
        config.forces.true = (1.0 + 1.0 * i) * np.ones(shape=(1, 3))
        config.forces.predicted = (2.0 + 1.0 * i) * np.ones(shape=(1, 3))
        config_set.append(config)

    assert len(config_set) == 10

    # simply check there are no issues running this code
    parity_plot(config_set)
