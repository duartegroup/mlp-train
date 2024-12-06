import numpy as np
from autode.atoms import Atom
from mlptrain.configurations import ConfigurationSet, Configuration
from mlptrain.configurations.plotting import parity_plot
from mlptrain.utils import work_in_tmp_dir


@work_in_tmp_dir()
def test_parity_plot():
    config_set = ConfigurationSet(allow_duplicates=True)

    for i in range(1000):
        config = Configuration(atoms=[Atom('H'), Atom('C'), Atom('C')])

        # energies
        config.energy.true = 0.0 + (1.0 * i)
        config.energy.predicted = 1.0 + (1.0 * i)

        # forces
        config.forces.true = np.random.uniform(1, 100, 3).reshape(1, 3)
        config.forces.predicted = np.random.uniform(1, 100, 3).reshape(1, 3)

        config_set.append(config)

    # simply check there are no issues running this code
    parity_plot(config_set)
