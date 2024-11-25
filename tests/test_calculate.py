import pytest
import numpy as np
from mlptrain.configurations import Configuration
from mlptrain.config import Config  # Import Config from mlptrain.config
from mlptrain.energy import Energy
from mlptrain.forces import Forces
from mlptrain.configurations.calculate import (
    run_autode,
    _method_and_keywords,
    _orca_keywords,
    _gaussian_keywords,
)


class MockSpecies:
    """Mock class for autode.species.Species"""

    def __init__(self, name, atoms, charge, mult):
        self.name = name
        self.atoms = atoms
        self.charge = charge
        self.mult = mult


class MockCalculation:
    """Mock class for autode.calculation.Calculation"""

    def __init__(self, name, molecule, method, keywords, n_cores):
        self.name = name
        self.molecule = molecule
        self.method = method
        self.keywords = keywords
        self.n_cores = n_cores
        self._energy_success = (
            True  # Flag for controlling energy calculation success
        )
        self._gradient_success = (
            True  # Flag for controlling gradient calculation success
        )

    def run(self):
        """Simulate the run behavior."""
        pass  # Assume the calculation runs successfully

    def get_gradients(self):
        """Return mock gradients or raise an error if _gradient_success is False"""
        if self._gradient_success:
            return MockUnitConversion(np.array([-0.1, 0.2, -0.3]))
        else:
            raise Exception('Could not get gradients')

    def get_energy(self):
        """Return mock energy or None if _energy_success is False"""
        return MockUnitConversion(-1.0) if self._energy_success else None

    def get_atomic_charges(self):
        """Return mock partial charges"""
        return np.array([0.5, -0.5])

    @property
    def output(self):
        """Simulate output attribute with file lines"""

        class Output:
            file_lines = ['Some output log line'] * 50
            exists = True

        return Output()


class MockUnitConversion:
    """Mock class for autode's units conversion, simulating to()"""

    def __init__(self, value):
        self.value = value

    def to(self, unit):
        return self.value


# Fixtures


@pytest.fixture
def configuration():
    """Fixture for a mock Configuration object"""
    config = Configuration(atoms=['H', 'O'], charge=0, mult=1)
    config.forces = Forces()
    config.energy = Energy()
    config.partial_charges = None
    return config


@pytest.fixture
def mock_method_and_keywords(monkeypatch):
    """Fixture to mock _method_and_keywords function"""

    def _mock_method_and_keywords(method_name):
        return 'mock_method', 'mock_keywords'

    monkeypatch.setattr(
        'mlptrain.configurations.calculate._method_and_keywords',
        _mock_method_and_keywords,
    )


@pytest.fixture
def mock_autode(monkeypatch):
    """Fixture to mock autode Species and Calculation"""
    monkeypatch.setattr('autode.species.Species', MockSpecies)
    monkeypatch.setattr('autode.calculation.Calculation', MockCalculation)


@pytest.fixture
def set_config(monkeypatch):
    """Fixture to set required config values for ORCA and Gaussian keywords"""
    gauss_kws = Config.gaussian_keywords
    orca_kws = Config.orca_keywords
    Config.orca_keywords = ['PBE', 'def2-SVP', 'EnGrad']
    Config.gaussian_keywords = ['B3LYP', '6-31G(d)', 'Force']
    yield
    Config.gaussian_keywords = gauss_kws
    Config.orca_keywords = orca_kws


# Tests


def test_run_autode_success(
    mock_autode, mock_method_and_keywords, configuration
):
    """Test run_autode for a successful calculation"""

    run_autode(configuration, method_name='mock_method', n_cores=1)

    # Assertions to verify configuration attributes were set as expected
    assert (configuration.forces.true == np.array([0.1, -0.2, 0.3])).all()
    assert configuration.energy.true == -1.0
    assert (configuration.partial_charges == [0.5, -0.5]).all()


def test_run_autode_failed_energy(
    mock_autode, mock_method_and_keywords, configuration, capsys
):
    """Test run_autode when energy calculation fails but gradients succeed"""

    # Set up mock Calculation to simulate energy calculation failure
    calc_instance = MockCalculation(
        name='tmp', molecule=None, method=None, keywords=None, n_cores=1
    )
    calc_instance._energy_success = False  # Fail energy calculation
    calc_instance._gradient_success = True  # Succeed in gradient calculation

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            'autode.calculation.Calculation',
            lambda *args, **kwargs: calc_instance,
        )
        run_autode(configuration, method_name='mock_method', n_cores=1)

    captured = capsys.readouterr()
    assert 'Failed to calculate the energy' in captured.err
    assert configuration.energy.true is None


def test_method_and_keywords_success(set_config):
    """Test _method_and_keywords for valid methods"""
    methods = {'orca': 'orca', 'g09': 'g09', 'g16': 'g16', 'xtb': 'xtb'}
    for method_name, expected in methods.items():
        method, keywords = _method_and_keywords(method_name)
        assert (
            method.name == expected
        )  # Mocked ORCA, G09, etc., should have these names


def test_method_and_keywords_invalid():
    """Test _method_and_keywords raises ValueError for an invalid method"""
    with pytest.raises(ValueError, match='Unknown method'):
        _method_and_keywords('invalid_method')


# @pytest.mark.xfail will be removed after autode update
@pytest.mark.xfail
def test_orca_keywords_success(set_config):
    """Test _orca_keywords retrieves the ORCA keywords from Config"""
    keywords = _orca_keywords()
    assert keywords == Config.orca_keywords


def test_orca_keywords_no_config():
    """Test _orca_keywords raises ValueError when ORCA keywords are not set"""
    with pytest.raises(
        ValueError,
        match='For ORCA training GTConfig.orca_keywords must be set',
    ):
        _orca_keywords()


# @pytest.mark.xfail will be removed after autode update
@pytest.mark.xfail
def test_gaussian_keywords_success(set_config):
    """Test _gaussian_keywords retrieves the Gaussian keywords from Config"""
    keywords = _gaussian_keywords()
    assert keywords == Config.gaussian_keywords


def test_gaussian_keywords_no_config():
    """Test _gaussian_keywords raises ValueError when Gaussian keywords are not set"""
    with pytest.raises(
        ValueError,
        match='To train with Gaussian QM calculations mlt.Config.gaussian_keywords must be set',
    ):
        _gaussian_keywords()
