"""Tests for mlptrain.potentials.mace module"""

import importlib.util
import logging

import pytest

import mlptrain
from mlptrain.box import Box
from mlptrain.log import logger as mlp_logger
from mlptrain.utils import work_in_tmp_dir

# Only run these tests if MACE is installed
pytestmark = pytest.mark.skipif(
    not importlib.util.find_spec('mace'),
    reason='requires MACE',
)


@pytest.fixture
def mock_mace_run_train(monkeypatch, tmp_path):
    """Fixture to mock mace.cli.run_train.run function"""

    def train_mace_mock(args) -> None:
        # We obviously don't run any training,
        # but we do setup MACE logging in a similar way as the real train function
        # by calling mace.tools.setup_logger
        from mace.tools import setup_logger

        logging.info('mock_mace_run_train: Before setup_logger')
        setup_logger(directory=tmp_path)
        logging.info('mock_mace_run_train: After setup_logger')

    # from mace.cli.run_train import run as train_mace
    monkeypatch.setattr(
        'mace.cli.run_train.run',
        train_mace_mock,
    )


# This is a regression test for the log doubling issue.
# It uses the pytest caplog fixture, see:
# https://docs.pytest.org/en/stable/how-to/logging.html#caplog-fixture
# To see the log messages during testing run:
# $ pytest --log-cli-level=INFO tests/test_mace.py::test_train_logging
@work_in_tmp_dir()
def test_train_logging(
    caplog, mock_mace_run_train, h2, h2_configuration, h2o_configuration
):
    from mlptrain.potentials import MACE

    system = mlptrain.System(h2, box=Box([10, 10, 10]))
    h2_configuration.energy.true = -0.5
    h2o_configuration.energy.true = -1.0

    confs = mlptrain.ConfigurationSet(h2_configuration, h2o_configuration)

    caplog.clear()
    mlp = MACE(name='test', system=system)
    # Check that we print MACE version in the constructor
    assert caplog.records[0].message.startswith('MACE version:')

    mlp.atomic_energies = {'H': -0.5}

    caplog.clear()

    mlp.train(confs)

    num_messages = len(caplog.records)
    assert num_messages != 0

    # Make sure we print the nodename at the start of training
    assert caplog.records[0].message.startswith('Training on nodename')

    # Make sure that the number of log messages is the same on second call
    caplog.clear()
    mlp.train(confs)
    assert len(caplog.records) == num_messages

    # Make sure logging is not doubled
    caplog.clear()
    mlp_logger.info('test info from mlp logger')
    logging.info('test info from root logger')

    assert len(caplog.records) == 2
