import mlptrain
import autode
from typing import Tuple
from mlptrain.log import logger
from mlptrain.utils import work_in_tmp_dir
from mlptrain.config import Config


@work_in_tmp_dir()
def run_autode(
    configuration: 'mlptrain.Configuration', method_name: str, n_cores: int = 1
) -> None:
    """
    Run an autodE calculation

    ---------------------------------------------------------------------------
    Arguments:
        configuration:

        method_name: Name of the method. Case insensitive

        n_cores: Number of cores to use for the calculation
    """
    from autode.species import Species
    from autode.calculations import Calculation
    from autode.exceptions import CouldNotGetProperty

    method, kwds = _method_and_keywords(method_name=method_name.lower())
    logger.info(f'Running a {method_name} calculation at: {kwds}')

    calc = Calculation(
        name='tmp',
        molecule=Species(
            name='tmp',
            atoms=configuration.atoms,
            charge=configuration.charge,
            mult=configuration.mult,
        ),
        method=method,
        keywords=kwds,
        n_cores=n_cores,
    )
    calc.run()

    try:
        configuration.forces.true = -calc.molecule.gradient.to('eV Å^-1')

    except CouldNotGetProperty:
        logger.error('Failed to set forces')

    energy = calc.molecule.energy
    if energy is None:
        logger.error('Failed to calculate the energy')
        if calc.output.exists:
            print(''.join(calc.output.file_lines[-50:]))

        return None

    configuration.energy.true = energy.to('eV')
    configuration.partial_charges = calc.molecule.partial_charges
    return None


def _method_and_keywords(
    method_name: str,
) -> Tuple['autode.wrappers.Method', 'autode.wrappers.keywords.Keywords']:
    """Get the method and associated keywords to use in a QM calculation"""
    from autode.methods import ORCA, XTB, G16, G09

    if method_name == 'orca':
        method, kwds = ORCA(), _orca_keywords()

    elif method_name == 'g09' or method_name == 'g16':
        method = G09() if method_name == 'g09' else G16()
        kwds = _gaussian_keywords()

    elif method_name == 'xtb':
        method = XTB()
        kwds = method.keywords.grad

    else:
        raise ValueError(f'Unknown method {method_name}')

    return method, kwds


def _orca_keywords() -> 'autode.wrappers.keywords.Keywords':
    """Keywords e.g. functional and basis set to use for an ORCA calculation"""

    if len(Config.orca_keywords) == 0:
        raise ValueError(
            'For ORCA training GTConfig.orca_keywords must be'
            ' set. e.g.\nmlt.Config.orca_keywords '
            "= ['PBE', 'def2-SVP', 'EnGrad'])"
        )

    return Config.orca_keywords


def _gaussian_keywords() -> 'autode.wrappers.keywords.Keywords':
    """Keywords e.g. functional and basis set to use for an Gaussian
    calculation, either Gaussian09 or Gaussian16"""

    if len(Config.gaussian_keywords) == 0:
        raise ValueError(
            'To train with Gaussian QM calculations '
            'mlt.Config.gaussian_keywords must be set.'
        )

    return Config.gaussian_keywords
