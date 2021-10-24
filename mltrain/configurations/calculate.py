from mltrain.log import logger
from mltrain.utils import work_in_tmp_dir
from mltrain.config import Config


@work_in_tmp_dir()
def run_autode(configuration: 'mltrain.Configuration',
               method_name:   str,
               n_cores:       int = 1
               ) -> None:
    """
    Run an autodE calculation

    --------------------------------------------------------------------------
    Arguments:
        configuration: (mltrain.Configuration)

        method_name:

        n_cores: (int) Number of cores to use for the calculation

    kwds: (autode.wrappers.keywords.Keywords)
    """
    from autode.species import Species
    from autode.calculation import Calculation
    from autode.exceptions import CouldNotGetProperty
    from autode.methods import ORCA, XTB, G16, G09

    method_name = method_name.lower()
    kwds = None

    if method_name == 'orca':
        method = ORCA()

        if Config.orca_keywords is None:
            raise ValueError("For ORCA training GTConfig.orca_keywords must be"
                             " set. e.g.\nmlt.Config.orca_keywords "
                             "= ['PBE', 'def2-SVP', 'EnGrad'])")

        kwds = Config.orca_keywords

    elif method_name == 'g09' or method_name == 'g16':

        if Config.gaussian_keywords is None:
            raise ValueError("To train with Gaussian QM calculations "
                             "mlt.Config.gaussian_keywords must be set.")

        kwds = Config.gaussian_keywords
        method = G09() if method_name.lower() == 'g09' else G16

    elif method_name == 'xtb':
        method = XTB()

    else:
        raise ValueError(f'Unknown method {method_name}')

    if kwds is None:                   # Default to a gradient calculation
        kwds = method.keywords.grad

    calc = Calculation(name='tmp',
                       molecule=Species(name=configuration.name,
                                        atoms=configuration.atoms,
                                        charge=configuration.charge,
                                        mult=configuration.mult),
                       method=method,
                       keywords=kwds,
                       n_cores=n_cores)
    calc.run()

    try:
        configuration.forces.true = -calc.get_gradients().to('eV Å-1')

    except CouldNotGetProperty:
        logger.error('Failed to set forces')

    energy = calc.get_energy()
    if energy is None:
        logger.error('Failed to calculate the energy')
        return None

    configuration.energy.predicted = energy.to('eV')
    configuration.partial_charges = calc.get_atomic_charges()

    return None
