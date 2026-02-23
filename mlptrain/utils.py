import os
import re
import shutil
import numpy as np
import mlptrain as mlt
from mlptrain.log import logger
import autode as ade
from tempfile import mkdtemp
from functools import wraps
from typing import Optional, List, Sequence, Union, overload
from ase import units as ase_units
from autode.values import PotentialEnergy, Gradient


def work_in_tmp_dir(
    kept_substrings: Optional[Sequence[str]] = None,
    copied_substrings: Optional[Sequence[str]] = None,
):
    """
    Execute a function in a temporary directory

    ---------------------------------------------------------------------------
    Arguments:

        kept_substrings: List of substrings with which files are copied back
                         from the temporary directory
                         e.g. '.json', 'trajectory_1.traj'

        copied_substrings: List of substrings with which files are copied to
                           the temporary directory
    """

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here_path = os.getcwd()
            tmpdir_path = mkdtemp()

            if copied_substrings is not None:
                for filename in os.listdir(here_path):
                    if _name_contains_substring(
                        name=filename,
                        substrings=copied_substrings,
                        regex=False,
                    ):
                        shutil.copy(
                            src=os.path.join(here_path, filename),
                            dst=os.path.join(tmpdir_path, filename),
                        )

            # Move directories and execute
            os.chdir(tmpdir_path)

            try:
                out = func(*args, **kwargs)

            finally:
                if kept_substrings is not None:
                    for filename in os.listdir(tmpdir_path):
                        if _name_contains_substring(
                            name=filename,
                            substrings=kept_substrings,
                            regex=False,
                        ):
                            shutil.copy(
                                src=os.path.join(tmpdir_path, filename),
                                dst=os.path.join(here_path, filename),
                            )

                os.chdir(here_path)

                # Remove the temporary dir with all files and return the output
                shutil.rmtree(tmpdir_path)

            return out

        return wrapped_function

    return func_decorator


def _name_contains_substring(
    name: str, substrings: Sequence[str], regex: bool
) -> bool:
    """Returns True if one of the regex or regular substrings are found
    in the name"""

    if regex:
        for substr in substrings:
            if re.search(substr, name) is not None:
                return True

        return False

    else:
        return any(substr in name for substr in substrings)


def work_in_dir(dirname: str):
    """
    Execute a function in a different directory.

    ---------------------------------------------------------------------------
    Arguments:

        dirname: (str) Name of the directory
    """

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here_path = os.getcwd()
            dir_path = os.path.join(here_path, dirname)

            os.chdir(dir_path)
            out = func(*args, **kwargs)
            os.chdir(here_path)

            return out

        return wrapped_function

    return func_decorator


def unique_name(name: str, path: Optional[str] = None) -> str:
    """
    Returns a unique name for a file or directory in the specified directory
    by adding bck0, bck1, ... to the front of the name until a unique name
    is found.

    ---------------------------------------------------------------------------
    Arguments:

        name: (str) Name of the file or directory

        path: (str) Path of the directory where the uniqueness of the name
              is checked

    Returns:

        (str): Unique name
    """

    def _name_exists():
        if path is not None:
            return os.path.exists(os.path.join(path, name))

        else:
            return os.path.exists(name)

    i = 0
    old_name = name
    while _name_exists():
        name = f'bck{i}_{old_name}'
        i += 1

    return name


def move_files(
    moved_substrings: List[str],
    dst_folder: str,
    src_folder: Optional[str] = None,
    unique: bool = True,
    regex: bool = False,
) -> None:
    """
    Move files with given regex or regular substrings from a directory
    src_folder to a directory dst_folder. If dst_folder already exists
    the function renames the existing folder (in the case of unique == True).

    ---------------------------------------------------------------------------
    Arguments:

        moved_substrings: List of regex substrings specifying which files
                          are moved

        dst_folder: Name of the new directory where files are moved

        src_folder: Name of the directory where files are located

        unique: (bool) If False the existing directory is not renamed and the
                files are moved to that directory

        regex: (bool) If True the supplied substrings will be interpreted as
                      regex patterns
    """

    if src_folder is None:
        src_folder = os.getcwd()

    if os.path.exists(dst_folder) and unique:
        name = dst_folder.split('/')[-1]
        path = '/'.join(dst_folder.split('/')[:-1])
        unique_dst_folder = os.path.join(path, unique_name(name, path))

        os.rename(dst_folder, unique_dst_folder)
        os.makedirs(dst_folder)

    elif not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for filename in os.listdir(src_folder):
        if _name_contains_substring(
            name=filename, substrings=moved_substrings, regex=regex
        ):
            source = os.path.join(src_folder, filename)
            destination = os.path.join(dst_folder, filename)
            shutil.move(src=source, dst=destination)

    return None


def convert_exponents(string: str) -> str:
    """
    Finds exponents in a string and modifies the string such that the exponents
    are shown as exponents in plots.

    ---------------------------------------------------------------------------
    Arguments:

        string: String to be modified

    Returns

        (str): The modified string
    """

    def _modified_exponent(exponent):
        return f'$^{{{exponent.group(2)}}}$'

    exponent_pattern = re.compile(r'(\^?)(-?\d+)')

    return re.sub(exponent_pattern, _modified_exponent, string)


def convert_ase_time(
    time_array: Union[np.ndarray, float], units: str
) -> Union[np.ndarray, float]:
    """
    Converts ASE time units to different time units.

    ---------------------------------------------------------------------------
    Arguments:

        time_array: Array or a single number containing time in ase units

        units: (str) Name of the units to convert to

    Returns:

        (np.ndarray): Numpy array containing time in the other units
    """

    if units == 'fs':
        conversion = 1 / ase_units.fs
        time_array *= conversion

    elif units == 'ps':
        conversion = 1 / (ase_units.fs * 10**3)
        time_array *= conversion

    elif units == 'ns':
        conversion = 1 / (ase_units.fs * 10**6)
        time_array *= conversion

    else:
        raise ValueError(f'Unknown time time_units: {units}')

    return time_array


@overload
def convert_ase_energy(energy_array: np.ndarray, units: str) -> np.ndarray:
    ...


@overload
def convert_ase_energy(energy_array: float, units: str) -> float:
    ...


def convert_ase_energy(
    energy_array: Union[np.ndarray, float], units: str
) -> Union[np.ndarray, float]:
    """
    Converts ASE energy units to different energy units.

    ---------------------------------------------------------------------------
    Arguments:

        energy_array: Array or a single number containing energy in ase units

        units: Name of the units to convert to

    Returns:

        (np.ndarray): Numpy array containing energy in the other units
    """

    if units.lower() == 'ev':
        pass

    elif units.lower() == 'kcal mol-1':
        energy_array *= 23.060541945329334

    elif units.lower() == 'kj mol-1':
        energy_array *= 96.48530749925793

    else:
        raise ValueError(f'Unknown energy units: {units}')

    return energy_array


def orca_output_to_npz(
    file_paths: List[str],
    out_name: str,
    out_dir: str = '.',
    load_energies: bool = True,
    load_forces: bool = True,
    load_dipole: bool = False,
    save_xyz: bool = True,
) -> None:
    """
    Generate npz file from existing outputs of orca calculations.

    -----------------------------------------------------------
    Arguments:

    file_paths: (List[str]) List of orca .out file paths to save in npz format.

    out_name: (str) Output file name without file extension.

    out_dir: (str) Output directory.

    load_energies: (bool) If True, load energies from the files.

    load_forces: (bool) If True, load forces from the files.

    # load_dipole : (bool) If True, load dipole moments form the files.
    # NOTE: (Dipole will be implement after autode modification)

    save_xyz: (bool) If True database will be saved as extxyz.
    """

    dataset = mlt.ConfigurationSet()

    logger.info(f'Processing {len(file_paths)} ORCA .out files')
    err_count = 0
    for fpath in file_paths:
        if not fpath.endswith('.out'):
            raise TypeError('Function require ORCA output file .out')

        if not os.path.exists(fpath):
            raise FileNotFoundError(f'File {fpath} was not found.')

        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        if not any('ORCA TERMINATED NORMALLY' in line for line in lines):
            logger.warning(
                f'ORCA did not terminate normally for {fpath}. Skipping...'
            )
            err_count += 1
            continue

        atoms = []

        print_coord = False

        for cline in lines:
            if 'Total Charge' in cline:
                charge = cline.split()[4]

        for line in lines:
            if 'Multiplicity' in line:
                mult = line.split()[-1]

        for line in lines:
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                atoms = []
                print_coord = True
            elif line in ['\n', '\r\n']:
                print_coord = False

            if (
                print_coord
                and '----' not in line
                and 'CARTESIAN COORDINATES (ANGSTROEM)' not in line
            ):
                element, x, y, z = line.split()
                atom = ade.atoms.Atom(
                    atomic_symbol=element,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                )
                atoms.append(atom)

        if load_energies:
            if (
                any('FINAL SINGLE POINT ENERGY' in line for line in lines)
                is False
            ):
                raise ValueError(
                    'Single point energy not found. Check the output file.'
                )

            for en_line in reversed(lines):
                if 'FINAL SINGLE POINT ENERGY' in en_line:
                    energy = PotentialEnergy(en_line.split()[4], units='Ha')

        if load_forces:
            print_forces = False

            if not any(
                ('CARTESIAN GRADIENT' in line)
                or ('The final MP2 gradient' in line)
                for line in lines
            ):
                raise ValueError('Gradients not found. Check the output file.')

            for line in lines:
                gradient_start = [
                    'CARTESIAN GRADIENT',
                    'The final MP2 gradient',
                ]

                gradient_ends = [
                    'Difference to translation invariance',
                    'Norm of the Cartesian gradient',
                    'NORM OF THE MP2 GRADIENT:',
                ]

                if any(substring in line for substring in gradient_start):
                    gradients = []
                    print_forces = True
                elif any(substring in line for substring in gradient_ends):
                    print_forces = False

                if print_forces:
                    if len(line.split()) <= 3:
                        continue
                    else:
                        dadx, dady, dadz = line.split()[-3:]

                        gradients.append(
                            [float(dadx), float(dady), float(dadz)]
                        )
                        forces = -Gradient(gradients, units='Ha a0^-1').to(
                            'Ha Å^-1'
                        )

        # Dipole implementation provided here but currently not used - waiting for autode update
        # if load_dipole:
        #    for d_line in reversed(lines):
        #        if 'Total Dipole Moment' in d_line:
        #            dipx, dipy, dipz = line.split()[-3:]
        #            dipole = [float(dipx), float(dipy), float(dipz)]

        config = mlt.Configuration(atoms=atoms, charge=charge, mult=mult)

        #   config.dipole.true = dipole.to()
        config.energy.true = energy.to('eV')
        config.forces.true = forces.to('eV Å^-1')

        dataset.append(config)

    logger.info(
        f'Successfully processed {len(dataset)} configs with {err_count} errors'
    )

    out_fpath = f'{out_dir}/{out_name}'
    logger.info(f'Saving {len(dataset)} configs to npz file: {out_fpath}.npz')
    dataset.save(out_fpath + '.npz')

    if save_xyz:
        logger.info(
            f'Saving {len(dataset)} configs to xyz file: {out_fpath}.xyz'
        )
        dataset.save_xyz(out_fpath + '.xyz', true=True)
