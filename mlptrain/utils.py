import os
import re
import shutil
import numpy as np
from tempfile import mkdtemp
from functools import wraps
from typing import Optional, List, Sequence, Union
from ase import units as ase_units
import mlptrain as mlt
from mlptrain.log import logger


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
) -> np.ndarray:
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


def convert_ase_energy(
    energy_array: Union[np.ndarray, float], units: str
) -> np.ndarray:
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


def npz_to_xyz(npz_filename: str) -> None:
    """
    Converts a Trajectory or ConfigurationSet saved as an .npz file to a .xyz file.

    ---------------------------------------------------------------------------
    Arguments:

        npz_file: (str) Name of the .npz file to be converted, eg. "my_data.npz"


    Creates a .xyz file named the same way, eg. -> 'my_data.xyz'

    Returns: None
    """

    if npz_filename[-4:] != '.npz':
        raise ValueError('Input filename must end with .npz extension.')

    xyz_filename = npz_filename[:-4] + '.xyz'

    data = mlt.ConfigurationSet()

    if os.path.exists(npz_filename):
        data.load(npz_filename)
        data.save(xyz_filename)
        logger.info(f'Converted {npz_filename} to {xyz_filename}')

    else:
        raise FileNotFoundError(f'File {npz_filename} not found.')

    return None
