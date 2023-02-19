import os
import re
import shutil
from tempfile import mkdtemp
from functools import wraps
from typing import Optional, List


def work_in_tmp_dir(kept_exts:   Optional[List[str]] = None,
                    copied_exts: Optional[List[str]] = None):
    """
    Execute a function in a temporary directory

    ---------------------------------------------------------------------------
    Arguments:
        kept_exts: File extensions copied back from the tmp dir

        copied_exts: File extensions that are copied to the tmp dir
    """

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here_path = os.getcwd()
            tmpdir_path = mkdtemp()

            for item in os.listdir(here_path):

                if copied_exts is None:
                    continue

                if any(item.endswith(ext) for ext in copied_exts):
                    # logger.info(f'Copying {item}')
                    shutil.copy(item, tmpdir_path)

            # Move directories and execute
            os.chdir(tmpdir_path)
            out = func(*args, **kwargs)

            if kept_exts is not None:
                for filename in os.listdir(tmpdir_path):
                    if any(filename.endswith(ext) for ext in kept_exts):
                        shutil.copy(src=filename,
                                    dst=os.path.join(here_path, filename))

            os.chdir(here_path)

            # Remove the temporary dir with all files and return the output
            shutil.rmtree(tmpdir_path)

            return out

        return wrapped_function
    return func_decorator


def work_in_dir(dirname: str):
    """
    Execute a function in a different directory.

    ---------------------------------------------------------------------------
    Arguments:

        dirname (str): Name of the directory
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


def unique_filename(filename: str) -> str:
    """
    Return a unique filename based on not clashing with other files with the
    same. Append 0, 1... iteratively until something unique is found.

    ---------------------------------------------------------------------------
    Arguments:

        filename (str): Name of the file

    Returns:

        (str): Unique name which contains the name of the file as a root
    """
    if '.' not in filename:
        raise ValueError('Filename must have an extension to be uniquified!')

    basename = ".".join(filename.split('.')[:-1])
    ext = filename.split('.')[-1]

    if not _name_exists(basename, ext):
        return filename

    old_basename = basename
    i = 0
    while _name_exists(basename, ext):
        basename = f'{old_basename}_{i}'
        i += 1

    return f'{basename}.{ext}'


def unique_dirname(dirname: str) -> str:
    """
    Return a unique directory name based on not clashing with other files with
    the same. Append 0, 1... iteratively until something unique is found.

    ---------------------------------------------------------------------------
    Arguments:

        dirname (str): Name of the directory

    Returns

        (str): Unique name which contains the name of the directory as a root
    """
    if not _name_exists(dirname):
        return dirname

    old_dirname = dirname
    i = 0
    while _name_exists(dirname):
        dirname = f'{old_dirname}_{i}'
        i += 1

    return dirname


def move_files(moved_exts: List[str], folder: str) -> None:
    """
    Move files with given extensions from the current directory to a new
    directory specified by the folder.

    ---------------------------------------------------------------------------
    Arguments:

        moved_exts (List[str]): List of extentions specifying which files
                                are moved

        folder (str): Name of the new directory where files are moved.
    """

    if not os.path.exists(folder):
        os.mkdir(folder)

    for filename in os.listdir():
        if any(filename.endswith(ext) for ext in moved_exts):
            destination = os.path.join(folder, filename)
            shutil.move(src=filename, dst=destination)

    return None


def _name_exists(basename: str,
                 extension: Optional[str] = None) -> bool:
    """Return a bool based on whether a file or a folder with the given name
    already exists in the current directory"""

    if extension is None:
        return os.path.exists(basename)

    else:
        return os.path.exists(f'{basename}.{extension}')


def _newest_dirname(basename: str) -> str:
    """Return a dirname with a highest coefficient which would correspond to
    the newest directory created with a given basename using unique_dirname"""

    _basenames = []
    for filename in os.listdir():
        if basename in filename:
            _basenames.append(filename)

    if len(_basenames) == 1:
        newest_dirname = basename

    else:
        index_pattern = re.compile(r'(?<=_)(\d+)(?=\.|$)')
        largest_index = 0

        for _basename in _basenames:
            matched_object = index_pattern.search(_basename)

            if matched_object is not None:
                index = int(matched_object.group())

                if index > largest_index:
                    largest_index = index

        newest_dirname = f'{basename}_{largest_index}'

    return newest_dirname
