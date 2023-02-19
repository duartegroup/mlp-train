import os
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

            for item in os.listdir(os.getcwd()):

                if copied_exts is None:
                    continue

                if any(item.endswith(ext) for ext in copied_exts):
                    # logger.info(f'Copying {item}')
                    shutil.copy(item, tmpdir_path)

            # Move directories and execute
            os.chdir(tmpdir_path)
            out = func(*args, **kwargs)

            if kept_exts is not None:
                for filename in os.listdir(os.getcwd()):
                    if any(filename.endswith(ext) for ext in kept_exts):
                        shutil.copy(src=filename,
                                    dst=os.path.join(here_path, filename))

            os.chdir(here_path)

            # Remove the temporary dir with all files and return the output
            shutil.rmtree(tmpdir_path)

            return out

        return wrapped_function
    return func_decorator


def work_in_dir(dirname: str,
                moved_exts: Optional[List[str]] = None):
    """
    Execute a function in a different directory.

    ---------------------------------------------------------------------------
    Arguments:

        dirname (str): Name of the directory

        moved_exts (Optional[List[str]]): File extentions that are moved back
                                          from the different directory
    """

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here_path = os.getcwd()
            dir_path = os.path.join(here_path, dirname)

            os.chdir(dir_path)
            out = func(*args, **kwargs)

            if moved_exts is not None:
                for filename in os.listdir(os.getcwd()):
                    if any(filename.endswith(ext) for ext in moved_exts):
                        shutil.move(src=filename,
                                    dst=os.path.join(here_path, filename))

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


def move_files(moved_ext: str, folder: str) -> None:
    """
    In the current directory make a new directory and move all files containing
    a specific extention to that folder.

    ---------------------------------------------------------------------------
    Arguments:

        moved_ext (str): Extention with which files are moved

        folder (str): Name of the new directory where files are moved. If a
                      directory with the specified name already exists a unique
                      name to the new directory is generated
    """
    unique_folder = unique_dirname(folder)
    os.mkdir(unique_folder)

    for file in os.listdir():
        ext = f'.{file.split(".")[-1]}'

        if ext == moved_ext:
            destination = os.path.join(unique_folder, file)
            shutil.move(file, destination)

    return None


def _name_exists(basename: str,
                 extension: Optional[str] = None) -> bool:
    """Return a bool based on whether a file or a folder with the given name
    already exists in the current directory"""

    if extension is None:
        return os.path.exists(basename)

    else:
        return os.path.exists(f'{basename}.{extension}')
