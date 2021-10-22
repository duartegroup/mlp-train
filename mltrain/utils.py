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


def unique_filename(filename: str) -> str:
    """
    Return a unique filename based on not clashing with other files with the
    same. Append 0, 1... iteratively until something unique is found

    Arguments:
        filename:

    Returns:
        (str):
    """
    if '.' not in filename:
        raise ValueError('Filename must have an extension to be uniquified!')

    basename = ".".join(filename.split('.')[:-1])
    ext = filename.split('.')[-1]

    def any_exist():
        """Do any of the filenames with the possible extensions exist?"""
        return os.path.exists(f'{basename}.{ext}')

    if not any_exist():
        return filename

    old_basename = basename
    i = 0
    while any_exist():
        basename = f'{old_basename}{i}'
        i += 1

    return f'{basename}.{ext}'
