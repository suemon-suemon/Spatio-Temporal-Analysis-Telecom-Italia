# fix pythonpath when working locally
import sys
from os import getcwd
from os.path import basename, dirname


def fix_python_path_if_working_locally():
    """Add the parent path to pythonpath if current working dir is darts/examples"""
    cwd = getcwd()
    if basename(cwd) == "experiments":
        sys.path.insert(0, dirname(cwd))
    else:
        sys.path.insert(0, cwd)