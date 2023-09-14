import sys
import types
import os

from src.lib.sutils import *

ROOT_DIR = os.environ.get('SL_ROOT_DIR', None)
if ROOT_DIR is not None:
    ROOT_DIR = Path(ROOT_DIR)

if ROOT_DIR is None or not ROOT_DIR.exists() or not ROOT_DIR.is_dir():
    ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / 'data'
safe_create_dir(DATA_DIR)

RESULT_DIR = ROOT_DIR / 'results'
safe_create_dir(RESULT_DIR)


def str_to_class(field):
    """
    Converts a str to call a class
    :param field: str
        Name of the function
    :return: str
        The function if found
    """
    try:
        identifier = getattr(sys.modules[__name__], field)
    except AttributeError:
        raise NameError("%s doesn't exist." % field)
    if isinstance(identifier, (types.ClassType, types.TypeType)):
        return identifier
    raise TypeError("%s is not a class." % field)


def get_safe_data_file(file_path: Path):
    """
    Parameters
    ----------
    file_path: :obj:`pathlib.Path`
    Data file path, if it is absolute will use as is, otherwise will look under data folder
    Returns
    -------
    normalized data file path
    """
    file_path = get_safe_path_obj(file_path)
    if file_path.is_absolute():
        return file_path
    return DATA_DIR / file_path


def ensure_dir(file_path):
    """
    :param file_path: str or Path object
        Path of the file to check
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
