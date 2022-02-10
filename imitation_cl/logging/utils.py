import os
import datetime
import json
import logging
import inspect
import numpy as np
from pprint import pprint
import itertools
import glob

class Dictobject(object):
    """
    Used for accessing dict items as attributes
    """
    def __init__(self, d):
        self.__dict__ = d

def read_dict(json_path):
    """Reads a json file into a dict and returns it.

    Args:
        json_path (str): Path

    Returns:
        dict: Python dict
    """
    json_dict = None
    with open(json_path, "r") as json_file:
        json_dict = json.load(json_file)
    return json_dict

def write_dict(json_path, json_dict):
    """Writes dict to json file.

    Args:
        json_path (str): Path
        json_dict (dict): Python dict
    """
    with open(json_path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4)

def read_numpy(numpy_file_path):
    """Reads a numpy archive file and returns the array

    Args:
        numpy_file_path (str): Path of the numpy archive file

    Returns:
        numpy arr: The array read from the file.
    """
    arr = np.load(numpy_file_path, mmap_mode='r')
    return arr

def get_id():
    """Creates a unique string ID based on the name of the calling
       script and the timestamp.

    Returns:
        str: Identifier string
    """
    # Get the name of the script from which get_id is called
    # https://www.stefaanlippens.net/python_inspect/
    caller_filename = os.path.basename(inspect.stack()[1][1])[0:-3]

    # Timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create an ID
    identifier = f"{caller_filename}_{timestamp}"

    return identifier

def custom_logging_setup(args):
    """
    Sets up logging directories, saves commandline args to a file,
    and enables saving a dump of the console logs
    """

    # Set-up output directories
    identifier = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    identifier = f"{identifier}_seed{args.seed}"
    save_folder = os.path.join(args.log_dir, args.description, identifier)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Make a folder for storing trained models
    model_dir = os.path.join(save_folder, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    return save_folder, identifier

def count_lines(file_name):
    """
    Counts number of non-empty lines in the file `file_name`

    Args:
        file_name (str): Path to file

    Returns:
        int: Number of non-empty lines
        list(str): List of strings, one str for each line
    """
    with open(file_name) as f:
        lines = [line.rstrip() for line in f.readlines()]
    nonblanklines = [line for line in lines if line]
    return len(nonblanklines), nonblanklines


### Functions for checking if all the required log files are present ###

def build_dirtree(startpath, level=0):
    """Recurses down a directory tree and returns
    a dict containing the tree.

    Args:
        startpath (str): Root directory
        level (int, optional): Level of the root directory. Defaults to 0.

    Returns:
        dict: Python dict containing the directory tree
    """
    s = os.path.basename(startpath)
    tree_dict={'self': s, 'level': level, 'children': list()}
    files = os.listdir(startpath)
    for f in files:
        f = os.path.join(startpath, f)
        if os.path.isdir(f):
            tree_dict['children'].append(build_dirtree(f, level=level+1))
        elif os.path.isfile(f):
            tree_dict['children'].append({'self': os.path.basename(f), 'level': level+1, 'children': list()})
    return tree_dict

def pprint_dirtree(tree_dict):
    """Prints the directory tree in a pretty format.
    Use `build_dirtree` to create the directory tree
    dictionary.

    Args:
        tree_dict (dict): Python dict containing the directory tree
    """
    indent = '|   ' * tree_dict['level']
    print(f'{indent}{tree_dict["self"]}')
    for c in tree_dict['children']:
        pprint_dirtree(c)

def check_dirtree(root_dir, check, ignore_dir_id='_processed'):
    """Checks a directory tree to make sure that all required
    files (as specified in the dict `check`) are present.

    For example, `check` can be a dict like:

    check = {1: {'prefix': ['tr_node','tr_ft_node','tr_si_node', 'tr_mas_node', 'tr_mas_node', 'tr_chn_node', 'tr_hn_node'], 
                'suffix': []},
             2: {'prefix': [], 
                'suffix': ['seed200', 'seed400', 'seed800', 'seed1000', 'seed1200']},
             3: {'prefix': ['models', 'eval_results.json', 'log.log', 'plot_trajectories*.pdf', 'commandline_args.json'], 
                'suffix': []},
            }
    Here the keys (1,2,3) denote the leves in the tree starting at `root_dir` (level 0) and
    the prefix and suffix corresponding to each key denote the prefix/suffix of the files/subdirs
    which should be present at that level. A cross-product is performed between the prefix/suffix
    lists of the different levels to get all the paths that should be present.

    Args:
        root_dir (str): Root dir
        check (dict): See docstring above
        ignore_dir_id (str, optional): Paths containing this substring will be ignored. 
                                       Defaults to '_processed'.
    """

    levels = sorted(list(check.keys()))

    level_list_map = dict()

    for level in levels:
        level_list_map[level] = list()
        if len(check[level]['prefix'])==0:
            prefix_list = ['' for item in check[level]['suffix']]
        else:
            prefix_list = check[level]['prefix']
        if len(check[level]['suffix'])==0:
            suffix_list = ['' for item in check[level]['prefix']]
        else:
            suffix_list = check[level]['suffix']

        for prefix, suffix in zip(prefix_list, suffix_list):
            level_list_map[level].append(f'{prefix}*{suffix}')

    check_paths_tup = (list(itertools.product(*[level_list_map[i] for i in levels])))
    check_paths_list = list()

    for cp in check_paths_tup:
        check_paths_list.append(os.path.sep.join(cp))

    # Check if all paths are present
    error_occured = False
    # Remove the procesed dir (avoid double counting)
    for path in check_paths_list:
        path = os.path.join(root_dir, path)
        found_paths = glob.glob(path)
        for fp in found_paths:
            if ignore_dir_id in fp:
                found_paths.remove(fp)

        if len(found_paths)!=1:
            print(f'Path {path} not found')
            error_occured = True

    if not error_occured:
        print('All required log files/dirs are present')

    return 1 if error_occured else 0