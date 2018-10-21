import logging
import logging.config
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import torch

from torchnlp.datasets import Dataset

logger = logging.getLogger(__name__)

FB2M_KG = '../data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt'
FB5M_KG = '../data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt'

# Get the path relative to the directory this file is in
_directory_path = os.path.dirname(os.path.realpath(__file__))
FB2M_KG = os.path.realpath(os.path.join(_directory_path, FB2M_KG))
FB5M_KG = os.path.realpath(os.path.join(_directory_path, FB5M_KG))

FB2M_KG_TABLE = 'fb_two_kg'
FB5M_KG_TABLE = 'fb_five_kg'
FB2M_NAME_TABLE = 'fb_two_subject_name'


def resplit_datasets(dataset, other_dataset, random_seed=None, cut=None):
    """ Deterministic shuffle and split algorithm.

    Given the same two datasets and the same `random_seed`, the split happens the same exact way
    every call.

    Args:
        dataset (torchnlp.datasets.Dataset)
        other_dataset (torchnlp.datasets.Dataset)
        random_seed (int, optional)
        cut (float, optional): float between 0 and 1 to cut the dataset; otherwise, the same
            proportions are kept.
    Returns:
        dataset (torchnlp.datasets.Dataset)
        other_dataset (torchnlp.datasets.Dataset)
    """
    concat = dataset.rows + other_dataset.rows
    # Reference:
    # https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result
    # NOTE: Shuffle the same way every call of `shuffle_datasets` where the `random_seed` is given
    random.Random(random_seed).shuffle(concat)
    if cut is None:
        return Dataset(concat[:len(dataset)]), Dataset(concat[len(dataset):])
    else:
        cut = max(min(round(len(concat) * cut), len(concat)), 0)
        return Dataset(concat[:cut]), Dataset(concat[cut:])


def config_logging():
    """ Configure the root logger with basic settings.
    """
    logging.basicConfig(
        format='[%(asctime)s][%(processName)s][%(name)s][%(levelname)s] %(message)s',
        level=logging.INFO,
        stream=sys.stdout)


def get_root_path():
    """ Get the path to the root directory

    Returns (str):
        Root directory path
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


def new_experiment_folder(label='', parent_directory='experiments/'):
    """
    Get a experiment directory that includes start time.
    """
    start_time = time.time()
    name = '%s.%s' % (label, time.strftime('%m_%d_%H:%M:%S', time.localtime()))
    path = os.path.join(parent_directory, name)
    os.makedirs(path)

    # TODO: If the folder is empty then delete it after the execution finishes

    return path


# Reference:
# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batch(iterable, n=1):
    if not hasattr(iterable, '__len__'):
        # Slow version if len is not defined
        current_batch = []
        for item in iterable:
            current_batch.append(item)
            if len(current_batch) == n:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch
    else:
        # Fast version is len is defined
        for ndx in range(0, len(iterable), n):
            yield iterable[ndx:min(ndx + n, len(iterable))]


# Reference: https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
class StreamFork(object):

    def __init__(self, filename, stream):
        self.stream = stream
        self.file_ = open(filename, 'a')

    @property
    def closed(self):
        return self.file_.closed and self.stream.closed

    def write(self, message):
        self.stream.write(message)
        self.file_.write(message)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

    def flush(self):
        self.file_.flush()
        self.stream.flush()

    def close(self):
        self.file_.close()
        self.stream.close()


def save_standard_streams(directory=''):
    """
    Save stdout and stderr to a `{directory}/stdout.log` and `{directory}/stderr.log`.
    """
    sys.stdout = StreamFork(os.path.join(directory, 'stdout.log'), sys.stdout)
    sys.stderr = StreamFork(os.path.join(directory, 'stderr.log'), sys.stderr)


def device_default(device=None):
    """
    Using torch, return the default device to use.
    Args:
        device (int or None): -1 for CPU, None for default GPU or CPU, and 0+ for GPU device ID
    Returns:
        device (int or None): -1 for CPU and 0+ for GPU device ID
    """
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    return device


def get_total_parameters(model):
    """ Return the total number of trainable parameters in model """
    params = filter(lambda p: p.requires_grad, model.parameters())
    return sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params)


def seed(random_seed, is_cuda=False):
    """
    Attempt to apply a `random_seed` is every possible library that may require it. Our goal is
    to make our software reporducible.
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if is_cuda:
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    logger.info('Seed: %s', random_seed)


def torch_equals_ignore_index(tensor, tensor_other, ignore_index=None):
    """
    Compute torch.equals with the optional mask parameter.

    Args:
        ignore_index (int, optional): specifies a tensor1 index that is ignored
    Returns:
        (bool) iff target and prediction are equal
    """
    if ignore_index is not None:
        assert tensor.size() == tensor_other.size()
        mask_arr = tensor.ne(ignore_index)
        tensor = tensor.masked_select(mask_arr)
        tensor_other = tensor_other.masked_select(mask_arr)

    return torch.equal(tensor, tensor_other)


def get_connection():
    # Load .env file
    pass_ = {}

    # Get the path relative to the directory this file is in
    _directory_path = os.path.dirname(os.path.realpath(__file__))
    pass_path = os.path.join(_directory_path, '../.pass')
    for line in open(pass_path):
        split = line.strip().split('=')
        pass_[split[0]] = split[1]

    # Connect
    return psycopg2.connect(
        dbname=pass_['DB_NAME'],
        port=pass_['DB_PORT'],
        user=pass_['DB_USER'],
        host=pass_['DB_HOST'],
        password=pass_['DB_PASS'])


def format_pipe_table(*args, **kwargs):
    # Rows is a dictionary of keys
    df = pd.DataFrame(*args, **kwargs)
    ret = ''

    columns = ['Index'] + list(df.columns.values)

    # Header
    ret += '| ' + ' | '.join(columns) + ' |\n'
    ret += '| ' + ' | '.join(['---' for _ in columns]) + ' |\n'

    # Add values
    for index, row in df.iterrows():
        values = [index] + list(row)
        values = [str(v) for v in values]
        ret += '| ' + ' | '.join(values) + ' |\n'
    return ret