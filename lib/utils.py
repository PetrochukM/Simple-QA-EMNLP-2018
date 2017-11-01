from functools import lru_cache
from os import path
from os import makedirs

import ctypes
import logging
import logging.config

import torch
import yaml

from lib.text_encoders import PADDING_INDEX

logger = logging.getLogger(__name__)


def get_root_path():
    """ Get the path to the root directory
    
    Returns (str):
        Root directory path
    """
    return path.join(path.dirname(path.realpath(__file__)), '..')


DEFAULT_SAVE_DIRECTORY = path.join(get_root_path(), 'log')


def init_logging(config_path='lib/logging.yaml'):
    """ Setup logging configuration using logging.yaml.
    """
    # Only configure logging if it has not been configured yet
    if len(logging.root.handlers) == 0:
        if not path.exists('log'):
            makedirs('log')

        with open(config_path, 'rt') as file_:
            config = yaml.safe_load(file_.read())

        logging.config.dictConfig(config)


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


@lru_cache(maxsize=1)
def cuda_devices():
    """
    Checks for all CUDA devices with free memory.
    Returns:
        (list [int]) the CUDA devices available
    """

    # Find Cuda
    cuda = None
    for libname in ('libcuda.so', 'libcuda.dylib', 'cuda.dll'):
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break

    # Constants taken from cuda.h
    CUDA_SUCCESS = 0

    num_gpu = ctypes.c_int()
    error = ctypes.c_char_p()
    free_memory = ctypes.c_size_t()
    total_memory = ctypes.c_size_t()
    context = ctypes.c_void_p()
    device = ctypes.c_int()
    ret = []  # Device IDs that are not used.

    def run(result, func, *args):
        nonlocal error
        result = func(*args)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error))
            logger.warn("%s failed with error code %d: %s", func.__name__, result,
                        error.value.decode())
            return False
        return True

    # Check if Cuda is available
    if not cuda:
        return ret

    result = cuda.cuInit(0)

    # Get number of GPU
    if not run(result, cuda.cuDeviceGetCount, ctypes.byref(num_gpu)):
        return ret

    for i in range(num_gpu.value):
        if (not run(result, cuda.cuDeviceGet, ctypes.byref(device), i) or
                not run(result, cuda.cuDeviceGet, ctypes.byref(device), i) or
                not run(result, cuda.cuCtxCreate, ctypes.byref(context), 0, device) or
                not run(result, cuda.cuMemGetInfo,
                        ctypes.byref(free_memory), ctypes.byref(total_memory))):
            continue

        percent_free_memory = float(free_memory.value) / total_memory.value
        logger.info('CUDA device %d has %f free memory [%d MiB of %d MiB]', i, percent_free_memory,
                    free_memory.value / 1024**2, total_memory.value / 1024**2)
        if percent_free_memory > 0.98:
            logger.info('CUDA device %d is available', i)
            ret.append(i)

        cuda.cuCtxDetach(context)

    return ret


def get_total_parameters(model):
    """ Return the total number of trainable parameters in model """
    params = filter(lambda p: p.requires_grad, model.parameters())
    return sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params)


def add_logger_file_handler(filename, level=logging.DEBUG, formatter=None):
    """
    Add a filehandler to the root logger.
    
    Useful to store a copy of the logs during training or evaluation.
    """
    logger = logging.getLogger()  # Root logger
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    if formatter is None:
        handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)


def pad(batch):
    """ Pad a list of tensors with PADDING_INDEX. Sort by decreasing lengths as well. """
    # PyTorch RNN requires batches to be sorted in decreasing length order
    lengths = [len(row) for row in batch]
    max_len = max(lengths)
    padded = []
    for row in batch:
        n_padding = max_len - len(row)
        padding = torch.LongTensor(n_padding * [PADDING_INDEX])
        padded.append(torch.cat((row, padding), 0))
    return padded, lengths


def collate_fn(batch, input_key, output_key, sort_key=None, preprocess=pad):
    """ Collate a batch of tensors not ready for training to padded, sorted, transposed,
    contiguous and cuda tensors ready for training. Used with torch.utils.data.DataLoader. """
    if sort_key:
        batch = sorted(batch, key=lambda row: len(row[sort_key]), reverse=True)
    input_batch, input_lengths = preprocess([row[input_key] for row in batch])
    output_batch, output_lengths = preprocess([row[output_key] for row in batch])

    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    ret = {}
    ret[input_key] = [torch.stack(input_batch).t_().contiguous(), torch.LongTensor(input_lengths)]
    ret[output_key] = [
        torch.stack(output_batch).t_().contiguous(), torch.LongTensor(output_lengths)
    ]
    for key in batch[0].keys():
        if key not in [input_key, output_key]:
            ret[key] = [row[key] for row in batch]

    ret[input_key] = tuple(ret[input_key])
    ret[output_key] = tuple(ret[output_key])
    return ret
