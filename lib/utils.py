from functools import lru_cache

import atexit
import ctypes
import logging
import logging.config
import os
import time

import random
import torch

from lib.text_encoders import PADDING_INDEX
from lib.configurable import log_config

logger = logging.getLogger(__name__)


def get_root_path():
    """ Get the path to the root directory
    
    Returns (str):
        Root directory path
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


def get_log_directory_path(label='', parent_directory='logs/'):
    """
    Get a save directory that includes start time and execution time.

    The execution time is used as the main sorting key to help distinguish between finished,
    halted, and running experiments.
    """
    start_time = time.time()
    name = '0000.%s.%s' % (time.strftime('%m-%d_%H:%M:%S', time.localtime()), label)
    path = os.path.join(parent_directory, name)

    def exit_handler():
        if os.path.exists(path):
            # Add runtime to the directory name sorting
            difference = int(time.time() - start_time)
            os.rename(path, path.replace('0000', '%04d' % difference))

    atexit.register(exit_handler)
    os.makedirs(path)
    return path


_init_logging_return = None


def init_logging(log_directory):
    """ Setup logging in log_directory.

    Returns the same log_directory after init_logging is setup for the first time.
    """
    global _init_logging_return
    # Only configure logging if it has not been configured yet
    if _init_logging_return is not None:
        return _init_logging_return

    logging.config.dictConfig({
        'formatters': {
            'simple': {
                'format': '%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'info_file_handler', 'error_file_handler']
        },
        'disable_existing_loggers': False,
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'error_file_handler': {
                'class': 'logging.handlers.RotatingFileHandler',
                'encoding': 'utf8',
                'filename': os.path.join(log_directory, "error.log"),
                'backupCount': 2,
                'level': 'ERROR',
                'formatter': 'simple'
            },
            'info_file_handler': {
                'class': 'logging.handlers.RotatingFileHandler',
                'encoding': 'utf8',
                'filename': os.path.join(log_directory, "info.log"),
                'backupCount': 2,
                'level': 'INFO',
                'formatter': 'simple',
                'maxBytes': 10485760
            }
        },
        'version': 1
    })

    _init_logging_return = log_directory

    return log_directory


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
    ret[input_key] = [
        torch.stack(input_batch).t_().squeeze(0).contiguous(),
        torch.LongTensor(input_lengths)
    ]
    ret[output_key] = [
        torch.stack(output_batch).t_().squeeze(0).contiguous(),
        torch.LongTensor(output_lengths)
    ]
    for key in batch[0].keys():
        if key not in [input_key, output_key]:
            ret[key] = [row[key] for row in batch]

    ret[input_key] = tuple(ret[input_key])
    ret[output_key] = tuple(ret[output_key])
    return ret


def setup_training(dataset, checkpoint_path, log_directory, device, random_seed):
    """ Utility function to settup logger, hyperparameters, seed, device and checkpoint """
    # Prevent a circular dependency
    from lib.checkpoint import Checkpoint

    # Setup Device
    device = device_default(device)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    logger.info('Device: %s', device)

    # Random Seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
    logger.info('Seed: %s', random_seed)

    # Load Checkpoint
    if checkpoint_path:
        checkpoint = Checkpoint(checkpoint_path, device)
    else:
        checkpoint = Checkpoint.recent(log_directory, device)

    # Log the global configuration before starting to train.
    log_config()

    return checkpoint


def torch_equals_ignore_index(target, prediction, ignore_index=None):
    """
    Compute torch.equals with the optional mask parameter.
   
    Args:
        ignore_index (int, optional): specifies a target index that is ignored
    Returns:
        (bool) iff target and prediction are equal
    """
    if ignore_index is not None:
        mask_arr = target.ne(ignore_index)
        target = target.masked_select(mask_arr)
        prediction = prediction.masked_select(mask_arr)

    return torch.equal(target, prediction)
