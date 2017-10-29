import logging
import os
import sys
import tempfile
import time

from seq2seq.config import configurable
from seq2seq.fields.utils import build_input_vocab
from seq2seq.fields.utils import build_output_vocab
from seq2seq.fields.utils import get_input_field
from seq2seq.fields.utils import get_output_field
from seq2seq.util.util import get_variable_from_string
from seq2seq.util.util import call_filter_kwargs

import seq2seq

logger = logging.getLogger(__name__)


@configurable
def get_dataset(dataset,
                data_directory='.',
                input_field='SeqField',
                output_field='SeqField',
                load_test=True,
                load_dev=True,
                load_train=True,
                share_vocab=False,
                include_test_dev_vocab=False,
                checkpoint=None):
    """ Factory method to get a dataset.

    Details:
        - Downloads the datasets and/or loads it from disc
        - Tokenize and preprocesses
        - Builds a vocabulary
    Args:
        dataset (str or class): dataset to load
        data_directory (str): where to load data from 
        input_field (str or class): input_field to load
        output_field (str or class): output_field to load
        load_test (bool): if load test data
        load_dev (bool): if load dev data
        load_train (bool): if load train data
        share_vocab (bool): share vocab between input and output
        include_test_dev_vocab (bool): add test and dev tokens to vocabulary; careful, that this
            does not overfit to test.
        checkpoint (Checkpoint or None): checkpoint to load fields from
    Returns:
        *datasets: datasets loaded
        input_field: input field loaded
        output_field: output field loaded
    """
    input_field = get_input_field(input_field, checkpoint)
    output_field = get_output_field(output_field, checkpoint)
    dataset_class = get_variable_from_string(dataset, [seq2seq.datasets])

    # Load the dataset
    dataset_class_kwargs = {}
    if not load_test:
        dataset_class_kwargs['test'] = None
    if not load_dev:
        dataset_class_kwargs['dev'] = None
    if not load_train:
        dataset_class_kwargs['train'] = None
    logger.info('Loading `%s`', dataset)

    datasets = call_filter_kwargs(
        dataset_class.splits,
        input_field=input_field,
        output_field=output_field,
        data_directory=data_directory,
        **dataset_class_kwargs)

    if not checkpoint and load_train:
        logger.info('Building vocab')
        train_data = datasets[0]
        if share_vocab:
            if include_test_dev_vocab:
                examples = []
                for dataset in datasets:
                    examples += list(dataset.input) + list(dataset.output)
            else:
                examples = list(train_data.input) + list(train_data.output)
            build_input_vocab(input_field, examples)
            build_output_vocab(output_field, examples)
        else:
            if include_test_dev_vocab:
                input_data = []
                output_data = []
                for dataset in datasets:
                    input_data += list(dataset.input)
                    output_data += list(dataset.output)
            else:
                input_data = train_data.input
                output_data = train_data.output
            build_input_vocab(input_field, input_data)
            build_output_vocab(output_field, output_data)

    if checkpoint or load_train:
        logger.info('Input Vocab Size: %d', len(input_field.vocab))
        logger.info('Output Vocab Size: %d', len(output_field.vocab))
    else:
        logger.warn('Did not build Vocab. Need to `load_train` to build vocab.')

    logger.info('Preprocessing `%s` with vocab', dataset)
    [dataset.preprocess_with_vocab() for dataset in datasets]
    [dataset.print_sample() for dataset in datasets]
    return tuple(list(datasets) + [input_field, output_field])


def urlretrieve_reporthook(count, block_size, total_size):
    """
    `reporthook` for `urllib.request.urlretrieve`.

    During a download the report hook prints the progress of the download.
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%s - DOWNLOAD - %d%%, %d MB, %d KB/s, %d seconds passed" %
                     (__name__, percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
    is_done = percent == 100
    if is_done:
        print('\nDone!')


def extract_and_rename_tar_member(tar, filename, member, tmp_directory=tempfile.gettempdir()):
    """
    Extract `member` from `tar` and rename as `filename`. 
    `data_directory` is used to store the temporary file before it's renamed.

    Args:
        tar (TarFile): compressed file to extract from
        data_directory (str)
        filename (str)
        member (str)
    """
    logger.info('Renaming %s to %s', member, filename)
    tmp_path = os.path.join(tmp_directory, member)
    if not os.path.isfile(tmp_path):
        tar.extract(path=tmp_directory, member=member)
    os.rename(tmp_path, filename)
    logger.info('Renamed %s to %s', member, filename)
