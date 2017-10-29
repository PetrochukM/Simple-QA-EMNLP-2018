import logging
import math
import os
import random

import torchtext
import pandas as pd

from seq2seq.fields import SeqField
from seq2seq.config import configurable

logger = logging.getLogger(__name__)


class SeqInputOutputDataset(torchtext.data.Dataset):
    """
    Base class for all `seq2seq.datasets`.

    Dataset with 'Input' and 'Output' fields set to SeqField. Used by the rest of the library to
    learn.
    """

    input_column = 'input'
    output_column = 'output'

    @configurable
    def __init__(self, examples, fields, percent_data=1.0, filter_=None):
        assert not isinstance(examples,
                              str), """Examples must be a list of examples. Possibly attempting to
                                instantiate SeqInputOutputDataset from a file."""
        fields = dict(fields)
        assert set(fields.keys()) == set(
            [self.input_column, self.output_column]), """InputOutputDataset may only have input and
                output fields"""
        assert isinstance(fields['input'], SeqField), """Input field must be of type SeqField"""
        assert isinstance(fields['output'], SeqField), """Output field must be of type SeqField"""
        assert fields['output'].eos_token == SeqField.EOS_TOKEN
        examples = random.sample(examples, math.floor(len(examples) * percent_data))
        self.sort_key = lambda x: len(x.input)
        super().__init__(examples=examples, fields=fields, filter_pred=filter_)

    @configurable
    def print_sample(self, n_samples=5):
        """ Randomly sample some of the rows and columns and print """
        examples = random.sample(self.examples, n_samples)
        data = []
        for example in examples:
            row = [getattr(example, column) for column in [self.input_column, self.output_column]]
            data.append(row)
        logger.info('Data Sample:\n%s',
                    pd.DataFrame(data, columns=[self.input_column, self.output_column]))

    def preprocess_with_vocab(self):
        """ Run some `field.preprocess_with_vocab` on every example """
        for example in self.examples:
            for name, field in self.fields.items():
                setattr(example, name, field.preprocess_with_vocab(getattr(example, name)))

    @classmethod
    def shuffle_datasets(cls, *datasets):
        """
        Given a list of a datasets, shuffle together their data and resplit them.
        Args:
            *datasets (Dataset) list of datasets
        """
        lengths = []
        examples = []
        input_field = datasets[0].fields['input']
        output_field = datasets[0].fields['output']
        for dataset in datasets:
            assert isinstance(dataset, SeqInputOutputDataset), """Shuffle for SeqInputOutputDataset
                as they are guaranteed to have the same field names"""
            assert input_field == dataset.fields['input'], "Shuffle requires same `input_field`"
            assert output_field == dataset.fields['output'], "Shuffle requires same `output_field`"
            lengths.append(len(dataset))
            examples.extend(dataset.examples)
        random.shuffle(examples)
        for i, length in enumerate(lengths):
            dataset = datasets[i]
            dataset.examples = examples[:length]
            examples = examples[length:]

    @classmethod
    def splits(cls,
               input_field,
               output_field,
               data_directory='.',
               train='train.tsv',
               test='test.tsv',
               dev='dev.tsv',
               **kwargs):
        """Create dataset objects for splits of the a dataset.

        Arguments:
            input_field (torchtext.data.Field)
            output_field (torchtext.data.Field)
            data_directory (str): The directory containing train, test, and val
            train: The filename that contains the training examples
            test: The filename that contains the test examples
            dev: The filename that contains the dev examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        def make(filename):
            if filename is None:
                return None
            full_path = os.path.join(data_directory, filename)
            return cls(full_path, input_field, output_field, **kwargs)

        return tuple(d for d in (make(train), make(dev), make(test)) if d is not None)
