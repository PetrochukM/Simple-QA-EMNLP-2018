import os
import logging

from torchtext.data import Example

import pandas as pd
import sh

from lib.config import get_root_path
from lib.config import init_logging
from seq2seq.datasets.seq_input_output_dataset import SeqInputOutputDataset

logger = logging.getLogger(__name__)

# TODO: Allow access to original input before field for replace_unk feature.

# _SimpleQA downloads and processes large files; therefore, we add `pragma: no cover`
# Not reasonable to test this during test time


class _SimpleQA(SeqInputOutputDataset):  # pragma: no cover
    """
    Base class for simple question answering tasks.

    The class is responsible for downloading to disc the simple question answering data.
    """

    url = 'hdfs://hdpb.prn.parsec.apple.com/user/mpetrochuk/simple_qa'
    directory_name = 'simple_qa'
    n_rows = {'train.tsv': 74526, 'dev.tsv': 10639, 'test.tsv': 21300}

    @classmethod
    def download(cls, data_directory='.'):
        """
        Download simple questions data if `data_directory/${cls.directory_name}` does not exist.

        Args:
            data_directory (str): directory where all data is stored
        Returns:
            (str) `data_directory/${cls.directory_name}`
        """
        path = os.path.join(get_root_path(), data_directory, cls.directory_name)
        if not os.path.isdir(path):
            # Download
            logger.info('Downloading %s', cls.url)
            sh.hdfs('dfs', '-copyToLocal', cls.url, data_directory)
            logger.info('Downloaded %s', os.listdir(path))
        return path

    @classmethod
    def splits(cls, *args, data_directory='.', **kwargs):
        """
        Missing function definition can be found in `SeqInputOutputDataset`.
        """
        path = cls.download(data_directory=data_directory)
        return super().splits(*args, data_directory=path, **kwargs)


class SimpleQAObject(_SimpleQA):  # pragma: no cover
    """ Simple question answering derivative to predict the object in a question. """

    def __init__(self, path, question_field, object_mask_field, **kwargs):
        """
        Sample Data:
           Input: what language is angels vengeance in
           Output: c c c e e c
        """
        data = pd.read_table(path)
        data = data[data['Object EN Mask'].notnull()]
        fields = [('input', question_field), ('output', object_mask_field)]
        examples = []
        for _, row in data.iterrows():
            example = Example.fromlist([row['Question EN'].strip(), row['Object EN Mask']], fields)
            examples.append(example)
        super().__init__(examples, fields, **kwargs)


class SimpleQAFreebasePredicate(_SimpleQA):  # pragma: no cover
    """ Simple question answering derivative to predict the predicate in a question. """

    def __init__(self, path, question_field, predicate_field, **kwargs):
        """
        Sample Data:
           Input: what is the book e about?
           Output: www.freebase.com/book/written_work/subjects
        """
        data = pd.read_table(path)
        data = data[data['Freebase Property'].notnull()]
        fields = [('input', question_field), ('output', predicate_field)]
        examples = []
        for _, row in data.iterrows():
            example = Example.fromlist([row['Question EN'].strip(), row['Freebase Property']],
                                       fields)
            examples.append(example)
        super().__init__(examples, fields, **kwargs)


class SimpleQAWikiDataPredicate(_SimpleQA):  # pragma: no cover
    """ Simple question answering derivative to predict the predicate in a question. """

    def __init__(self, path, question_field, predicate_field, **kwargs):
        """
        Sample Data:
           Input: what country was the film the debt from
           Output: P495
        """
        data = pd.read_table(path)
        data = data[data['WikiData Property'].notnull()]
        data = data[~data['WikiData Property'].str.contains(':')]  # Ignore modifiers
        fields = [('input', question_field), ('output', predicate_field)]
        examples = []
        for _, row in data.iterrows():
            example = Example.fromlist([row['Question EN'].strip(), row['WikiData Property']],
                                       fields)
            examples.append(example)
        super().__init__(examples, fields, **kwargs)


class SimpleQAQuestionGeneration(_SimpleQA):  # pragma: no cover
    """ Simple question answering derivative to generate questions. """

    def __init__(self, path, triple_field, question_field, **kwargs):
        """
        Sample Data:
           Input: book/written_work/subjects | E
           Output: what is the book e about
        """
        data = pd.read_table(path)
        data = data[data['Subject EN'].notnull() & data['Object EN'].notnull()
                    & data['Freebase Property'].notnull()]
        data['Freebase Property'] = data.apply(
            lambda row: row['Freebase Property'].replace('www.freebase.com/', '').strip(), axis=1)
        fields = [('input', triple_field), ('output', question_field)]
        examples = []
        for _, row in data.iterrows():
            input_ = ' | '.join([row['Subject EN'], row['Object EN'], row['Freebase Property']])
            example = Example.fromlist([input_, row['Question EN'].strip()], fields)
            examples.append(example)
        super().__init__(examples, fields, **kwargs)


if __name__ == '__main__':  # pragma: no cover
    init_logging()
    _SimpleQA.download(data_directory='data/')
