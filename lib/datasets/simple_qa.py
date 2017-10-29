import os
import logging

from torchtext.data import Example

import pandas as pd
import sh

from seq2seq.config import get_root_path
from seq2seq.config import init_logging
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


class SimpleQAObjectRecognition(_SimpleQA):  # pragma: no cover
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


class SimpleQAFreebasePredicateRecognition(_SimpleQA):  # pragma: no cover
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


class SimpleQAWikiDataPredicateRecognition(_SimpleQA):  # pragma: no cover
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


class SimpleQAFrenchQuestionToPredicate(_SimpleQA):  # pragma: no cover

    filename_translation = 'en_fr_train_2.tsv'
    url_translation = 'hdfs://hdpb.prn.parsec.apple.com/user/mpetrochuk/en_fr_train_2.tsv'
    to_english_token = '2En '
    to_predicate_token = '2Predicate '

    @classmethod
    def download(cls, data_directory='.'):
        # Download multilingual data
        path = _SimpleQA.download(data_directory=data_directory)
        path_translation = os.path.join(path, cls.filename_translation)
        if not os.path.isfile(path_translation):
            # Download
            logger.info('Downloading %s', cls.url_translation)
            sh.hdfs('dfs', '-copyToLocal', cls.url_translation, path)
            logger.info('Downloaded %s', path_translation)
        return path, path_translation

    @classmethod
    def preprocess_property(cls, property_):
        return property_.split('/')[-1].replace('_', ' ')

    @classmethod
    def init_test(cls, filename, simple_qa_path, input_field, output_field, **kwargs):
        """
        Returns development or test data depending the filename.

        Data Sample:
            input: "2predicate Quelle est la nationalité de Howard storm?"
            output: "nationality"
        Args:
            filename: name of the file in the simple qa fr directory
            simple_qa_fr_path: path to simple qa fr directory
        """
        if filename is None:
            return None

        simple_qa_test_path = os.path.join(simple_qa_path, filename)
        data = pd.read_table(simple_qa_test_path)
        data = data[data['Question FR'].notnull() & data['Freebase Property'].notnull()]

        fields = [('input', input_field), ('output', output_field)]
        examples = []
        for _, row in data.iterrows():
            input_ = cls.to_predicate_token + row['Question FR']
            output = cls.preprocess_property(row['Freebase Property'])
            example = Example.fromlist([input_, output], fields)
            examples.append(example)

        return cls(examples, fields, **kwargs)

    @classmethod
    def init_train(cls, filename, simple_qa_path, en_fr_path, input_field, output_field, **kwargs):
        """
        Data Sample:
            input: "2en Les messages doivent porter le caractère d’affaire commercial."
            output: "An announcement must be commercial character."
            input: "2predicate who's a musician that plays jazz piano"
            output: "Jazz piano"
        Args:
            simple_qa_path: path to simple qa directory
        """
        if filename is None:
            return None

        simple_qa_train_path = os.path.join(simple_qa_path, filename)
        simple_qa = pd.read_table(simple_qa_train_path)
        simple_qa = simple_qa[simple_qa['Question EN'].notnull()
                              & simple_qa['Freebase Property'].notnull()]

        fields = [('input', input_field), ('output', output_field)]
        examples = []
        for _, row in simple_qa.iterrows():
            input_ = cls.to_predicate_token + row['Question EN'].strip()
            output = cls.preprocess_property(row['Freebase Property'])
            example = Example.fromlist([input_, output], fields)
            examples.append(example)

        en_fr = pd.read_table(en_fr_path)
        for _, row in en_fr.iterrows():
            input_ = cls.to_english_token + row['French']
            example = Example.fromlist([input_, row['English']], fields)
            examples.append(example)

        logger.info('Got %d `simple_qa` and %d `en_fr` rows', len(simple_qa), len(en_fr))

        return cls(examples, fields, **kwargs)

    @classmethod
    def splits(cls,
               input_field,
               output_field,
               data_directory='.',
               train='train.tsv',
               test='test.tsv',
               dev='dev.tsv',
               **kwargs):
        """
        Missing function definition can be found in `SeqInputOutputDataset`.
        """
        simple_qa_path, path_translation = cls.download(data_directory)

        train_data = cls.init_train(train, simple_qa_path, path_translation, input_field,
                                    output_field, **kwargs)
        dev_data = cls.init_test(dev, simple_qa_path, input_field, output_field, **kwargs)
        test_data = cls.init_test(test, simple_qa_path, input_field, output_field, **kwargs)

        return tuple(d for d in (train_data, dev_data, test_data) if d is not None)


if __name__ == '__main__':  # pragma: no cover
    init_logging()
    _SimpleQA.download(data_directory='data/')
