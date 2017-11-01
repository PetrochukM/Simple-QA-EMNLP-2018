import random
import logging

import pandas as pd

from lib.observers.observer import Observer
from lib.observers.utils import output_to_prediction
from lib.configurable import configurable
from lib.observers.utils import torch_equals_ignore_index

logger = logging.getLogger(__name__)


class RandomSample(Observer):
    """ Print random samples during training and evaluation """

    @configurable
    def __init__(self, input_text_encoder, output_text_encoder, n_samples=1, ignore_index=None):
        self.output_text_encoder = output_text_encoder
        self.input_text_encoder = input_text_encoder
        self.reset()
        self.n_samples = n_samples
        self.ignore_index = ignore_index
        super().__init__('Random Sample')

    def reset(self):
        self.positive_samples = []
        self.negative_samples = []
        self.positive_row_count = 0
        self.negative_row_count = 0
        return self

    def dump(self):
        ret = '%s\n' % self.name

        for prefix, sample in [('Positive', self.positive_samples), ('Negative',
                                                                     self.negative_samples)]:
            data = []
            for source, target, prediction in sample:
                data.append([
                    self.input_text_encoder.decode(source), self.output_text_encoder.decode(target),
                    self.output_text_encoder.decode(prediction)
                ])
            ret += '\n%s Samples:\n%s\n' % (prefix, pd.DataFrame(
                data, columns=['Source', 'Target', 'Prediction']))
        logger.info(ret)
        return self

    def observed(self):
        return self.positive_samples, self.negative_samples

    def update(self, event):
        if 'target_batch' not in event or 'output_batch' not in event or 'source_batch' not in event:
            return

        batch_size = event['output_batch'].size(1)
        # seq_len x batch_size x dictionary_size -> batch_size x seq_len x dictionary_size
        output_batch = event['output_batch'].data.transpose(0, 1)
        # seq len x batch size -> batch size x seq len
        target_batch = event['target_batch'].data.transpose(0, 1)
        # seq len x batch size -> batch size x seq len
        source_batch = event['source_batch'].data.transpose(0, 1)

        def sample(dest, row_count, item):
            # Args: dest (list) to add sample too
            # Sample from a list of unknown size
            # https://en.wikipedia.org/wiki/Reservoir_sampling
            if len(dest) < self.n_samples:
                dest.append(item)
                if len(dest) == self.n_samples:  # Filled up for the first time
                    random.shuffle(dest)
            else:
                rand_int = random.randint(0, row_count)
                if rand_int < self.n_samples:
                    dest[rand_int] = item

        for i in range(batch_size):
            prediction = output_to_prediction(output_batch[i])
            item = tuple([source_batch[i], target_batch[i], prediction])
            if torch_equals_ignore_index(
                    target_batch[i], prediction, ignore_index=self.ignore_index):
                sample(self.positive_samples, self.positive_row_count, item)
                self.positive_row_count += 1
            else:
                sample(self.negative_samples, self.negative_row_count, item)
                self.negative_row_count += 1
        return self
