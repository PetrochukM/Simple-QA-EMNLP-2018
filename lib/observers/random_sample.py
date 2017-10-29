import random

import pandas as pd

from seq2seq.metrics.metric import Metric
from seq2seq.metrics.utils import output_to_prediction
from seq2seq.config import configurable
from seq2seq.metrics.utils import torch_equals_mask


class RandomSample(Metric):
    """ Print random samples during training and evaluation """

    @configurable
    def __init__(self, output_field, input_field, n_samples=1, mask=None):
        self.output_field = output_field
        self.input_field = input_field
        self.reset()
        self.n_samples = n_samples
        self.mask = mask
        super().__init__('Random Sample')

    def reset(self):
        self.positive_samples = []
        self.negative_samples = []
        self.positive_row_count = 0
        self.negative_row_count = 0

    def __str__(self):

        ret = '%s\n' % self.name

        for prefix, sample in [('Positive', self.positive_samples), ('Negative',
                                                                     self.negative_samples)]:
            data = []
            for source, target, prediction in sample:
                data.append([
                    self.input_field.denumeralize(source),
                    self.output_field.denumeralize(target),
                    self.output_field.denumeralize(prediction)
                ])
            ret += '\n%s Samples:\n%s\n' % (
                prefix, pd.DataFrame(data, columns=['Source', 'Target', 'Prediction']))
        return ret

    def get_measurement(self):
        return self.positive_samples, self.negative_samples

    def eval_batch(self, outputs, batch):
        batch_size = outputs.size(1)
        targets, _ = batch.output
        sources, _ = batch.input
        # seq_len x batch_size x dictionary_size -> batch_size x seq_len x dictionary_size
        outputs = outputs.data.transpose(0, 1)
        # seq len x batch size -> batch size x seq len
        targets = targets.data.transpose(0, 1)
        # seq len x batch size -> batch size x seq len
        sources = sources.data.transpose(0, 1)

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
            prediction = output_to_prediction(outputs[i])
            item = tuple([sources[i], targets[i], prediction])
            if torch_equals_mask(targets[i], prediction, mask=self.mask):
                sample(self.positive_samples, self.positive_row_count, item)
                self.positive_row_count += 1
            else:
                sample(self.negative_samples, self.negative_row_count, item)
                self.negative_row_count += 1
