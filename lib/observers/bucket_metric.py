from collections import defaultdict

import numpy as np
import pandas as pd

from lib.observers.utils import df_to_string_option_context
from lib.observers.utils import iterate_batch
from lib.observers.observer import Metric
from lib.observers.accuracy import Accuracy
from lib.utils import get_variable_from_string
from lib.observers.utils import get_metrics
from lib.observers.utils import output_to_prediction
from lib.configurable import configurable

import seq2seq

# TODO: Batch should have a bucketing key


class BucketMetric(Metric):
    """ Bucketed a metric by the target sequence.

    TODO: Accuracy eval_batch and eval use standard python API. Can they be made faster with numpy
    or Tensor operators?
    Note:
        Slow to compute. Avoid using as a training metric.
        The bucket <unk> is when the target had OOV words because of a limited dictionary.
    """

    @configurable
    def __init__(self,
                 output_field,
                 input_field,
                 metric=None,
                 option_context=None,
                 ignore_index=None,
                 bucket_key='target',
                 **kwargs):
        """
        Args:
            output_field (SeqField)
            input_field (SeqField)
            metric (Metric or str, optional): Compute some metric class per bucket
            bucket_key (str or callable): given source, target, output compute a key for bucketing
            ignore_index (int, optional): specifies a target index that is ignored
            option_context (list, optional) Pandas options context for display
              https://pandas.pydata.org/pandas-docs/stable/generated/pandas.option_context.html#pandas.option_context
            **kwargs: key word arguments used for instantiating the next metric
        """
        self.get_bucket_key = get_variable_from_string(bucket_key, [seq2seq.metrics.bucket_metric])
        metric = Accuracy(ignore_index=ignore_index) if metric is None else metric
        metric = get_metrics(
            [metric],
            output_field=output_field,
            input_field=input_field,
            ignore_index=ignore_index,
            option_context=option_context,
            **kwargs)[0] if isinstance(metric, str) else metric
        if not isinstance(metric, Metric):
            raise ValueError('Metric must be must be a seq2seq.metrics.Metric')
        self.metric = metric
        self.bucket_metric = defaultdict(self.metric.deepcopy)
        self.output_field = output_field
        self.input_field = input_field
        self.option_context = option_context
        self.ignore_index = ignore_index
        super().__init__('Bucket Metric')

    def reset(self):
        self.bucket_metric = defaultdict(self.metric.deepcopy)

    def __str__(self):
        data, sequences, columns = self.get_measurement()
        df = pd.DataFrame(data, index=sequences, columns=columns)
        return '%s %s:\n%s\n' % (self.name, self.metric.name,
                                 df_to_string_option_context(df, self.option_context))

    def get_measurement(self):
        """
        Compute the Bucket metrics.

        Returns:
            data (numpy [1]): a 1D numpy array of the metric
            labels (list [N]): row bucket names
            columns (list [3]): column names
        """
        data = []
        keys = []  # List of bucket names
        columns = [self.metric.name]
        for i, key in enumerate(sorted(self.bucket_metric.keys())):
            keys.append(key)
            data.append(str(self.bucket_metric[key]))
        return data, keys, columns

    def eval_batch(self, outputs, batch):
        targets, _ = batch.output
        sources, _ = batch.input
        for source, target, output in iterate_batch(sources, targets, outputs):
            prediction = output_to_prediction(output)

            source_masked = source
            target_masked = target
            prediction_masked = prediction
            if self.ignore_index:
                # Mark target, output
                mask_arr = target.ne(self.ignore_index)
                target_masked = target.masked_select(mask_arr)
                prediction_masked = prediction.masked_select(mask_arr)

                # Mask source
                mask_arr = source.ne(self.ignore_index)
                source_masked = source.masked_select(mask_arr)

            key = self.get_bucket_key(
                source=source_masked,
                target=target_masked,
                prediction=prediction_masked,
                input_field=self.input_field,
                output_field=self.output_field)
            self.bucket_metric[key].eval(output, target)


# Some bucketing strategies


# TODO: Better document these functions. How to make a custom one? What does each one do?
def target(target, output_field, **kwargs):
    return output_field.decode(target.data)


def target_first_token(target, output_field, **kwargs):
    return output_field.decode(target.data[:1])


def source(source, input_field, **kwargs):
    return input_field.decode(source.data)


def source_first_token(source, input_field, **kwargs):
    return input_field.decode(source.data[:1])
