from collections import defaultdict

import numpy as np
import pandas as pd

from seq2seq.metrics.utils import df_to_string_option_context
from seq2seq.metrics.utils import iterate_batch
from seq2seq.metrics.metric import Metric
from seq2seq.metrics.utils import output_to_prediction


class ConfusionMatrix(Metric):
    """ Confusion matrix.

    TODO: Accuracy eval_batch and eval use standard python API. Can they be made faster with numpy
    or Tensor operators?

    Note:
        Slow to compute. Avoid using as a training metric.
    """

    def __init__(self, output_field, option_context=None, mask=None, same_rows_columns=False):
        """
        Args:
            mask (int): index of masked token, i.e. weight[mask] = 0.
              For computing equality, a `masked_select` is used determined from target.
              With mask, this is not commutative.
            option_context: Pandas options context for display
              https://pandas.pydata.org/pandas-docs/stable/generated/pandas.option_context.html#pandas.option_context
            output_field (SeqField)
            same_vocab: Same rows and columns in the confusion matrix between target and output. 
        """
        self.confusion_matrix = defaultdict(dict)  # [target, dict[output, num_predictions]]
        self.prediction_keys = set()  # Rows
        self.target_keys = set()  # Columns
        self.mask = mask
        self.output_field = output_field  # torchtext.Vocab object used for __str__
        self.option_context = option_context
        self.same_rows_columns = same_rows_columns  # Same rows/columns in confusion matrix
        super().__init__('Confusion Matrix')

    def reset(self):
        self.confusion_matrix = defaultdict(dict)
        self.prediction_keys = set()
        self.target_keys = set()

    def __str__(self):
        confusion_matrix, prediction_sequences, target_sequences = self.get_measurement()
        df = pd.DataFrame(confusion_matrix, index=prediction_sequences, columns=target_sequences)
        return '%s:\n%s\n' % (self.name, df_to_string_option_context(df, self.option_context))

    def get_measurement(self):
        """
        Compute the 2D confusion matrix.
        
        Returns:
            data (numpy [N, M]): a 2D numpy array for the confusion matrix
            target_sequences (list [M]): names of the columns
            prediction_sequences (list [N]): names of the rows
        """
        self.target_keys = sorted(self.target_keys)
        self.prediction_keys = sorted(self.prediction_keys)

        # Two maps for efficient lookups
        target_keys_to_index = {k: i for i, k in enumerate(self.target_keys)}
        prediction_keys_to_index = {k: i for i, k in enumerate(self.prediction_keys)}

        # Table index
        target_sequences = [self.output_field.decode(k) for k in self.target_keys]
        # Table Columns
        prediction_sequences = [self.output_field.decode(k) for k in self.prediction_keys]

        # Run through the dict[target, dict[output, num_predictions]] setting numpy
        data = np.zeros((len(prediction_sequences), len(target_sequences)))  # Num Predictions
        for target_key in self.confusion_matrix:
            target_key_index = target_keys_to_index[target_key]
            for prediction_key in self.confusion_matrix[target_key]:
                prediction_key_index = prediction_keys_to_index[prediction_key]
                val = self.confusion_matrix[target_key][prediction_key]
                data[prediction_key_index][target_key_index] = val
        return data, prediction_sequences, target_sequences

    def eval_batch(self, outputs, batch):
        targets, _ = batch.output
        for output, target in iterate_batch(outputs, targets):
            prediction = output_to_prediction(output)

            # Mask part of the target and prediction sequence.
            if self.mask is not None:
                mask_arr = target.ne(self.mask)
                target = target.masked_select(mask_arr)
                prediction = prediction.masked_select(mask_arr)

            target_key = tuple(target.data.tolist())
            prediction_key = tuple(prediction.data.tolist())

            self.target_keys.add(target_key)
            self.prediction_keys.add(prediction_key)

            if prediction_key not in self.confusion_matrix[target_key]:
                self.confusion_matrix[target_key][prediction_key] = 1
            else:
                self.confusion_matrix[target_key][prediction_key] += 1

        # Ensure they have the same keys
        if self.same_rows_columns:
            self.target_keys.update(self.prediction_keys)
            self.prediction_keys.update(self.target_keys)
