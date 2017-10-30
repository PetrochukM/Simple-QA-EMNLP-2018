from collections import defaultdict

import numpy as np
import pandas as pd

from lib.observers.utils import df_to_string_option_context
from lib.observers.utils import iterate_batch
from lib.observers.observer import Metric
from lib.observers.utils import output_to_prediction


class ConfusionMatrix(Metric):
    """ Confusion matrix.

    TODO: Accuracy eval_batch and eval use standard python API. Can they be made faster with numpy
    or Tensor operators?

    Note:
        Slow to compute. Avoid using as a training metric.
    """

    def __init__(self,
                 output_field,
                 option_context=None,
                 ignore_index=None,
                 same_rows_columns=False):
        """
        Args:
            ignore_index (int, optional): specifies a target index that is ignored
            option_context: Pandas options context for display
              https://pandas.pydata.org/pandas-docs/stable/generated/pandas.option_context.html#pandas.option_context
            output_field (SeqField)
            same_vocab: Same rows and columns in the confusion matrix between target and output. 
        """
        self.confusion_matrix = defaultdict(dict)  # [target, dict[output, num_predictions]]
        self.prediction_keys = set()  # Rows
        self.target_keys = set()  # Columns
        self.ignore_index = ignore_index
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
            if self.ignore_index is not None:
                mask_arr = target.ne(self.ignore_index)
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
