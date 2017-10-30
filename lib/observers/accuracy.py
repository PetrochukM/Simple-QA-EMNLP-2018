from lib.observers.utils import iterate_batch
from lib.observers.utils import torch_equals_ignore_index
from lib.observers.utils import output_to_prediction
from lib.observers.observer import Observer


class Accuracy(Observer):
    """
    TODO: Accuracy eval_batch and eval use standard python API. Can they be made faster with numpy
    or Tensor operators?

    Note:
        ignore_index should be the same as the loss ignore_index. If not, then your accuracy will be
        negatively affected by tokens loss does not optimize for like '<pad>'.

        TokenAccuracy > Accuracy will not always be true. Token accuracy evaluates on special
        characters as well. For example extra EOS characters.
    Args:
        ignore_index (int, optional): specifies a target index that is ignored
    """

    def __init__(self, ignore_index=None, name='Accuracy'):
        self.num_correct_predictions = 0.0
        self.total_predictions = 0.0
        self.ignore_index = ignore_index
        super().__init__(name)

    def reset(self):
        self.num_correct_predictions = 0.0
        self.total_predictions = 0.0

    def __str__(self):
        return '%s: %s [%d of %d]' % (tuple([
            self.name, self.get_measurement(), self.num_correct_predictions, self.total_predictions
        ]))

    def get_measurement(self):
        if self.total_predictions == 0:
            accuracy = None
        else:
            accuracy = self.num_correct_predictions / self.total_predictions

        return accuracy

    def update(self, input_batch, output_batch):
        targets, _ = input_batch.output
        for output, target in iterate_batch(output_batch, targets):
            self._update(output, target)

    def _update(self, output, target):
        prediction = output_to_prediction(output)
        if torch_equals_ignore_index(target.data, prediction.data, ignore_index=self.ignore_index):
            self.num_correct_predictions += 1
        self.total_predictions += 1
