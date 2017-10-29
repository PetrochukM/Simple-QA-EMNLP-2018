from seq2seq.metrics.utils import iterate_batch
from seq2seq.metrics.utils import torch_equals_mask
from seq2seq.metrics.utils import output_to_prediction
from seq2seq.metrics.metric import Metric


class Accuracy(Metric):
    """
    TODO: Accuracy eval_batch and eval use standard python API. Can they be made faster with numpy
    or Tensor operators?

    Note:
        Mask should be the same as the loss mask. If not, then your accuracy will be negatively
        affected by tokens loss does not optimize for like '<pad>'.

        TokenAccuracy > Accuracy will not always be true. Token accuracy evaluates on special
        characters as well. For example extra EOS characters.
    Args:
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """

    def __init__(self, mask=None, name='Accuracy'):
        self.num_correct_predictions = 0.0
        self.total_predictions = 0.0
        self.mask = mask
        super().__init__(name)

    def reset(self):
        self.num_correct_predictions = 0.0
        self.total_predictions = 0.0

    def __str__(self):
        return '%s: %s [%d of %d]' % (tuple([
            self.name,
            self.get_measurement(), self.num_correct_predictions, self.total_predictions
        ]))

    def get_measurement(self):
        if self.total_predictions == 0:
            accuracy = None
        else:
            accuracy = self.num_correct_predictions / self.total_predictions

        return accuracy

    def eval_batch(self, outputs, batch):
        targets, _ = batch.output
        for output, target in iterate_batch(outputs, targets):
            self.eval(output, target)

    def eval(self, output, target):
        prediction = output_to_prediction(output)
        if torch_equals_mask(target.data, prediction.data, mask=self.mask):
            self.num_correct_predictions += 1
        self.total_predictions += 1
