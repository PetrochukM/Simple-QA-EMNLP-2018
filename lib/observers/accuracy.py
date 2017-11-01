import logging

from lib.observers.utils import iterate_batch
from lib.observers.utils import torch_equals_ignore_index
from lib.observers.utils import output_to_prediction
from lib.observers.observer import Observer

logger = logging.getLogger(__name__)


class Accuracy(Observer):
    """
    TODO: Accuracy eval_batch and eval use standard python API. Can they be made faster with numpy
    or Tensor operators?

    Usage Notes:
        - ignore_index should be the same as the loss ignore_index. If not, then your accuracy will
        be negatively affected by tokens loss does not optimize for like '<pad>'.
        - TokenAccuracy > Accuracy will not always be true. Token accuracy evaluates on special
        characters as well. For example extra EOS characters.
    Args:
        ignore_index (int, optional): specifies a target index that is ignored
    """

    def __init__(self, ignore_index=None, name='Accuracy'):
        super().__init__(name)
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.num_correct_predictions = 0.0
        self.total_predictions = 0.0
        return self

    def observed(self, save_directory=None):
        """ Save directory is not used """
        if self.total_predictions == 0:
            accuracy = None
        else:
            accuracy = self.num_correct_predictions / self.total_predictions
        return accuracy

    def dump(self):
        logger.info('%s: %s [%d of %d]', self.name,
                    self.observed(), self.num_correct_predictions, self.total_predictions)
        return self

    def update(self, event):
        if 'target_batch' in event and 'output_batch' in event:
            for target_tensor, output_tensor in iterate_batch(event['target_batch'],
                                                              event['output_batch']):
                self.update({'target_tensor': target_tensor, 'output_tensor': output_tensor})
        elif 'target_tensor' in event and 'output_tensor' in event:
            prediction_tensor = output_to_prediction(event['output_tensor'])
            if torch_equals_ignore_index(
                    event['target_tensor'].data, prediction_tensor.data,
                    ignore_index=self.ignore_index):
                self.num_correct_predictions += 1
            self.total_predictions += 1
        return self
