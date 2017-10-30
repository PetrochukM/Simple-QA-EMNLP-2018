from lib.observers.observer import Metric
from lib.observers.utils import flatten_batch


class TokenAccuracy(Metric):
    """
    Note:
        TokenAccuracy > Accuracy will not always be true. Token accuracy evaluates on special
        characters as well. For example extra EOS characters.

        Mask should be the same as the loss mask. If not, then your accuracy will be negatively
        affected by tokens loss does not optimize for like '<pad>'.
    Args:
        ignore_index (int, optional): specifies a target index that is ignored
    """

    COLUMNS = ['Accuracy', 'Num Correct', 'Total']

    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index
        self.num_correct_tokens = 0.0
        self.total_tokens = 0.0
        super().__init__('Token Accuracy')

    def reset(self):
        self.num_correct_tokens = 0.0
        self.total_tokens = 0.0

    def __str__(self):
        return '%s: %s [%d of %d]' % (tuple([self.name] + list(self.get_measurement())))

    def get_measurement(self):
        if self.total_tokens == 0:
            accuracy = None
        else:
            accuracy = self.num_correct_tokens / self.total_tokens
        # NOTE: Must return in the same order as COLUMNS
        return accuracy, self.num_correct_tokens, self.total_tokens

    def eval_batch(self, outputs, batch):
        targets, _ = batch.output
        output, target, _ = flatten_batch(outputs, targets)
        self.eval(output, target)

    def eval(self, output, target):
        predictions = output.max(1)[1]
        if self.ignore_index:
            masked_arr = target.data.ne(self.ignore_index)
            self.num_correct_tokens += predictions.data.eq(
                target.data).masked_select(masked_arr).sum()
            self.total_tokens += masked_arr.sum()
        else:
            self.total_tokens += len(target)
            self.num_correct_tokens += predictions.data.eq(target.data).sum()
