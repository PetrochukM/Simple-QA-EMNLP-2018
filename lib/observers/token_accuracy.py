from seq2seq.metrics.metric import Metric
from seq2seq.metrics.utils import flatten_batch


class TokenAccuracy(Metric):
    """
    Note:
        TokenAccuracy > Accuracy will not always be true. Token accuracy evaluates on special
        characters as well. For example extra EOS characters.

        Mask should be the same as the loss mask. If not, then your accuracy will be negatively
        affected by tokens loss does not optimize for like '<pad>'.
    Args:
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
    """

    COLUMNS = ['Accuracy', 'Num Correct', 'Total']

    def __init__(self, mask=None):
        self.mask = mask
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
        if self.mask:
            mask = target.data.ne(self.mask)
            self.num_correct_tokens += predictions.data.eq(target.data).masked_select(mask).sum()
            self.total_tokens += mask.sum()
        else:
            self.total_tokens += len(target)
            self.num_correct_tokens += predictions.data.eq(target.data).sum()
