from lib.observers.accuracy import Accuracy
from lib.configurable import configurable


class AccuracyTopN(Accuracy):
    """
    TODO: Accuracy eval_batch and eval use standard python API. Can they be made faster with numpy
    or Tensor operators?

    Note:
        Ignore_index should be the same as the loss ignore_index. If not, then your accuracy will be negatively
        affected by tokens loss does not optimize for like '<pad>'.

        TokenAccuracy > Accuracy will not always be true. Token accuracy evaluates on special
        characters as well. For example extra EOS characters.
    Args:
        ignore_index (int, optional): specifies a target index that is ignored
        top_n (int): number of predictions to consider
    """

    @configurable
    def __init__(self, ignore_index=None, top_n=5):
        self.top_n = top_n
        super().__init__(ignore_index, name='Accuracy Top %d' % self.top_n)

    def eval(self, output, target):
        predictions = output.topk(self.top_n, dim=1)[1]
        if self.ignore_index is not None:
            mask_arr = target.ne(self.ignore_index)
            target = target.masked_select(mask_arr)
        correct_indexes = set()
        for i in range(self.top_n):
            prediction = predictions[:, i]

            if self.ignore_index is not None:
                prediction = prediction.masked_select(mask_arr)

            for i in range(target.data.size()[0]):
                if target.data[i] == prediction.data[i]:
                    correct_indexes.add(i)

        if len(correct_indexes) == target.data.size()[0]:
            self.num_correct_predictions += 1

        self.total_predictions += 1
