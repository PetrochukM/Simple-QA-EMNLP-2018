import math
import logging

from seq2seq.metrics.metric import Metric

import seq2seq

logger = logging.getLogger(__name__)


class Loss(Metric):
    """
    This class defines interfaces that are commonly used with loss functions
    in training and inference.  For information regarding individual loss
    functions, please refer to http://pytorch.org/docs/master/nn.html#loss-functions
    Args:
        name (str): Name of the loss function used by logging messages.
        criterion (torch.nn._Loss): One of PyTorch's loss function.
            Refer to http://pytorch.org/docs/master/nn.html#loss-functions for
            a list of them.
    Attributes:
        accumulated_loss (int or torch.nn.Tensor): Variable that stores accumulated loss
    """

    def __init__(self, criterion):
        super().__init__(criterion.__class__.__name__)
        self.criterion = criterion
        self.size_average = self.criterion.size_average
        self.reset()

    def reset(self):
        """ Reset the accumulated loss. """
        self.accumulated_loss = 0
        self.count_batches = 0

    def get_perplexity(self):
        return math.exp(min(self.get_loss(), 100))

    def get_measurement(self):
        """
        Alias to get_loss
        """
        return self.get_loss()

    def get_loss(self):
        """ Get the loss.
        This method defines how to calculate the averaged loss given the
        accumulated loss and the normalization term.  Override to define your
        own logic.
        Returns:
            loss (float): value of the loss.
        """
        if self.count_batches == 0:
            return None
        elif self.size_average:
            return self.accumulated_loss.data[0] / self.count_batches
        else:
            return self.accumulated_loss.data[0]

    def eval_batch(self, outputs, batch):
        targets, _ = batch.output
        outputs, targets, batch_size = seq2seq.metrics.utils.flatten_batch(outputs, targets)
        self.accumulated_loss += self.criterion(outputs, targets)
        self.count_batches += 1

    def cuda(self):
        self.criterion.cuda()

    def backward(self):
        if self.count_batches == 0:
            raise ValueError("No loss to back propagate.")
        self.accumulated_loss.backward()
