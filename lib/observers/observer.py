import copy
import logging

logger = logging.getLogger(__name__)

######################################################################
#   Metric Definitions
######################################################################


class Observer(object):
    """ Base class for encapsulation of a Observer.

    Observer observers batches of data. With `update_batch` a Observer accumulates data. With
    `reset` Observer resets and returns an observation.

    Args:
        name (str): name of the metric used by logging messages.
    """

    def __init__(self, name=None):
        self.name = self.__class__.__name__ if name is None else name

    def __str__(self):
        """ Return a string representation of this metric. Must be in the form of `self.name: * \n`.
        """
        return '%s: %s' % (self.name, str(self.reset()))

    def reset(self):
        """ Reset the accumulated variables and output an observation. """
        raise NotImplementedError

    def update_batch(self, input_batch, output_batch):
        """ Evaluate and accumulate variables given outputs and expected results.

        This method is called after each batch with the batch outputs and the target (expected)
        results. Override it to define your own accumulation method.
    
        Args:
            outputs (torch.FloatTensor [seq_len, batch_size, dictionary_size]): outputs of a
                batch.
            batch (torch.LongTensor [seq_len, batch_size]): expected output of a batch.
        """
        # TODO: By default, call update and iterate through batch 
        raise NotImplementedError

    def update(self, input, output):
        """ Evaluate and accumulate variables given outputs and expected results.

        This method is called after each batch with the batch outputs and the target (expected)
        results. Override it to define your own accumulation method.

        # TODO: Update this input/output
    
        Args:
            output (torch.Tensor [seq_len, dictionary_size]): output of a batch.
            target (torch.Tensor [seq_len]): expected output of a batch.
        """
        raise NotImplementedError

    def clone(self):
        """ Clone the observer. """
        return copy.deepcopy(self)
