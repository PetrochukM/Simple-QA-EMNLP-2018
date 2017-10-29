import copy
import logging

logger = logging.getLogger(__name__)

######################################################################
#   Metric Definitions
######################################################################


class Observer(object):

    def __init__(self, name=None):
        self.name = self.__class__.__name__ if name is None else name

    def reset(self):
        """ Reset the accumulated variables and output an observation. """
        raise NotImplementedError

    def update(self, input_batch, output_batch):
        """ Update the observer with a batch """
        raise NotImplementedError

    def clone(self):
        """ Clone the observer. """
        return copy.deepcopy(self)


class Observer(Metric):
    """ Base class for encapsulation of a metric.
    A Metric computes some measurement over some batches of data. In `eval_batch`, a metric
    accumulates data. In `get_measurement`, Metric returns a measurement.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        name (str): name of the metric used by logging messages.
    """
    COLUMNS = None  # Names of the the columns for every measurement value returned

    def __init__(self, name=None):
        self.name = self.__class__.__name__ if name is None else name

    def reset(self):
        """ Reset the accumulated variables. """
        raise NotImplementedError

    def __str__(self):
        """ Return a string representation of this metric.
        Must be in the form of `self.name: * \n`
        """
        return '%s: %s' % (self.name, str(self.get_measurement()))

    def get_measurement(self):
        """ Get the measurement.
        This method defines how to calculate the measurement proceeding evaluating a series of 
        batches. Override to define your own logic.
      
        Returns:
            measurement (float): value of the measurement.
        """
        raise NotImplementedError

    def eval_batch(self, outputs, batch):
        """ Evaluate and accumulate variables given outputs and expected results.
        This method is called after each batch with the batch outputs and the target (expected)
        results. Override it to define your own accumulation method.
    
        Args:
            outputs (torch.FloatTensor [seq_len, batch_size, dictionary_size]): outputs of a
                batch.
            # TODO
            batch (torch.LongTensor [seq_len, batch_size]): expected output of a batch.
        """
        raise NotImplementedError

    def eval(self, output, target):
        """ Evaluate and accumulate variables given outputs and expected results.
        This method is called after each batch with the batch outputs and the target (expected)
        results. Override it to define your own accumulation method.
    
        Args:
            output (torch.Tensor [seq_len, dictionary_size]): output of a batch.
            target (torch.Tensor [seq_len]): expected output of a batch.
        """
        raise NotImplementedError

    def deepcopy(self):
        return copy.deepcopy(self)
