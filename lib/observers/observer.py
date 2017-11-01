import copy
import logging

logger = logging.getLogger(__name__)


class Observer(object):
    """ Base class for encapsulation of a Observer.

    Observer observers batches of data. With `reset` dump resets and returns an observation.

    Args:
        name (str): name of the Observer used by logging messages.
    """

    def __init__(self, name=None):
        self.name = self.__class__.__name__ if name is None else name

    def reset(self):
        """ Reset the accumulated variables. """
        raise NotImplementedError

    def dump(self, params):
        """ Output an observation to the `save_directory` or to console depending on the observable.

        Args:
            params (dict): arbitrary parameters used for saving the observation.
        """
        raise NotImplementedError

    def observed(self):
        """ Get the observation.
        """
        raise NotImplementedError

    def update(self, event):
        """ Update the state of the `Observer`.

        Args:
            event (dict): arbitrary dictionary that every observer accepts
        """
        raise NotImplementedError

    def clone(self):
        """ Clone the observer. """
        return copy.deepcopy(self)
