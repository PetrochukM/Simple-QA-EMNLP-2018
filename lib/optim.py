import torch

from seq2seq.config import configurable

# TODO: Replace components with PyTorch scheduler


class Optimizer(object):
    """ The Optimizer class encapsulates torch.optim package and provide:
        - learning rate scheduling
        - gradient norm clipping

    @TODO: 'decay_after_epoch' is an abstract optimization technique. Consider
           abstracting it away in its own class?.

    @TODO: Move default kwargs value into config.

    Args:
        optim_class (torch.optim.Optimizer):
            Optimization method. Find out all the optimization methods here:
            http://pytorch.org/docs/master/optim.html. Based on Nils et al., 2017, we recommend
            Adam. <https://arxiv.org/pdf/1707.06799v1.pdf>

        max_grad_norm (float, optional):
            Threshhold value for gradient normalization.
            Set to 0 to disable (default 1)
            Based on Nils et al., 2017, we recommend 1.0.
            <https://arxiv.org/pdf/1707.06799v1.pdf>

        lr_decay (float, optional):
            Value for learning rate decay:
            lr = lr_decay * lr (default 1)

        decay_after_epoch (float, optional):
            Learning rate starts to decay after the specified epoch number.
            Set 0 to disable (default 0)

        lr (float, optional):
            Starting learning rate. If adagrad/adadelta/adam is used, then this is the global
            learning rate. Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001.

        **kwargs:
            Arguments for the given optimizer class.
            Refer to http://pytorch.org/docs/optim.html#algorithms
    """

    # Supported arguments string from **kwargs
    _ARG_MAX_GRAD_NORM = "max_grad_norm"
    _ARG_DECAY_AFTER = "decay_after_epoch"
    _ARG_LR_DECAY = "lr_decay"
    _ARG_LR = "lr"

    @configurable
    def __init__(self, optim_class_name, **kwargs):
        self.optim_class = getattr(torch.optim, optim_class_name)
        self.optimizer = None
        self.parameters = None

        self.max_grad_norm = self._get_remove(kwargs, Optimizer._ARG_MAX_GRAD_NORM, 1)
        self.lr_decay = self._get_remove(kwargs, Optimizer._ARG_LR_DECAY, 1)
        self.decay_after_epoch = self._get_remove(kwargs, Optimizer._ARG_DECAY_AFTER, 0)
        self.optim_args = kwargs

    def _get_remove(self, args, key, default):
        value = default
        if key in args:
            value = args[key]
            del args[key]
        return value

    def set_parameters(self, parameters):
        """ Set the parameters to optimize.

        Args:
            parameters (iterable): An iterable of torch.nn.Parameter.
        """
        self.parameters = filter(lambda p: p.requires_grad, parameters)
        self.optimizer = self.optim_class(self.parameters, **self.optim_args)
        self.load_state_dict = self.optimizer.load_state_dict
        self.state_dict = self.optimizer.state_dict

    def step(self):
        """ Performs a single optimization step, including gradient norm clipping if necessary. """
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm(self.parameters, self.max_grad_norm)
        self.optimizer.step()

    def update(self, loss, epoch):
        """ Update the learning rate if the conditions are met. Override this method
        to implement your own learning rate schedule.

        Args:
            loss (float):
                The current loss. It could be training loss or developing loss
                depending on the caller. By default the supervised trainer uses
                developing loss.

            epoch (int):
                The current epoch number.
        """
        after_decay_epoch = self.decay_after_epoch != 0 and epoch >= self.decay_after_epoch
        if after_decay_epoch:
            self.optimizer.param_groups[0]['lr'] *= self.lr_decay
