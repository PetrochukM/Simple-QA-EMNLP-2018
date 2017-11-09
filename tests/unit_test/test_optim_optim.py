import unittest

import torch
import mock

from seq2seq.optim import Optimizer


def get_random_parameters():
    return [torch.nn.Parameter(torch.randn(2, 3, 4))]


class TestOptimizer(unittest.TestCase):

    def test_init(self):
        optimizer = Optimizer('SGD')

    def test_set_parameters(self):
        learning_rate = 1
        optim = Optimizer('SGD', lr=learning_rate)
        optim.set_parameters(get_random_parameters())

        self.assertTrue(type(optim.optimizer) is torch.optim.SGD)
        self.assertEquals(optim.optimizer.param_groups[0]['lr'], learning_rate)

    def test_update(self):
        optim = Optimizer('SGD', lr=1, decay_after_epoch=5, lr_decay=0.5)
        optim.set_parameters(get_random_parameters())
        optim.update(0, 10)
        self.assertEquals(optim.optimizer.param_groups[0]['lr'], 0.5)

    @mock.patch("torch.nn.utils.clip_grad_norm")
    def test_step(self, mock_clip_grad_norm):
        optim = Optimizer('Adam', max_grad_norm=5)
        optim.set_parameters(get_random_parameters())
        optim.step()
        mock_clip_grad_norm.assert_called_once()

    def test_load_dict(self):
        optim = Optimizer('Adam', max_grad_norm=5)
        optim.set_parameters(get_random_parameters())
        state_dict = optim.state_dict()
        optim.load_state_dict(state_dict)
